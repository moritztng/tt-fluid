#!/bin/bash
# vrt-fluid: one rented-GPU SPUMA session on the occDrivAer mid mesh (5,564,064 cells), from cfd-baseline session65.sh.
# Ships spumaM + spumaS bundles, both bound() fix libs and the packed case from .110 ~/vrt-fluid/mid/gpu/ship, runs
# occ_plan.sh (SPDP <iters> graded + fp64 800), pulls every 10 min, destroys the box and confirms it gone; guard at the cap.
# Usage: session_mid.sh <out-dir> <iters> <pool-GiB> <max-$/h> <gpu-name> <min-gpu-MiB> <hard-cap-hours>
set -u
OUT=$1; IT=$2; POOL=$3; MAXP=$4; GPU=$5; MINMEM=$6; CAPH=$7
G=ubuntu@172.16.102.110; B=vrt-fluid/mid/gpu/ship; CASE=occmid_np1; CP=/home/ubuntu/vrt-fluid/mid/gpu/stage; X2="spumaS_rt.tar.zst libspumaBoundFix_S.so"
RT=spumaM_rt.tar.zst; BF=libspumaBoundFix.so; RTN=$(basename $RT)
K=$(cat ~/.config/vastai/vast_api_key); API=https://console.vast.ai/api/v0
BUDGET=~/.coworker/state/vast-budget
HERE=$(cd $(dirname $0) && pwd)
mkdir -p $OUT; exec >> $OUT/driver.log 2>&1
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
api() { curl -s -H "Authorization: Bearer $K" "$@"; }
ID=""; GUARD=""
destroy() {
  [ -n "$GUARD" ] && kill $GUARD 2>/dev/null
  [ -z "$ID" ] && return
  api -X DELETE $API/instances/$ID/; echo
  sleep 15
  if api $API/instances/ | python3 -c "import json,sys; sys.exit(any(i['id']==$ID for i in json.load(sys.stdin)['instances']))"; then
    stamp "DESTROYED $ID, confirmed gone"
    echo "#$ID DESTROYED $(date -u +%FT%TZ) ($1), confirmed gone" >> $BUDGET
    ID=""
  else
    stamp "DESTROY of $ID NOT CONFIRMED"
  fi
}
trap 'destroy trap' EXIT

ssh -o BatchMode=yes $G "grep -q PACK-DONE /home/ubuntu/vrt-fluid/mid/gpu/pack.status" || { stamp "mid case not packed on .110"; exit 1; }

# The bundle carries its own CUDA 13.3 runtime + compat libcuda, so the small base image is enough (a 5 GB devel
# pull left a KR box "loading" for 20 min). Up to 3 offers, 12 min each to come up; a stalled one is destroyed.
VQ=${VERIFIED:-'"verified":{"eq":true}'}
SKIP="43751089"  # KR host stays "loading" on every image (2026-10-09, twice)
for try in 1 2 3; do
stamp "rent: $GPU >= $MINMEM MiB, <= \$$MAXP/h (try $try)"
OFFER=$(api -G $API/bundles/ --data-urlencode "q={\"gpu_name\":{\"eq\":\"$GPU\"},\"num_gpus\":{\"eq\":${NGPU:-1}},\"gpu_ram\":{\"gte\":$MINMEM},\"rentable\":{\"eq\":true},$VQ,\"disk_space\":{\"gte\":${DISK:-100}},\"inet_down\":{\"gte\":500},\"reliability2\":{\"gte\":0.97},\"dph_total\":{\"lte\":$MAXP},\"order\":[[\"dph_total\",\"asc\"]],\"type\":\"on-demand\"}" \
  | python3 -c "import json,sys; o=[x for x in json.load(sys.stdin)['offers'] if str(x['id']) not in '$SKIP'.split()]; print(o[0]['id'], round(o[0]['dph_total'],3), o[0]['geolocation']) if o else print('')")
[ -z "$OFFER" ] && { stamp "no offer"; exit 1; }
stamp "offer $OFFER"
set -- $OFFER; SKIP="$SKIP $1"
ID=$(api -X PUT -H 'Content-Type: application/json' $API/asks/$1/ -d "{\"client_id\":\"me\",\"image\":\"nvidia/cuda:12.6.3-base-ubuntu24.04\",\"disk\":${DISK:-100},\"runtype\":\"ssh_direc ssh_proxy\",\"label\":\"vrt-fluid\"}" \
  | python3 -c "import json,sys; print(json.load(sys.stdin).get('new_contract',''))")
[ -z "$ID" ] && { stamp "create failed"; continue; }
echo "$ID  $CAPH  vrt-fluid   # created $(date -u +%FT%TZ), session occDrivAer mid SPUMA, $GPU offer $1 \$$2/h" >> $BUDGET
stamp "created $ID"; CREATED=$(date +%s)
( sleep $(python3 -c "print(int($CAPH*3600))"); curl -s -X DELETE -H "Authorization: Bearer $K" $API/instances/$ID/ \
  && echo "#$ID DESTROYED $(date -u +%FT%TZ) (hard-cap guard)" >> $BUDGET ) > /dev/null 2>&1 &
GUARD=$!

for i in $(seq 36); do
  read -r HOST PORT ST < <(api $API/instances/$ID/ | python3 -c "
import json,sys; d=json.load(sys.stdin); i=d.get('instances',d); p=(i.get('ports') or {}).get('22/tcp')
print(i.get('public_ipaddr','').strip() if p else i.get('ssh_host'), p[0]['HostPort'] if p else i.get('ssh_port'), i.get('actual_status'))")
  [ "$ST" = running ] && ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -p $PORT root@$HOST true 2>/dev/null && break
  sleep 20
done
[ "$ST" = running ] && break
stamp "never came up ($ST)"; destroy stalled; [ -n "$ID" ] && exit 1
done
[ -n "$ID" ] || { stamp "no box came up"; exit 1; }
S="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p $PORT root@$HOST"
stamp "up $HOST:$PORT"
ssh -o BatchMode=yes $G cat .ssh/cfd_vast.pub | $S 'echo >> ~/.ssh/authorized_keys; cat >> ~/.ssh/authorized_keys; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv'
$S 'nohup bash -c "apt-get update -qq && apt-get install -y -qq curl zstd python3 openssl >/dev/null && curl -s https://dl.openfoam.com/add-debian-repo.sh | bash >/dev/null && apt-get install -y -qq openfoam2512 >/dev/null; echo APT-DONE" > /root/apt.out 2>&1 &'
$S 'mkdir -p /root/s65'
if [ -n "${STAGE:-}" ]; then
# STAGE = HF dataset made by stage_hf.sh: the box pulls the encrypted payload from the CDN, the key goes over ssh.
stamp "fetch from hf:$STAGE"
FILES=$(ssh -o BatchMode=yes $G "cd $B && basename -a $RT $BF $X2 part.* | tr '\n' ' '; du -cbL $RT $BF $X2 part.* | tail -1 > /tmp/ship.size" && ssh -o BatchMode=yes $G cat /tmp/ship.size > $OUT/ship.size)
$S 'umask 077; cat > /root/.k' < ~/.coworker/artifacts/cfd-baseline/stage.key
timeout 1200 $S "cd /root/s65 && until command -v curl && command -v openssl; do sleep 5; done > /dev/null; \
  for f in $FILES; do ( curl -sfL --retry 5 https://huggingface.co/datasets/$STAGE/resolve/main/\$f.enc | openssl enc -d -aes-256-ctr -pbkdf2 -pass file:/root/.k -out \$f ) & done; wait; rm -f /root/.k" || stamp "fetch rc=$?"
else
stamp "ship from .110"
# One scp per part in parallel; a dropped stream ("lost connection") is retried, then sizes are compared.
ssh -o BatchMode=yes $G "cd $B && C='scp -q -i ~/.ssh/cfd_vast -o BatchMode=yes -o StrictHostKeyChecking=no -P $PORT' && \
  for f in $RT $BF $X2 part.*; do ( for t in 1 2 3; do \$C \$f root@$HOST:/root/s65/ && break; sleep 5; done ) & done; wait; \
  du -cbL $RT $BF $X2 part.* | tail -1" > $OUT/ship.size || stamp "ship rc=$?"
fi
WANT=$(tail -1 $OUT/ship.size | cut -f1); GOT=$($S "du -cb /root/s65/$RTN /root/s65/libspumaBoundFix.so /root/s65/spumaS_rt.tar.zst /root/s65/libspumaBoundFix_S.so /root/s65/part.* | tail -1" | tail -1 | cut -f1)
stamp "shipped $GOT of $WANT bytes"
[ -n "$WANT" ] && [ "$GOT" = "$WANT" ] || { stamp "ship incomplete"; exit 1; }
scp -q -o StrictHostKeyChecking=no -P $PORT $HERE/spuma_run.sh $HERE/occ_plan.sh $HERE/fvSolution.spuma.occ root@$HOST:/root/s65/
stamp "unpack"
$S 'cd /root/s65 && ls -la && until grep -q APT-DONE /root/apt.out; do sleep 10; done && tar xf '$RTN' -I zstd -C / && tar xf spumaS_rt.tar.zst -I zstd -C / && cat part.* | zstd -dq | tar xf - && rm part.* && du -sh '$CASE || { stamp "unpack failed"; exit 1; }
stamp "run plan: $IT it, pool $POOL GiB"
# The plan runs detached on the box and is pulled every 10 min, so a dropped ssh or a host that falls off the network
# (55119115, 00:34Z 10-10) loses at most 10 minutes of log, not the session.
# The launch ssh is capped at 60 s: on 10-10 it never returned (2.8 h), so the driver never pulled; the box plan is setsid and outlives it.
# The deadline counts from create, like the guard, less 15 min for the last pull and the destroy.
DL=$(python3 -c "print(int($CAPH*3600) - ($(date +%s) - $CREATED) - ${TAIL:-900})"); END=$(( $(date +%s) + DL ))
timeout 60 $S "cd /root/s65 && nohup setsid bash -c 'bash occ_plan.sh /root/s65/$CASE $IT $POOL $DL > plan.out 2>&1; echo PLAN-DONE >> plan.out' > /dev/null 2>&1 < /dev/null &"
pull() { timeout 300 $S "cd /root/s65 && tar czf - plan.out run_*.out /root/apt.out $CASE/log.simpleFoam* $CASE/gpu.csv* $CASE/iters.tsv* $CASE/summary.txt* $CASE/postProcessing* $CASE/system $CASE/diag.txt $CASE/*/uniform/profiling prof_* ${PULLX:-} 2>/dev/null" | tar xzf - -C $OUT; }
while [ $(date +%s) -lt $END ]; do
  sleep 600; pull
  grep -q PLAN-DONE $OUT/plan.out 2>/dev/null && break
  stamp "progress: $(grep -a '^Time = ' $OUT/$CASE/log.simpleFoam 2>/dev/null | tail -1)"
done
stamp "pull"
pull; tail -5 $OUT/plan.out
destroy done
stamp "SESSION65-DONE"
