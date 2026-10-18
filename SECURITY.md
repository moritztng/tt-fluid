# Security policy

## Supported versions

Security fixes go into the latest release.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private vulnerability reporting instead
(the repository's **Security** tab, then **Report a vulnerability**). You will get an answer within a week.

tt-fluid reads OpenFOAM case files and Python pickles that `ttfluid.device.run_mesh --save` and `--ref` write. Treat
those pickles like code: load only files you produced yourself, never ones from an untrusted source.
