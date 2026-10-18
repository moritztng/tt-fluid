/*---------------------------------------------------------------------------*\
    Drop-in replacement for Foam::bound(volScalarField&, const dimensionedScalar&)
    for SPUMA (OpenFOAM v2412 GPU port), loaded with LD_PRELOAD.

    SPUMA's device min()/max() of a field (cudaExecutor reductionCompare) misses
    negative cells, so bound() never fires after the first iterations and a
    negative omega reaches the SST model, which turns it into NaN. This file
    keeps upstream bound() and takes the field minimum with a thrust reduction.

    License: GPL-3.0-or-later (derived from OpenFOAM's bound.C).
\*---------------------------------------------------------------------------*/

#include "bound.H"
#include "volFields.H"
#include "fvc.H"

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

namespace
{

template<class Op>
Foam::scalar reduceList(const Foam::UList<Foam::scalar>& f, Foam::scalar init, Op op)
{
    if (f.empty())
    {
        return init;
    }
    const Foam::scalar* p = f.cdata();
    return f.usePool()
      ? thrust::reduce(thrust::device, p, p + f.size(), init, op)
      : thrust::reduce(thrust::host, p, p + f.size(), init, op);
}

template<class Op>
Foam::scalar reduceField(const Foam::volScalarField& vsf, Foam::scalar init, Op op)
{
    Foam::scalar r = reduceList(vsf.primitiveField(), init, op);
    for (const auto& pf : vsf.boundaryField())
    {
        r = op(r, reduceList(pf, init, op));
    }
    return r;
}

}


Foam::volScalarField&
Foam::bound(volScalarField& vsf, const dimensionedScalar& lowerBound)
{
    scalar minVsf = reduceField(vsf, VGREAT, thrust::minimum<scalar>());
    reduce(minVsf, minOp<scalar>());

    if (minVsf < lowerBound.value())
    {
        scalar maxVsf = reduceField(vsf, -VGREAT, thrust::maximum<scalar>());
        reduce(maxVsf, maxOp<scalar>());

        Info<< "bounding " << vsf.name()
            << ", min: " << minVsf
            << " max: " << maxVsf
            << " average: " << gAverage(vsf.primitiveField())
            << endl;

        vsf.primitiveFieldRef() = max
        (
            max
            (
                vsf.primitiveField(),
                fvc::average(max(vsf, lowerBound))().primitiveField()
              * pos0(-vsf.primitiveField())
            ),
            lowerBound.value()
        );

        vsf.boundaryFieldRef() = max(vsf.boundaryField(), lowerBound.value());

        vsf.boundaryFieldRef().evaluateCoupled<coupledFvPatch>();
    }

    return vsf;
}
