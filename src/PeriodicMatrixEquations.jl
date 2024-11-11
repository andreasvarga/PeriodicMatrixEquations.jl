module PeriodicMatrixEquations

using Reexport
@reexport using PeriodicMatrices
using FastLapackInterface
using IRKGaussLegendre
using LinearAlgebra
using MatrixEquations
using OrdinaryDiffEq
using PeriodicSchurDecompositions
using Symbolics

import LinearAlgebra: BlasInt, BlasFloat, BlasReal, BlasComplex

export pdlyap, pdlyap2, prdlyap, pfdlyap, pslyapd, pslyapd2, pdlyaps!, pdlyaps1!, pdlyaps2!, pdlyaps3!, dpsylv2, dpsylv2!, pslyapdkr, dpsylv2krsol!, kronset!
export prdplyap, pfdplyap, pdplyap, psplyapd
export pclyap, pfclyap, prclyap, pgclyap, pgclyap2, tvclyap_eval
export pcplyap, pfcplyap, prcplyap, pgcplyap, tvcplyap_eval
export pcric, prcric, pfcric, tvcric, pgcric, prdric, pfdric, tvcric_eval


include("psclyap.jl")
include("psdlyap.jl")
include("pscric.jl")
include("psdric.jl")

end
