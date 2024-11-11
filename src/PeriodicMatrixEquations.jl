module PeriodicMatrixEquations

using Reexport
@reexport using PeriodicMatrices
#using ApproxFun
using DescriptorSystems
using FastLapackInterface
using IRKGaussLegendre
using LinearAlgebra
using MatrixEquations
using MatrixPencils
using OrdinaryDiffEq
using PeriodicSchurDecompositions
using Symbolics

# import Base: +, -, *, /, \, (==), (!=), ^, isapprox, iszero, convert, promote_op, size, length, ndims, reverse,
#              hcat, vcat, hvcat, inv, show, lastindex, require_one_based_indexing, print, show, one, zero, eltype
# import DescriptorSystems: isstable, horzcat, vertcat, blockdiag, parallel, series, append, isconstant
import LinearAlgebra: BlasInt, BlasFloat, BlasReal, BlasComplex
import PeriodicMatrices: iscontinuous, isdiscrete


# export PeriodicStateSpace
# export ps, islti, ps_validation
# export psaverage, psc2d, psmrc2d, psteval, pseval, psparallel, psseries, psappend, pshorzcat, psvertcat, psinv, psfeedback
# export ps2fls, ps2frls, ps2ls, ps2spls
# export pspole, pszero, isstable, psh2norm, pshanorm, pslinfnorm, pstimeresp, psstepresp
export pdlyap, pdlyap2, prdlyap, pfdlyap, pslyapd, pslyapd2, pdlyaps!, pdlyaps1!, pdlyaps2!, pdlyaps3!, dpsylv2, dpsylv2!, pslyapdkr, dpsylv2krsol!, kronset!
export prdplyap, pfdplyap, pdplyap, psplyapd
export pclyap, pfclyap, prclyap, pgclyap, pgclyap2, tvclyap_eval
export pcplyap, pfcplyap, prcplyap, pgcplyap, tvcplyap_eval
export pcric, prcric, pfcric, tvcric, pgcric, prdric, pfdric, tvcric_eval
# export psfeedback, pssfeedback, pssofeedback
# export pcpofstab_sw, pcpofstab_hr, pdpofstab_sw, pdpofstab_hr, pclqr, pclqry, pdlqr, pdlqry, pdkeg, pckeg, pdkegw, pckegw, pdlqofc, pdlqofc_sw, pclqofc_sw, pclqofc_hr


include("psdlyap.jl")
include("psclyap.jl")
include("pscric.jl")
include("psdric.jl")

end
