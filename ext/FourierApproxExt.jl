module FourierApproxExt

using PeriodicMatrixEquations
using ApproxFun
using LinearAlgebra
using Symbolics
using OrdinaryDiffEq
using IRKGaussLegendre
using MatrixEquations


include("pscsylv_Fourier.jl")
include("psclyap_Fourier.jl")
include("pscric_Fourier.jl")

end