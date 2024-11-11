module FourierApproxExt

using PeriodicMatrixEquations
using ApproxFun
using LinearAlgebra
using Symbolics
using OrdinaryDiffEq
using IRKGaussLegendre


include("psclyap_Fourier.jl")
include("pscric_Fourier.jl")

end