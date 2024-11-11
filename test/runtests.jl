module Runtests

using Test, PeriodicMatrixEquations

@testset "Test PeriodicMatrixEquations" begin
include("test_psdlyap.jl")
include("test_psclyap.jl")
include("test_pscric.jl")
include("test_psdric.jl")
end

end
