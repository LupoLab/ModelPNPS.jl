using ModelPNPS
using Test

@testset "ModelPNPS.jl" begin
    include("tracesimulation_test.jl")
    include("input_pulse_test.jl")
    include("device_test.jl")
end
