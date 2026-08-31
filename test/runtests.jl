using SafeTestsets: @safetestset

const GROUP = get(ENV, "GROUP", "All")
const VALID_GROUPS = ("All", "Core", "Physics", "Quality", "Docs")

GROUP in VALID_GROUPS || throw(
    ArgumentError(
        "unknown test group '$GROUP'; expected one of " *
            join(VALID_GROUPS, ", ")
    )
)

if GROUP in ("All", "Core")
    @time @safetestset "Inference" include("inference_tests.jl")
    @time @safetestset "Input pulse" include("input_pulse_test.jl")
    @time @safetestset "Device support" include("device_test.jl")
end

if GROUP in ("All", "Physics")
    @time @safetestset "Trace simulation" include("tracesimulation_test.jl")
    @time @safetestset "Frozen-transverse ablation" include("frozen_transverse_test.jl")
end

if GROUP in ("All", "Quality")
    @time @safetestset "Package quality" include("quality_tests.jl")
end

if GROUP in ("All", "Docs")
    @time @safetestset "Doctests" include("docs_tests.jl")
end
