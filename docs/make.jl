import Documenter
import Documenter: DocMeta, deploydocs, makedocs
import ModelPNPS

DocMeta.setdocmeta!(ModelPNPS, :DocTestSetup, :(using ModelPNPS); recursive = true)

makedocs(;
    modules = [ModelPNPS],
    authors = "John Travers <jtravs@gmail.com> and contributors",
    sitename = "ModelPNPS.jl",
    warnonly = false,
    checkdocs = :all,
    doctest = true,
    format = Documenter.HTML(;
        canonical = "https://lupo-lab.com/ModelPNPS.jl",
        edit_link = "main",
        assets = String[],
        mathengine = Documenter.MathJax3(),
        size_threshold_warn = 150 * 1024,
    ),
    pages = [
        "Home" => "index.md",
        "PNPS Framework" => "pnps.md",
        "Trace Simulation" => "trace_simulation.md",
        "Input Pulses" => "input_pulses.md",
        "Nonlinear Response" => "nonlinear_response.md",
        "Field-Resolved Mode" => "field_mode.md",
        "Running on a GPU" => "gpu.md",
        "Accuracy and Validation" => "accuracy.md",
        "API Reference" => "interface.md",
    ],
)

if get(ENV, "CI", "false") == "true"
    deploydocs(;
        repo = "github.com/LupoLab/ModelPNPS.jl",
        devbranch = "main",
    )
end
