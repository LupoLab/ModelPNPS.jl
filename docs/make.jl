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
        canonical = "https://lupolab.github.io/ModelPNPS.jl",
        edit_link = "main",
        assets = String[],
        mathengine = Documenter.MathJax3(),
        size_threshold_warn = 150 * 1024,
    ),
    pages = [
        "Home" => "index.md",
        "PNPS Framework" => "pnps.md",
        "Trace Simulation" => "trace_simulation.md",
        "API Reference" => "interface.md",
    ],
)

if get(ENV, "CI", "false") == "true"
    deploydocs(;
        repo = "github.com/LupoLab/ModelPNPS.jl",
        devbranch = "main",
    )
end
