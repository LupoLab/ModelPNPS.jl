using ModelPNPS
using Documenter

DocMeta.setdocmeta!(ModelPNPS, :DocTestSetup, :(using ModelPNPS); recursive=true)

makedocs(;
    modules=[ModelPNPS],
    authors="John Travers <jtravs@gmail.com> and contributors",
    sitename="ModelPNPS.jl",
    warnonly=[:missing_docs],
    format=Documenter.HTML(;
        canonical="https://jtravs.github.io/ModelPNPS.jl",
        edit_link="main",
        assets=String[],
        mathengine=Documenter.MathJax3(),
    ),
    pages=[
        "Home" => "index.md",
        "PNPS Framework" => "pnps.md",
        "Trace Simulation" => "trace_simulation.md",
        "API Reference" => "interface.md",
    ],
)

deploydocs(;
    repo="github.com/jtravs/ModelPNPS.jl",
    devbranch="main",
)
