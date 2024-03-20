using RandomFeatures
using Documenter

DocMeta.setdocmeta!(RandomFeatures, :DocTestSetup, :(using RandomFeatures); recursive=true)

makedocs(;
    modules=[RandomFeatures],
    authors="murrellb <murrellb@gmail.com> and contributors",
    sitename="RandomFeatures.jl",
    format=Documenter.HTML(;
        canonical="https://murrellb.github.io/RandomFeatures.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/murrellb/RandomFeatures.jl",
    devbranch="main",
)
