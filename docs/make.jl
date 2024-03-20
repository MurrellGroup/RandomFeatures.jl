using RandomFeatures
using Documenter

DocMeta.setdocmeta!(RandomFeatures, :DocTestSetup, :(using RandomFeatures); recursive=true)

makedocs(;
    modules=[RandomFeatures],
    authors="murrellb <murrellb@gmail.com> and contributors",
    sitename="RandomFeatures.jl",
    format=Documenter.HTML(;
        canonical="https://MurrellGroup.github.io/RandomFeatures.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/RandomFeatures.jl",
    devbranch="main",
)
