using Documenter, PeriodicMatrixEquations
DocMeta.setdocmeta!(PeriodicMatrixEquations, :DocTestSetup, :(using PeriodicMatrixEquations); recursive=true)

makedocs(warnonly = true, 
  modules  = [PeriodicMatrixEquations],
  sitename = "PeriodicMatrixEquations.jl",
  authors  = "Andreas Varga",
  format   = Documenter.HTML(prettyurls = false),
  pages    = [
      "Home"   => "index.md",
      "Library" => [ 
          "pslyap.md",
          "psric.md",
         ],
     "Utilities" => [
      "pstools.md"
      ],
     "Index" => "makeindex.md"
  ]
)

deploydocs(
  repo = "github.com/andreasvarga/PeriodicMatrixEquations.jl.git",
  target = "build",
  devbranch = "master"
)
