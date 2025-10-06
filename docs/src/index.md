```@meta
CurrentModule = PeriodicMatrixEquations
DocTestSetup = quote
    using PeriodicMatrixEquations
end
```

# PeriodicMatrixEquations.jl

[![DocBuild](https://github.com/andreasvarga/PeriodicMatrixEquations.jl/workflows/CI/badge.svg)](https://github.com/andreasvarga/PeriodicMatrixEquations.jl/actions)
[![Code on Github.](https://img.shields.io/badge/code%20on-github-blue.svg)](https://github.com/andreasvarga/PeriodicMatrixEquations.jl)

`PeriodicMatrixEquations.jl` is a collection of Julia functions for the solution of several categories of periodic differential/difference equations. The implementation of solvers relies on the periodic matrix objects defined within the [`PeriodicMatrices`](https://github.com/andreasvarga/PeriodicMatrices.jl) package. 
The available functions cover both continuous-time and discrete-time settings, by solving, respectively, periodic differential and difference Lyapunov and Riccati equations with real periodic matrices. The available solvers rely on efficient structure preserving methods using the periodic Schur decomposition of a product of matrices. The solutions of periodic differential equations are determined as single- or multiple-point periodic generators, which allow the efficient computation of the solutions at arbitrary time values by integrating the appropriate differential equations. Akternatively, interpolation with cubic splines can be used to determine the solution at arbitrary time values.   

The current version of the package includes the following functions:

**Solving periodic Sylvester equations (WIP)**

* **[`pdsylv`](@ref)** Solution of periodic discrete-time Sylvester equations. 
* **[`pfdsylv`](@ref)** Solution of forward-time periodic discrete-time Sylvester equations.
* **[`prdsylv`](@ref)** Solution of reverse-time periodic discrete-time Sylvester equations.
* **[`pdsylvc`](@ref)** Solution of periodic discrete-time Sylvester equations of continuous-time flavor. 
* **[`pfdsylvc`](@ref)** Solution of forward-time periodic discrete-time Sylvester equations of continuous-time flavor.
* **[`prdsylvc`](@ref)** Solution of reverse-time periodic discrete-time Sylvester equations of continuous-time flavor.

**Solving periodic Lyapunov equations**

* **[`pclyap`](@ref)** Solution of periodic Lyapunov differential equations. 
* **[`prclyap`](@ref)** Solution of reverse-time periodic Lyapunov differential equations. 
* **[`pfclyap`](@ref)**  Solution of forward-time periodic Lyapunov differential equations.
* **[`pgclyap`](@ref)** Computation of periodic generators for periodic Lyapunov differential equations.
* **[`pdlyap`](@ref)** Solution of periodic discrete-time Lyapunov equations. 
* **[`pdlyap2`](@ref)** Solution of a pair of periodic discrete-time Lyapunov equations. 
* **[`prdlyap`](@ref)** Solution of reverse-time periodic discrete-time Lyapunov equations. 
* **[`pfdlyap`](@ref)**  Solution of forward-time periodic discrete-time Lyapunov equations.
* **[`pcplyap`](@ref)** Solution of positve periodic Lyapunov differential equations. 
* **[`prcplyap`](@ref)** Solution of positve reverse-time periodic Lyapunov differential equations.
* **[`pfcplyap`](@ref)**  Solution of positve forward-time periodic Lyapunov differential equations.
* **[`pdplyap`](@ref)** Solution of positve periodic discrete-time Lyapunov equations. 
* **[`prdplyap`](@ref)** Solution of positve reverse-time periodic discrete-time Lyapunov equations. 
* **[`pfdplyap`](@ref)**  Solution of positve forward-time periodic discrete-time Lyapunov equations.

**Solving periodic Riccati equations**

* **[`pcric`](@ref)** Solution of periodic Riccati differential equations. 
* **[`prcric`](@ref)** Solution of control-related reverse-time periodic Riccati differential equation. 
* **[`pfcric`](@ref)**  Solution of filtering-related forward-time periodic Riccati differential equation.
* **[`pgcric`](@ref)** Computation of periodic generators for periodic Riccati differential equations.
* **[`prdric`](@ref)** Solution of control-related reverse-time periodic Riccati difference equation. 
* **[`pfdric`](@ref)** Solution of filtering-related forward-time periodic Riccati difference equation. 



## [Release Notes](https://github.com/andreasvarga/PeriodicMatrixEquations.jl/blob/master/ReleaseNotes.md)

## Main developer

[Andreas Varga](https://sites.google.com/view/andreasvarga/home)

License: MIT (expat)

## References

[1] A. Varga. On solving periodic differential matrix equations with applications to periodic system norms computation. Proc. IEEE CDC/ECC, Seville, 2005.

[2] A. Varga. Periodic Lyapunov equations: some applications and new algorithms. Int. J. Control, vol, 67, pp, 69-87, 1997. 

[3] A. Varga. On solving periodic Riccati equations. Numerical Linear Algebra with Applications, 15:809-835, 2008. 
