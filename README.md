# PeriodicMatrixEquations.jl

<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4568159.svg)](https://doi.org/10.5281/zenodo.4568159) -->
[![DocBuild](https://github.com/andreasvarga/PeriodicMatrixEquations.jl/workflows/CI/badge.svg)](https://github.com/andreasvarga/PeriodicMatrixEquations.jl/actions)
[![codecov.io](https://codecov.io/gh/andreasvarga/PeriodicMatrixEquations.jl/coverage.svg?branch=master)](https://codecov.io/gh/andreasvarga/PeriodicMatrixEquations.jl?branch=master)
[![stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://andreasvarga.github.io/PeriodicMatrixEquations.jl/stable/)
[![dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://andreasvarga.github.io/PeriodicMatrixEquations.jl/dev/)
[![The MIT License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square)](https://github.com/andreasvarga/PeriodicMatrixEquations.jl/blob/master/LICENSE.md)

## Solution of periodic differential/difference matrix equations 

## Compatibility

Julia 1.10 and higher.

<!-- ## How to install

````JULIA
pkg> add PeriodicMatrixEquations
pkg> test PeriodicMatrixEquations
```` -->

## About

`PeriodicMatrixEquations.jl` is intended to be a collection of Julia functions for the solution of several categories of periodic differential/difference equations. The implementation of solvers relies on the periodic matrix objects defined within the [`PeriodicMatrices`](https://github.com/andreasvarga/PeriodicMatrices.jl) package. 
The available functions cover both continuous-time and discrete-time settings, by solving, respectively, periodic differential and difference Lyapunov and Riccati equations with real periodic matrices. The available solvers rely on efficient structure preserving methods using the periodic Schur decomposition of a product of matrices. The solution of periodic differential equations consists of multiple-point periodic generators, which allow the high-precision evaluation of the solution by integrating the appropriate differential equations. By default, interpolation is used for fast evaluation of the solution at arbitrary time values.   


[1] A. Varga. On solving periodic differential matrix equations with applications to periodic system norms computation. Proc. IEEE CDC/ECC, Seville, 2005.

[2] A. Varga. Periodic Lyapunov equations: some applications and new algorithms. Int. J. Control, vol, 67, pp, 69-87, 1997. 

[3] A. Varga. On solving periodic Riccati equations. Numerical Linear Algebra with Applications, 15:809-835, 2008. 
