"""
    pcric(A, R, Q; K = 10, adj = false, solver, reltol, abstol, fast, intpol, intpolmeth) -> (X, EVALS)

Solve the periodic Riccati differential equation

    .                                                 
    X(t) = A(t)X(t) + X(t)A(t)' + Q(t) - X(t)R(t)X(t) ,  if adj = false,

or 

     .                                                
    -X(t) = A(t)'X(t) + X(t)A(t) + Q(t) - X(t)R(t)X(t) , if adj = true
    
and compute the stable closed-loop characteristic multipliers in `EVALS` (see [`pgcric`](@ref) for details).

The periodic matrices `A`, `R` and `Q` must have the same type, the same dimensions and commensurate periods, 
and additionally `R` and `Q` must be symmetric. The resulting symmetric periodic solution `X` has the type `PeriodicFunctionMatrix` and 
`X(t)` can be used to evaluate the value of `X` at time `t`. 
`X` has the period set to the least common commensurate period of `A`, `R` and `Q` and 
the number of subperiods is adjusted accordingly. 

If `fast = true` (default) the multiple-shooting method is used in conjunction with fast pencil reduction techniques, as proposed in [1],
to determine the periodic solution in `t = 0` and a multiple point generator of the appropriate periodic differential Riccati equation
is determined  (see [2] for details). 
If `fast = false`, the multiple-shooting method is used in 
conjunction with the periodic Schur decomposition to determine multiple point generators directly from the stable periodic invariant subspace of 
an appropriate symplectic transition matrix (see also [2] for more details). 

The keyword argument `K` specifies the number of grid points to be used
for the resulting multiple point periodic generator (default: `K = 10`). 
The obtained periodic generator is finally converted into a periodic function matrix which determines for a given `t` 
the function value `X(t)` by interpolating the solution of the appropriate differential equations if `intpol = true` (default)  
or by integrating the appropriate ODE from the nearest grid point value if `intpol = false`.
For the interplation-based evaluation the integer keyword argument `N` can be used to split the integration domain (i.e., one period) 
into `N` subdomains to perform the interpolations separately in each domain.
The default value of `N` is `N = 1`.  

For the determination of the multiple point periodic generators an implicit Runge-Kutta Gauss-Legendre 16th order method
from the [IRKGaussLegendre.jl](https://github.com/SciML/IRKGaussLegendre.jl) package is employed to integrate the appropriate Hamiltonian system [2]. 

For the evaluation of solution via interpolation or ODE integration, the 
following solvers from the [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) package can be selected using the keyword argument `solver`: 

`solver = "non-stiff"` - use a solver for non-stiff problems (`Tsit5()` or `Vern9()`);

`solver = "stiff"` - use a solver for stiff problems (`Rodas4()` or `KenCarp58()`);

`solver = ""` - use the default solver, which automatically detects stiff problems (`AutoTsit5(Rosenbrock23())` or `AutoVern9(Rodas5())`). 

The accuracy of the computed solutions can be controlled via 
the relative accuracy keyword `reltol` (default: `reltol = 1.e-4`) and 
absolute accuracy keyword `abstol` (default: `abstol = 1.e-7`). 
Depending on the desired relative accuracy `reltol`, lower order solvers are employed for `reltol >= 1.e-4`, 
which are generally very efficient, but less accurate. If `reltol < 1.e-4`,
higher order solvers are employed able to cope with high accuracy demands. 

For large values of `K`, parallel computation of the matrices of the discrete-time problem can be alternatively performed 
by starting Julia with several execution threads. 
The number of execution threads is controlled either by using the `-t/--threads` command line argument 
or by using the `JULIA_NUM_THREADS` environment variable.  

_References_

[1] A. Varga. On solving periodic differential matrix equations with applications to periodic system norms computation. 
    Proc. IEEE CDC/ECC, Seville, 2005. 

[2] A. Varga. On solving periodic Riccati equations.  
    Numerical Linear Algebra with Applications, 15:809-835, 2008.    
"""
      function PeriodicMatrixEquations.pcric(A::PM, R::PM, Q::PM; K::Int = 10, N::Int = 1, adj = false, PSD_SLICOT::Bool = true, solver1 = "symplectic", solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7, dt = 0.0, 
         fast = true,  intpol = true) where {PM <: FourierFunctionMatrix}
         At = convert(PeriodicFunctionMatrix,A)
         Rt = convert(PeriodicFunctionMatrix,R)
         Qt = convert(PeriodicFunctionMatrix,Q)
         X, EVALS = PeriodicMatrixEquations.pgcric(At, Rt, Qt, K;  adj, solver = solver1, reltol, abstol, dt, fast, PSD_SLICOT)
         if PeriodicMatrices.isconstant(X) 
            (K > 1 || (PeriodicMatrices.isconstant(A) && PeriodicMatrices.isconstant(R) && PeriodicMatrices.isconstant(Q))) && (return PeriodicFunctionMatrix(X(0),X.period,nperiod = X.nperiod), EVALS)
         end
         if intpol 
            return PeriodicMatrixEquations.pcric_intpol(At, Rt, Qt, X; N, adj, solver, reltol, abstol), EVALS
            #return convert(PeriodicFunctionMatrix, X, method = intpolmeth), EVALS
         else
            return PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvcric_eval(t, X, At, Rt, Qt; solver, adj, reltol, abstol),A.period), EVALS
         end
      end
      function PeriodicMatrixEquations.prcric(A::PM, B::PM, R::PM, Q::PM; K::Int = 10, N::Int = 1, PSD_SLICOT::Bool = true, solver1 = "symplectic", solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7, dt = 0.0, 
         fast = true,  intpol = true) where {PM <: FourierFunctionMatrix}
         n = size(A,1)
         n == size(A,2) || error("the periodic matrix A must be square")
         n == size(B,1) || error("the periodic matrix B must have the same number of rows as A")
         m = size(B,2)
         (m,m) == size(R) || error("the periodic matrix R must have the same dimensions as the column dimension of B")
         (n,n) == size(Q) || error("the periodic matrix Q must have the same dimensions as A")
         issymmetric(R) || error("the periodic matrix R must be symmetric")
         issymmetric(Q) || error("the periodic matrix Q must be symmetric")
         Rt = pmmultrsym(B*inv(R), B)
         X, EVALS = pcric(A, Rt, Q; K, N, adj = true, solver1, solver, reltol, abstol, dt, fast, PSD_SLICOT, intpol)
         return X, EVALS, inv(R)*transpose(B)*X
      end
      function PeriodicMatrixEquations.prcric(A::PM, B::PM, R::AbstractMatrix, Q::AbstractMatrix; K::Int = 10, N::Int = 1, PSD_SLICOT::Bool = true, solver1 = "symplectic", solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7, dt = 0.0, 
         fast = true,  intpol = true) where {PM <: FourierFunctionMatrix}
         n = size(A,1)
         n == size(A,2) || error("the periodic matrix A must be square")
         n == size(B,1) || error("the periodic matrix B must have the same number of rows as A")
         m = size(B,2)
         (m,m) == size(R) || error("the periodic matrix R must have the same dimensions as the column dimension of B")
         (n,n) == size(Q) || error("the periodic matrix Q must have the same dimensions as A")
         issymmetric(R) || error("the matrix R must be symmetric")
         issymmetric(Q) || error("the matrix Q must be symmetric")
         Rt = pmmultrsym(B*FourierFunctionMatrix(inv(R),B.period), B)
         X, EVALS = pcric(A, Rt, FourierFunctionMatrix(Q, A.period); K, N, adj = true, solver1, solver, reltol, abstol, dt, fast, PSD_SLICOT, intpol)
         return X, EVALS, inv(R)*B'*X
      end
      function PeriodicMatrixEquations.pfcric(A::PM, C::PM, R::PM, Q::PM; K::Int = 10, N::Int = 1, PSD_SLICOT::Bool = true, solver1 = "symplectic", solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7, dt = 0.0, 
         fast = true, intpol = true) where {PM <: FourierFunctionMatrix}
         n = size(A,1)
         n == size(A,2) || error("the periodic matrix A must be square")
         n == size(C,2) || error("the periodic matrix C must have the same number of columns as A")
         m = size(C,1)
         (m,m) == size(R) || error("the periodic matrix R must have same dimensions as the row dimension of C")
         (n,n) == size(Q) || error("the periodic matrix Q must have the same dimensions as A")
         issymmetric(R) || error("the periodic matrix R must be symmetric")
         issymmetric(Q) || error("the periodic matrix Q must be symmetric")
         Rt = pmtrmulsym(C,inv(R)*C)
         X, EVALS = pcric(A, Rt, Q; K, adj = false, solver1, solver, reltol, abstol, dt, fast, PSD_SLICOT, intpol)
         return X, EVALS, (X*C')*inv(R)
      end
      function PeriodicMatrixEquations.pfcric(A::PM, C::PM, R::AbstractMatrix, Q::AbstractMatrix; K::Int = 10, PSD_SLICOT::Bool = true, solver1 = "symplectic", solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7, dt = 0.0, 
                      fast = true, intpol = true) where {PM <: FourierFunctionMatrix}
         n = size(A,1)
         n == size(A,2) || error("the periodic matrix A must be square")
         n == size(C,2) || error("the periodic matrix C must have the same number of columns as A")
         m = size(C,1)
         (m,m) == size(R) || error("the periodic matrix R must have same dimensions as the row dimension of C")
         (n,n) == size(Q) || error("the periodic matrix Q must have the same dimensions as A")
         issymmetric(R) || error("the matrix R must be symmetric")
         issymmetric(Q) || error("the matrix Q must be symmetric")
         Rt = pmtrmulsym(C,FourierFunctionMatrix(inv(R), C.period)*C)
         X, EVALS = pcric(A, Rt, FourierFunctionMatrix(Q, A.period); K, adj = false, solver1, solver, reltol, abstol, dt, fast, PSD_SLICOT, intpol)
         return X, EVALS, (X*C')*inv(R)
      end
