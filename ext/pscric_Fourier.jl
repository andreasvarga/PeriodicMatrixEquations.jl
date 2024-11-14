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
_Note:_ Presently the `PeriodicSwitchingMatrix` type is not supported. 

If `fast = true` (default) the multiple-shooting method is used in conjunction with fast pencil reduction techniques, as proposed in [1],
to determine the periodic solution in `t = 0` and a multiple point generator of the appropriate periodic differential Riccati equation
is determined  (see [2] for details). 
If `fast = false`, the multiple-shooting method is used in 
conjunction with the periodic Schur decomposition to determine multiple point generators directly from the stable periodic invariant subspace of 
an appropriate symplectic transition matrix (see also [2] for more details). 

The keyword argument `K` specifies the number of grid points to be used
for the resulting multiple point periodic generator (default: `K = 10`). 
The obtained periodic generator is finally converted into a periodic function matrix which determines for a given `t` 
the function value `X(t)` by integrating the appropriate ODE from the nearest grid point value. 

To speedup function evaluations, interpolation based function evaluations can be used 
by setting the keyword argument `intpol = true` (default: `intpol = true` if `solver = "symplectic"`, otherwise `intpol = false`). 
In this case the interpolation method to be used can be specified via the keyword argument
`intpolmeth = meth`. The allowable values for `meth` are: `"constant"`, `"linear"`, `"quadratic"` and `"cubic"` (default).
   
The ODE solver to be employed to convert the continuous-time problem into a discrete-time problem can be specified using the keyword argument `solver`, together with
the required relative accuracy `reltol` (default: `reltol = 1.e-4`) and 
absolute accuracy `abstol` (default: `abstol = 1.e-7`). 
Depending on the desired relative accuracy `reltol`, lower order solvers are employed for `reltol >= 1.e-4`, 
which are generally very efficient, but less accurate. If `reltol < 1.e-4`,
higher order solvers are employed able to cope with high accuracy demands. 

The following solvers from the [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) package can be selected:

`solver = "non-stiff"` - use a solver for non-stiff problems (`Tsit5()` or `Vern9()`);

`solver = "stiff"` - use a solver for stiff problems (`Rodas4()` or `KenCarp58()`);

`solver = "linear"` - use a special solver for linear ODEs (`MagnusGL6()`) with fixed time step `dt`;

`solver = "symplectic"` - use a symplectic Hamiltonian structure preserving solver (`IRKGL16()`);

`solver = ""` - use the default solver, which automatically detects stiff problems (`AutoTsit5(Rosenbrock23())` or `AutoVern9(Rodas5())`). 

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
      function pcric(A::PM, R::PM, Q::PM; K::Int = 10, adj = false, PSD_SLICOT::Bool = true, solver = "symplectic", reltol = 1.e-4, abstol = 1.e-7, dt = 0.0, 
         fast = true,  intpol = solver == "symplectic" ? true : false, intpolmeth = "cubic") where {PM <: FourierFunctionMatrix}
         At = convert(PeriodicFunctionMatrix,A)
         Rt = convert(PeriodicFunctionMatrix,R)
         Qt = convert(PeriodicFunctionMatrix,Q)
         X, EVALS = PeriodicMatrixEquations.pgcric(At, Rt, Qt, K;  adj, solver, reltol, abstol, dt, fast, PSD_SLICOT)
         if intpol
            return convert(PeriodicFunctionMatrix, X, method = intpolmeth), EVALS
         else
            return PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvcric_eval(t, X, At, Rt, Qt; solver, adj, reltol, abstol),A.period), EVALS
         end
      end
      function PeriodicMatrixEquations.prcric(A::PM, B::PM, R::PM, Q::PM; K::Int = 10, PSD_SLICOT::Bool = true, solver = "symplectic", reltol = 1.e-4, abstol = 1.e-7, dt = 0.0, 
         fast = true,  intpol = solver == "symplectic" ? true : false, intpolmeth = "cubic") where {PM <: FourierFunctionMatrix}
         n = size(A,1)
         n == size(A,2) || error("the periodic matrix A must be square")
         n == size(B,1) || error("the periodic matrix B must have the same number of rows as A")
         m = size(B,2)
         (m,m) == size(R) || error("the periodic matrix R must have the same dimensions as the column dimension of B")
         (n,n) == size(Q) || error("the periodic matrix Q must have the same dimensions as A")
         issymmetric(R) || error("the periodic matrix R must be symmetric")
         issymmetric(Q) || error("the periodic matrix Q must be symmetric")
         Rt = pmmultrsym(B*inv(R), B)
         X, EVALS = pcric(A, Rt, Q; K, adj = true, solver, reltol, abstol, dt, fast, PSD_SLICOT, intpol, intpolmeth)
         return X, EVALS, inv(R)*transpose(B)*X
      end
      function PeriodicMatrixEquations.prcric(A::PM, B::PM, R::AbstractMatrix, Q::AbstractMatrix; K::Int = 10, PSD_SLICOT::Bool = true, solver = "symplectic", reltol = 1.e-4, abstol = 1.e-7, dt = 0.0, 
         fast = true,  intpol = solver == "symplectic" ? true : false, intpolmeth = "cubic") where {PM <: FourierFunctionMatrix}
         n = size(A,1)
         n == size(A,2) || error("the periodic matrix A must be square")
         n == size(B,1) || error("the periodic matrix B must have the same number of rows as A")
         m = size(B,2)
         (m,m) == size(R) || error("the periodic matrix R must have the same dimensions as the column dimension of B")
         (n,n) == size(Q) || error("the periodic matrix Q must have the same dimensions as A")
         issymmetric(R) || error("the matrix R must be symmetric")
         issymmetric(Q) || error("the matrix Q must be symmetric")
         Rt = pmmultrsym(B*FourierFunctionMatrix(inv(R),B.period), B)
         X, EVALS = pcric(A, Rt, FourierFunctionMatrix(Q, A.period); K, adj = true, solver, reltol, abstol, dt, fast, PSD_SLICOT, intpol, intpolmeth)
         return X, EVALS, inv(R)*B'*X
      end
      function PeriodicMatrixEquations.pfcric(A::PM, C::PM, R::PM, Q::PM; K::Int = 10, PSD_SLICOT::Bool = true, solver = "symplectic", reltol = 1.e-4, abstol = 1.e-7, dt = 0.0, 
         fast = true, intpol = solver == "symplectic" ? true : false, intpolmeth = "cubic") where {PM <: FourierFunctionMatrix}
         n = size(A,1)
         n == size(A,2) || error("the periodic matrix A must be square")
         n == size(C,2) || error("the periodic matrix C must have the same number of columns as A")
         m = size(C,1)
         (m,m) == size(R) || error("the periodic matrix R must have same dimensions as the row dimension of C")
         (n,n) == size(Q) || error("the periodic matrix Q must have the same dimensions as A")
         issymmetric(R) || error("the periodic matrix R must be symmetric")
         issymmetric(Q) || error("the periodic matrix Q must be symmetric")
         Rt = pmtrmulsym(C,inv(R)*C)
         X, EVALS = pcric(A, Rt, Q; K, adj = false, solver, reltol, abstol, dt, fast, PSD_SLICOT, intpol, intpolmeth)
         return X, EVALS, (X*C')*inv(R)
      end
      function PeriodicMatrixEquations.pfcric(A::PM, C::PM, R::AbstractMatrix, Q::AbstractMatrix; K::Int = 10, PSD_SLICOT::Bool = true, solver = "symplectic", reltol = 1.e-4, abstol = 1.e-7, dt = 0.0, 
                      fast = true, intpol = solver == "symplectic" ? true : false, intpolmeth = "cubic") where {PM <: FourierFunctionMatrix}
         n = size(A,1)
         n == size(A,2) || error("the periodic matrix A must be square")
         n == size(C,2) || error("the periodic matrix C must have the same number of columns as A")
         m = size(C,1)
         (m,m) == size(R) || error("the periodic matrix R must have same dimensions as the row dimension of C")
         (n,n) == size(Q) || error("the periodic matrix Q must have the same dimensions as A")
         issymmetric(R) || error("the matrix R must be symmetric")
         issymmetric(Q) || error("the matrix Q must be symmetric")
         Rt = pmtrmulsym(C,FourierFunctionMatrix(inv(R), C.period)*C)
         X, EVALS = pcric(A, Rt, FourierFunctionMatrix(Q, A.period); K, adj = false, solver, reltol, abstol, dt, fast, PSD_SLICOT, intpol, intpolmeth)
         return X, EVALS, (X*C')*inv(R)
      end
