"""
    pclyap(A, C; K = 10, N = 1, adj = false, solver, reltol, abstol, intpol -> X

Solve the periodic Lyapunov differential equation

    .
    X(t) = A(t)X(t) + X(t)A(t)' + C(t) , if adj = false,

or 

     .
    -X(t) = A(t)'X(t) + X(t)A(t) + C(t) , if adj = true.               

The periodic matrices `A` and `C` must have the same type, the same dimensions and commensurate periods. 
Additionally `C` must be symmetric. 
The resulting symmetric periodic solution `X` has the type `PeriodicFunctionMatrix` and 
`X(t)` can be used to evaluate the value of `X` at time `t`. 
`X` has the period set to the least common commensurate period of `A` and `C` and the number of subperiods
is adjusted accordingly. 

The multiple-shooting method of [1] is employed to convert the (continuous-time) periodic differential Lyapunov equation 
into a discrete-time periodic Lyapunov equation satisfied by a multiple point generator of the solution. 
The keyword argument `K` specifies the number of grid points to be used
for the discretization of the continuous-time problem (default: `K = 10`). 
If  `A` and `C` are of types `PeriodicTimeSeriesMatrix` or `PeriodicSwitchingMatrix`, then `K` specifies the number of grid points used between two consecutive switching time values (default: `K = 1`).  
The multiple point periodic generator is computed  by solving the appropriate discrete-time periodic Lyapunov 
equation using the periodic Schur method of [2]. 
The resulting periodic generator is finally converted into a periodic function matrix which determines for a given `t` 
the function value `X(t)` by integrating the appropriate ODE from the nearest grid point value. 

To speedup function evaluations, interpolation based function evaluations can be used 
by setting the keyword argument `intpol = true` (default: `intpol = true`). 
The integer keyword argument `N` can be used to split the integration domain (i.e., one period) into `N` subdomains to perform the interpolations separately in each domain.
The default value of `N` is `N = 1`.  

The ODE solver to be employed to convert the continuous-time problem into a discrete-time problem can be specified using the keyword argument `solver`, together with
the required relative accuracy `reltol` (default: `reltol = 1.e-4`) and 
absolute accuracy `abstol` (default: `abstol = 1.e-7`). 
Depending on the desired relative accuracy `reltol`, lower order solvers are employed for `reltol >= 1.e-4`, 
which are generally very efficient, but less accurate. If `reltol < 1.e-4`,
higher order solvers are employed able to cope with high accuracy demands. 

The following solvers from the [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) package can be selected:

`solver = "non-stiff"` - use a solver for non-stiff problems (`Tsit5()` or `Vern9()`);

`solver = "stiff"` - use a solver for stiff problems (`Rodas4()` or `KenCarp58()`);

`solver = "auto"` - use the default solver, which automatically detects stiff problems (`AutoTsit5(Rosenbrock23())` or `AutoVern9(Rodas5())`). 

Parallel computation of the matrices of the discrete-time problem can be alternatively performed 
by starting Julia with several execution threads. 
The number of execution threads is controlled either by using the `-t/--threads` command line argument 
or by using the `JULIA_NUM_THREADS` environment variable.  

_References_

[1] A. Varga. On solving periodic differential matrix equations with applications to periodic system norms computation. 
    Proc. IEEE CDC/ECC, Seville, 2005. 

[2] A. Varga. Periodic Lyapunov equations: some applications and new algorithms. 
    Int. J. Control, vol, 67, pp, 69-87, 1997.
    
"""
function PeriodicMatrixEquations.pclyap(A::PM, C::PM; K::Int = 10, N::Int = 1, adj = false, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7, intpol = true, stability_check = false) where {PM <: FourierFunctionMatrix}
   W0 = PeriodicMatrixEquations.pgclyap(A, C, K;  adj, solver, reltol, abstol, stability_check)
   if PeriodicMatrices.isconstant(W0) 
      (K > 1 || (PeriodicMatrices.isconstant(A) && PeriodicMatrices.isconstant(C) )) && (return PeriodicFunctionMatrix(W0(0),W0.period,nperiod=W0.nperiod))
   end
   if intpol
      return PeriodicMatrixEquations.pclyap_intpol(A, C, W0; N, adj, solver, reltol, abstol)
   else
      return PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, W0, A, C; solver, adj, reltol, abstol),A.period)
   end
end


for PM in (FourierFunctionMatrix, )
   @eval begin
      function PeriodicMatrixEquations.prclyap(A::$PM, C::$PM; K::Int = 10, N::Int = 1, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7, intpol = true) 
         PeriodicMatrixEquations.pclyap(A, C; K, N, adj = true, solver, reltol, abstol, intpol)
      end
      function PeriodicMatrixEquations.prclyap(A::$PM, C::AbstractMatrix; kwargs...)
         #prclyap(A, $PM(C, A.period; nperiod = A.nperiod); kwargs...)
         PeriodicMatrixEquations.prclyap(A, $PM(C, A.period); kwargs...)
      end
      function PeriodicMatrixEquations.pfclyap(A::$PM, C::$PM; K::Int = 10, N::Int = 1, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7, intpol = true) 
         PeriodicMatrixEquations.pclyap(A, C; K, N, adj = false, solver, reltol, abstol, intpol)
      end
      function PeriodicMatrixEquations.pfclyap(A::$PM, C::AbstractMatrix; kwargs...)
         #PeriodicMatrixEquations.pfclyap(A, $PM(C, A.period; nperiod = A.nperiod); kwargs...)
         PeriodicMatrixEquations.pfclyap(A, $PM(C, A.period); kwargs...)
      end
   end
end

"""
    pgclyap(A, C[, K = 1]; adj = false, solver, reltol, abstol) -> X

Compute periodic generators for the periodic Lyapunov differential equation

    .
    X(t) = A(t)X(t) + X(t)A(t)' + C(t) , if adj = false,

or 

     .
    -X(t) = A(t)'X(t) + X(t)A(t) + C(t) , if adj = true.
    
The periodic matrices `A` and `C` must have the same type, the same dimensions and commensurate periods, 
and additionally `C` must be symmetric. 
If `A` and `C` have the types `PeriodicFunctionMatrix`, `HarmonicArray`, `FourierFunctionMatrix` or `PeriodicTimeSeriesMatrix`, 
then the resulting `X` is a collection of periodic generator matrices determined 
as a periodic time-series matrix with `N` components, where `N = 1` if `A` and `C` are constant matrices
and `N = K` otherwise. 
If `A` and `C` have the type `PeriodicSwitchingMatrix`, then `X` is a collection of periodic generator matrices 
determined as a periodic switching matrix,
whose switching times are the unique switching times contained in the union of the switching times of `A` and `C`. 
If `K > 1`, a refined grid of `K` equidistant values is used for each two consecutive 
switching times in the union.      
The period of `X` is set to the least common commensurate period of `A` and `C` and the number of subperiods
is adjusted accordingly. Any component matrix of `X` is a valid initial value to be used to generate the  
solution over a full period by integrating the appropriate differential equation. 
The multiple-shooting method of [1] is employed, first, to convert the continuous-time periodic Lyapunov differential equation 
into a discrete-time periodic Lyapunov equation satisfied by 
the generator solution in the grid points and then to compute the solution by solving an appropriate discrete-time periodic Lyapunov 
equation using the periodic Schur method of [2]. 

The ODE solver to be employed to convert the continuous-time problem into a discrete-time problem can be specified using the keyword argument `solver`, 
together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`) and 
absolute accuracy `abstol` (default: `abstol = 1.e-7`).
Depending on the desired relative accuracy `reltol`, lower order solvers are employed for `reltol >= 1.e-4`, 
which are generally very efficient, but less accurate. If `reltol < 1.e-4`,
higher order solvers are employed able to cope with high accuracy demands. 

The following solvers from the [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) package can be selected:

`solver = "non-stiff"` - use a solver for non-stiff problems (`Tsit5()` or `Vern9()`);

`solver = "stiff"` - use a solver for stiff problems (`Rodas4()` or `KenCarp58()`);

`solver = "auto"` - use the default solver, which automatically detects stiff problems (`AutoTsit5(Rosenbrock23())` or `AutoVern9(Rodas5())`). 


Parallel computation of the matrices of the discrete-time problem can be alternatively performed 
by starting Julia with several execution threads. 
The number of execution threads is controlled either by using the `-t/--threads` command line argument 
or by using the `JULIA_NUM_THREADS` environment variable.  

_References_

[1] A. Varga. On solving periodic differential matrix equations with applications to periodic system norms computation. 
    Proc. IEEE CDC/ECC, Seville, 2005. 

[2] A. Varga. Periodic Lyapunov equations: some applications and new algorithms. 
    Int. J. Control, vol, 67, pp, 69-87, 1997.
    
"""
function PeriodicMatrixEquations.pgclyap(A::PM1, C::PM2, K::Int = 1; adj = false, solver = "non-stiff", reltol = 1e-4, abstol = 1e-7, stability_check = false) where
      {PM1 <: FourierFunctionMatrix, PM2 <:FourierFunctionMatrix} 
   K > 0 || throw(ArgumentError("number of grid ponts K must be greater than 0, got K = $K"))    
   period = promote_period(A, C)
   na = Int(round(period/A.period))
   nc = Int(round(period/C.period))
   nperiod = gcd(na*A.nperiod, nc*C.nperiod)
   n = size(A,1)
   Ts = period/K/nperiod
   
   T = promote_type(eltype(A),eltype(C),Float64)
   T == Num && (T = Float64)
   if PeriodicMatrices.isconstant(A) && PeriodicMatrices.isconstant(C)
      if stability_check
         ev = eigvals(tpmeval(A,0))
         maximum(real.(ev)) >= - sqrt(eps(T)) && error("system stability check failed")  
      end 
      X = adj ? lyapc(tpmeval(A,0)', tpmeval(C,0)) :  lyapc(tpmeval(A,0), tpmeval(C,0))
   else
      Ka = PeriodicMatrices.isconstant(A) ? 1 : max(1,Int(round(A.period/A.nperiod/Ts)))
      Ad = Array{T,3}(undef, n, n, Ka) 
      Cd = Array{T,3}(undef, n, n, K) 
      Threads.@threads for i = 1:Ka
         @inbounds Ad[:,:,i] = tvstm(A, i*Ts, (i-1)*Ts; solver, reltol, abstol) 
      end
      if stability_check
         ev = PeriodicMatrixEquations.pseig3(Ad)
         maximum(abs.(ev)) >= one(T) - sqrt(eps(T)) && error("system stability check failed")  
      end 
      if adj
         Threads.@threads for i = K:-1:1
            @inbounds Cd[:,:,i]  = PeriodicMatrixEquations.tvclyap(A, C, (i-1)*Ts, i*Ts; adj, solver, reltol, abstol) 
         end
         X = pslyapd(Ad, Cd; adj)
      else
         Threads.@threads for i = 1:K
               @inbounds Cd[:,:,i]  = PeriodicMatrixEquations.tvclyap(A, C, i*Ts, (i-1)*Ts; adj, solver, reltol, abstol) 
         end
         X = pslyapd(Ad, Cd; adj)
      end
   end
   return PeriodicTimeSeriesMatrix([X[:,:,i] for i in 1:size(X,3)], period; nperiod)
end
"""
    pgclyap2(A, C, E, [, K = 1]; solver, reltol, abstol) -> (X,Y)

Compute the solutions of the periodic differential Lyapunov equations

     -
     X(t) = A(t)*X(t) + X(t)*A'(t) + C(t)

and 

     .
    -Y(t) = A(t)'Y(t) + Y(t)A(t) + E(t).
    
The periodic matrices `A`, `C` and `E` must have the same dimensions, the same type and 
commensurate periods. Additionally `C` and `E` must be symmetric.  
If `A`, `C` and `E` have the types `PeriodicFunctionMatrix`, `HarmonicArray`, `FourierFunctionMatrix` or `PeriodicTimeSeriesMatrix`, 
then the resulting `X` and `Y` are collections of periodic generator matrices determined 
as periodic time-series matrices with `N` components, where `N = 1` if `A`, `C` and `E` are constant matrices
and `N = K` otherwise.     
The period `T` of `X` and `Y` is set to the least common commensurate period of `A`, `C` and `E` and the number of subperiods
is adjusted accordingly. Any component matrix of `X` or `Y` is a valid initial value to be used to generate the  
solution over a full period by integrating the appropriate differential equation. 
The multiple-shooting method of [1] is employed, first, to convert the continuous-time periodic Lyapunov equations 
into discrete-time periodic Lyapunov equations satisfied by 
the generator solutions in the grid points and then to compute the solutions by solving appropriate discrete-time periodic Lyapunov 
equations using the periodic Schur method of [2]. 

The ODE solver to be employed to convert the continuous-time problems into discrete-time problems can be specified using the keyword argument `solver`, 
together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`) and  
absolute accuracy `abstol` (default: `abstol = 1.e-7`).
Depending on the desired relative accuracy `reltol`, lower order solvers are employed for `reltol >= 1.e-4`, 
which are generally very efficient, but less accurate. If `reltol < 1.e-4`,
higher order solvers are employed able to cope with high accuracy demands. 

The following solvers from the [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) package can be selected:

`solver = "non-stiff"` - use a solver for non-stiff problems (`Tsit5()` or `Vern9()`);

`solver = "stiff"` - use a solver for stiff problems (`Rodas4()` or `KenCarp58()`);

`solver = "symplectic"` - use a symplectic Hamiltonian structure preserving solver (`IRKGL16()`);

`solver = "auto"` - use the default solver, which automatically detects stiff problems (`AutoTsit5(Rosenbrock23())` or `AutoVern9(Rodas5())`). 

Parallel computation of the matrices of the discrete-time problem can be alternatively performed 
by starting Julia with several execution threads. 
The number of execution threads is controlled either by using the `-t/--threads` command line argument 
or by using the `JULIA_NUM_THREADS` environment variable.  

_References_

[1] A. Varga. On solving periodic differential matrix equations with applications to periodic system norms computation. 
    Proc. IEEE CDC/ECC, Seville, 2005. 

[2] A. Varga. Periodic Lyapunov equations: some applications and new algorithms. 
    Int. J. Control, vol, 67, pp, 69-87, 1997.  
"""
function PeriodicMatrixEquations.pgclyap2(A::PM1, C::PM2, E::PM3, K::Int = 1; solver = "non-stiff", reltol = 1e-4, abstol = 1e-7, stability_check = false) where
      {PM1 <: FourierFunctionMatrix, PM2 <: FourierFunctionMatrix, PM3 <: FourierFunctionMatrix}
   K > 0 || throw(ArgumentError("number of grid ponts K must be greater than 0, got K = $K"))    
   period = promote_period(A, C, E)
   na = Int(round(period/A.period))
   nc = Int(round(period/C.period))
   ne = Int(round(period/E.period))
   nperiod = gcd(na*A.nperiod, nc*C.nperiod, ne*E.nperiod)
   n = size(A,1)
   Ts = period/K/nperiod
   solver == "symplectic" && dt == 0 && (dt = K >= 100 ? Ts : Ts*K/100/nperiod)
   
   T = promote_type(eltype(A),eltype(C),eltype(E),Float64)
   if PeriodicMatrices.isconstant(A) && PeriodicMatrices.isconstant(C) && PeriodicMatrices.isconstant(E)
      if stability_check
         ev = eigvals(tpmeval(A,0))
         maximum(real.(ev)) >= - sqrt(eps(T)) && error("system stability check failed")  
      end 
      X, Y = lyapc(tpmeval(A,0), tpmeval(C,0)),  lyapc(tpmeval(A,0)', tpmeval(E,0))
      #X, Y = MatrixEquations.lyapc2(tpmeval(A,0), tpmeval(C,0), tpmeval(E,0))
   else
      if stability_check
         ev = K < 100 ? PeriodicMatrices.pseig(A,100) : PeriodicMatrices.pseig(A,K)
         maximum(abs.(ev)) >= one(T) - sqrt(eps(T)) && error("system stability check failed")  
      end 
      Ka = PeriodicMatrices.isconstant(A) ? 1 : max(1,Int(round(A.period/A.nperiod/Ts)))
      Ad = Array{T,3}(undef, n, n, Ka) 
      Cd = Array{T,3}(undef, n, n, K) 
      Ed = Array{T,3}(undef, n, n, K) 
      Threads.@threads for i = 1:Ka
         @inbounds Ad[:,:,i] = tvstm(A, i*Ts, (i-1)*Ts; solver, reltol, abstol) 
      end
      Threads.@threads for i = 1:K
         @inbounds Cd[:,:,i]  = PeriodicMatrixEquations.tvclyap(A, C, i*Ts, (i-1)*Ts; adj = false, solver, reltol, abstol) 
      end
      Threads.@threads for i = K:-1:1
         @inbounds Ed[:,:,i]  = PeriodicMatrixEquations.tvclyap(A, E, (i-1)*Ts, i*Ts; adj = true, solver, reltol, abstol) 
      end
      X, Y = pslyapd2(Ad, Cd, Ed)
   end
   return PeriodicTimeSeriesMatrix([X[:,:,i] for i in 1:size(X,3)], period; nperiod), PeriodicTimeSeriesMatrix([Y[:,:,i] for i in 1:size(Y,3)], period; nperiod)
end
"""
    pgclyap2(A, C::AbstractMatrix, E, [, K = 1]; solver, reltol, abstol) -> (X,Y)

Compute the solution of the discrete-time periodic Lyapunov equation

    X(i+1) = Φ(i)*X(i)*Φ'(i) + W(i), i = 1, ..., K, X(K+1) := X(1)

and a periodic generator for the periodic Lyapunov differential equations

     .
    -Y(t) = A(t)'Y(t) + Y(t)A(t) + E(t).
    
The periodic matrices `A` and `E` and the constant matrix `C` must have the same dimensions, and `A` and `E` 
must have the same type and commensurate periods. Additionally `C` and `E` must be symmetric.  
`Φ(i)` denotes the transition matrix on the time interval `[Δ*(i-1), Δ*i]` corresponding to `A`, 
where `Δ = T/K` with `T` the common period of `A` and `E`. `W(i) = 0` for `i = 1, ..., K-1` and `W(K) = C`.  
If `A` and `E` have the types `PeriodicFunctionMatrix`, `HarmonicArray`, `FourierFunctionMatrix` or `PeriodicTimeSeriesMatrix`, 
then the resulting `Y` is a collection of periodic generator matrices determined 
as a periodic time-series matrix with `N` components, where `N = 1` if `A` and `E` are constant matrices
and `N = K` otherwise.     
The period `T` of `Y` is set to the least common commensurate period of `A` and `E` and the number of subperiods
is adjusted accordingly. Any component matrix of `Y` is a valid initial value to be used to generate the  
solution over a full period by integrating the appropriate differential equation. 
The multiple-shooting method of [1] is employed, first, to convert the continuous-time periodic Lyapunov into a discrete-time periodic Lyapunov equation satisfied by 
the generator solution in the grid points and then to compute the solution by solving an appropriate discrete-time periodic Lyapunov 
equation using the periodic Schur method of [2]. 

The ODE solver to be employed to convert the continuous-time problems into discrete-time problems can be specified using the keyword argument `solver`, 
together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`) and  
absolute accuracy `abstol` (default: `abstol = 1.e-7`).
Depending on the desired relative accuracy `reltol`, lower order solvers are employed for `reltol >= 1.e-4`, 
which are generally very efficient, but less accurate. If `reltol < 1.e-4`,
higher order solvers are employed able to cope with high accuracy demands. 

The following solvers from the [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) package can be selected:

`solver = "non-stiff"` - use a solver for non-stiff problems (`Tsit5()` or `Vern9()`);

`solver = "stiff"` - use a solver for stiff problems (`Rodas4()` or `KenCarp58()`);

`solver = "symplectic"` - use a symplectic Hamiltonian structure preserving solver (`IRKGL16()`);

`solver = "auto"` - use the default solver, which automatically detects stiff problems (`AutoTsit5(Rosenbrock23())` or `AutoVern9(Rodas5())`). 

Parallel computation of the matrices of the discrete-time problem can be alternatively performed 
by starting Julia with several execution threads. 
The number of execution threads is controlled either by using the `-t/--threads` command line argument 
or by using the `JULIA_NUM_THREADS` environment variable.  

_References_

[1] A. Varga. On solving periodic differential matrix equations with applications to periodic system norms computation. 
    Proc. IEEE CDC/ECC, Seville, 2005. 

[2] A. Varga. Periodic Lyapunov equations: some applications and new algorithms. 
    Int. J. Control, vol, 67, pp, 69-87, 1997.  
"""
function PeriodicMatrixEquations.pgclyap2(A::PM1, C::AbstractMatrix, E::PM3, K::Int = 1; solver = "non-stiff", reltol = 1e-4, abstol = 1e-7, stability_check = false) where
      {PM1 <: FourierFunctionMatrix, PM3 <: FourierFunctionMatrix}
   K > 0 || throw(ArgumentError("number of grid ponts K must be greater than 0, got K = $K"))    
   period = promote_period(A, E)
   na = Int(round(period/A.period))
   ne = Int(round(period/E.period))
   nperiod = gcd(na*A.nperiod, ne*E.nperiod)
   n = size(A,1)
   Ts = period/K/nperiod
   T = promote_type(eltype(A),eltype(C),eltype(E),Float64)
   if PeriodicMatrices.isconstant(A) && PeriodicMatrices.isconstant(E)
      A0 = tpmeval(A,0)
      if stability_check
         ev = eigvals(A0)
         maximum(real.(ev)) >= - sqrt(eps(T)) && error("system stability check failed")  
      end 
      Ad = exp(A0*period)
      X = lyapd(Ad,C)
      Y = lyapc(A0', tpmeval(E,0))
   else
      if stability_check
         ev = K < 100 ? PeriodicMatrices.pseig(A,100) : PeriodicMatrices.pseig(A,K)
         maximum(abs.(ev)) >= one(T) - sqrt(eps(T)) && error("system stability check failed")  
      end 
      Ka = PeriodicMatrices.isconstant(A) ? 1 : max(1,Int(round(A.period/A.nperiod/Ts)))
      Ad = Array{T,3}(undef, n, n, Ka) 
      Cd = zeros(T, n, n, K)
      Ed = Array{T,3}(undef, n, n, K) 
      Threads.@threads for i = 1:Ka
         @inbounds Ad[:,:,i] = tvstm(A, i*Ts, (i-1)*Ts; solver, reltol, abstol) 
      end
      copyto!(view(Cd,:,:,K),C)
      Threads.@threads for i = K:-1:1
         @inbounds Ed[:,:,i]  = PeriodicMatrixEquations.tvclyap(A, E, (i-1)*Ts, i*Ts; adj = true, solver, reltol, abstol) 
      end
      X, Y = pslyapd2(Ad, Cd, Ed)
   end
   return PeriodicTimeSeriesMatrix([X[:,:,i] for i in 1:size(X,3)], period; nperiod), PeriodicTimeSeriesMatrix([Y[:,:,i] for i in 1:size(Y,3)], period; nperiod)
end

"""
      tvclyap_eval(t, W, A, C; adj = false, solver, reltol, abstol) -> Xval

Compute the time value `Xval := X(t)` of the solution of the periodic Lyapunov differential equation

       .
       X(t) = A(t)X(t) + X(t)A(t)' + C(t) ,  X(t0) = W(t0), t > t0, if adj = false

or 

       .
      -X(t) = A(t)'X(t) + X(t)A(t) + C(t) ,  X(t0) = W(t0), t < t0, if adj = true,  

using the periodic generator `W` determined with the function [`pgclyap`](@ref) for the same periodic matrices `A` and `C`
and the same value of the keyword argument `adj`. 
The initial time `t0` is the nearest time grid value to `t`, from below, if `adj = false`, or from above, if `adj = true`. 

The above ODE is solved by employing the integration method specified via the keyword argument `solver`, 
together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`) and  
absolute accuracy `abstol` (default: `abstol = 1.e-7`).
Depending on the desired relative accuracy `reltol`, lower order solvers are employed for `reltol >= 1.e-4`, 
which are generally very efficient, but less accurate. If `reltol < 1.e-4`,
higher order solvers are employed able to cope with high accuracy demands. 

The following solvers from the [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) package can be selected:

`solver = "non-stiff"` - use a solver for non-stiff problems (`Tsit5()` or `Vern9()`);

`solver = "stiff"` - use a solver for stiff problems (`Rodas4()` or `KenCarp58()`);

`solver = "auto"` - use the default solver, which automatically detects stiff problems (`AutoTsit5(Rosenbrock23())` or `AutoVern9(Rodas5())`). 
"""
function PeriodicMatrixEquations.tvclyap_eval(t::Real,X::PeriodicTimeSeriesMatrix,A::PM1, C::PM2; adj = false, solver = "non-stiff", reltol = 1e-4, abstol = 1e-7) where
   {PM1 <: FourierFunctionMatrix, PM2 <: FourierFunctionMatrix} 
   tsub = X.period/X.nperiod
   ns = length(X.values)
   Δ = tsub/ns
   tf = mod(t,tsub)
   tf == 0 && (return X.values[1])
   if adj 
      ind = round(Int,tf/Δ)
      if ind == ns
         t0 = ind*Δ; ind = 1
      else
         t0 = (ind+1)*Δ; ind = ind+2; 
         ind > ns && (ind = 1) 
     end 
   else
      ind = round(Int,tf/Δ)
      ind == 0 && (ind = 1) 
      t0 = (ind-1)*Δ
   end
   return PeriodicMatrixEquations.tvclyap(A, C, tf, t0, X.values[ind]; adj, solver, reltol, abstol) 
end
function PeriodicMatrixEquations.tvclyap_eval(t::Real,X::PeriodicTimeSeriesMatrix,A::PM1; solver = "non-stiff", reltol = 1e-4, abstol = 1e-7) where
   {PM1 <: FourierFunctionMatrix} 
   tsub = X.period/X.nperiod
   ns = length(X.values)
   Δ = tsub/ns
   tf = mod(t,tsub)
   tf == 0 && (return X.values[1])
      ind = round(Int,tf/Δ)
      ind == 0 && (ind = 1) 
      t0 = (ind-1)*Δ
   #@show tf, t0
   return PeriodicMatrixEquations.tvclyap(A, FourierFunctionMatrix(zeros(eltype(A),size(A)...),A.period), tf, t0, X.values[ind]; adj = false, solver, reltol, abstol) 
end

function PeriodicMatrixEquations.tvclyap(A::PM1, C::PM2, tf, t0, W0::Union{AbstractMatrix,Missing} = missing; adj = false, solver = "auto", reltol = 1e-4, abstol = 1e-7) where
   {PM1 <: FourierFunctionMatrix, PM2 <: FourierFunctionMatrix} 
   #{PM1 <: Union{PeriodicFunctionMatrix,PeriodicSymbolicMatrix,HarmonicArray,FourierFunctionMatrix,PeriodicSwitchingMatrix}, PM2 <: Union{PeriodicFunctionMatrix,PeriodicSymbolicMatrix,HarmonicArray,FourierFunctionMatrix,PeriodicSwitchingMatrix}} 
   """
      tvclyap(A, C, tf, t0; adj, solver, reltol, abstol) -> W::Matrix

   Compute the solution at tf > t0 of the differential matrix Lyapunov equation 
            .
            W(t) = A(t)*W(t)+W(t)*A'(t)+C(t), W(t0) = 0, tf > t0, if adj = false

   or 
            .
            W(t) = -A(t)'*W(t)-W(t)*A(t)-C(t), W(t0) = 0, tf < t0, if adj = true. 

   The ODE solver to be employed to convert the continuous-time problem into a discrete-time problem can be specified using the keyword argument `solver`, 
   together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`) and  
   absolute accuracy `abstol` (default: `abstol = 1.e-7`). 
   """
   n = size(A,1)
   n == size(A,2) || error("the periodic matrix A must be square")
   (n,n) == size(C) || error("the periodic matrix C must have same dimensions as A")
   T = promote_type(typeof(t0), typeof(tf))
   # using OrdinaryDiffEq
   ismissing(W0) ? u0 = zeros(T,div(n*(n+1),2)) : u0 = MatrixEquations.triu2vec(W0)
   tspan = (T(t0),T(tf))
   fclyap!(du,u,p,t) = adj ? PeriodicMatrixEquations.muladdcsym!(du, u, -1, tpmeval(A,t)', tpmeval(C,t)) : PeriodicMatrixEquations.muladdcsym!(du, u, 1, tpmeval(A,t), tpmeval(C,t))
   prob = ODEProblem(fclyap!, u0, tspan)
   if solver == "stiff" 
      if reltol > 1.e-4  
         # standard stiff
         sol = solve(prob, Rodas4(); reltol, abstol, save_everystep = false)
      else
         # high accuracy stiff
         sol = solve(prob, KenCarp58(); reltol, abstol, save_everystep = false)
      end
   elseif solver == "non-stiff" 
      if reltol > 1.e-4  
         # standard non-stiff
         sol = solve(prob, Tsit5(); reltol, abstol, save_everystep = false)
      else
         # high accuracy non-stiff
         sol = solve(prob, Vern9(); reltol, abstol, save_everystep = false)
      end
   # elseif solver == "symplectic" 
   #    # high accuracy symplectic
   #    if dt == 0 
   #       sol = solve(prob, IRKGaussLegendre.IRKGL16(maxtrials=4); adaptive = true, reltol, abstol, save_everystep = false)
   #       #@show sol.retcode
   #       if sol.retcode == :Failure
   #          sol = solve(prob, IRKGaussLegendre.IRKGL16(); adaptive = false, reltol, abstol, save_everystep = false, dt = abs(tf-t0)/100)
   #       end
   #    else
   #         sol = solve(prob, IRKGaussLegendre.IRKGL16(); adaptive = false, reltol, abstol, save_everystep = false)
   #    end
   else 
      if reltol > 1.e-4  
         # low accuracy automatic selection
         sol = solve(prob, AutoTsit5(Rosenbrock23()) ; reltol, abstol, save_everystep = false)
      else
         # high accuracy automatic selection
         sol = solve(prob, AutoVern9(Rodas5(),nonstifftol = 11/10); reltol, abstol, save_everystep = false)
      end
   end
   return MatrixEquations.vec2triu(sol.u[end], her=true)     
end
function tvclyap_sol(A::PM1, C::PM2, tf, t0, W0::Union{AbstractMatrix,Missing} = missing; adj = false, solver = "auto", reltol = 1e-4, abstol = 1e-7) where
   {PM1 <: FourierFunctionMatrix, PM2 <: FourierFunctionMatrix} 
   """
      tvclyap_sol(A, C, tf, t0, W0; adj, solver, reltol, abstol) -> S(t)

   Compute the solution on the interval `[t0,tf]` of the differential matrix Lyapunov equation 
            .
            W(t) = A(t)*W(t)+W(t)*A'(t)+C(t), W(t0) = W0, tf > t0, if adj = false

   or on the interval `[tf,t0]` of
            .
            W(t) = -A(t)'*W(t)-W(t)*A(t)-C(t), W(t0) = W0, tf < t0, if adj = true. 

   The computed solution `S(t)` is the upper triangular part of the symmetric solution `W(t)`, as results using the
   solvers from the [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) package with the option
   `dense = true`.         

   The ODE solver to be employed to convert the continuous-time problem into a discrete-time problem can be specified using the keyword argument `solver`, 
   together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`) and  
   absolute accuracy `abstol` (default: `abstol = 1.e-7`). 
   """
   n = size(A,1)
   n == size(A,2) || error("the periodic matrix A must be square")
   (n,n) == size(C) || error("the periodic matrix C must have same dimensions as A")
   T = promote_type(typeof(t0), typeof(tf))
   # using OrdinaryDiffEq
   ismissing(W0) ? u0 = zeros(T,div(n*(n+1),2)) : u0 = MatrixEquations.triu2vec(W0)
   tspan = (T(t0),T(tf))
   fclyap!(du,u,p,t) = adj ? muladdcsym!(du, u, -1, tpmeval(A,t)', tpmeval(C,t)) : muladdcsym!(du, u, 1, tpmeval(A,t), tpmeval(C,t))
   prob = ODEProblem(fclyap!, u0, tspan)
   if solver == "stiff" 
      if reltol > 1.e-4  
         # standard stiff
         sol = solve(prob, Rodas4(); reltol, abstol, dense = true)
      else
         # high accuracy stiff
         sol = solve(prob, KenCarp58(); reltol, abstol, dense = true)
      end
   elseif solver == "non-stiff" 
      if reltol > 1.e-4  
         # standard non-stiff
         sol = solve(prob, Tsit5(); reltol, abstol, dense = true)
      else
         # high accuracy non-stiff
         sol = solve(prob, Vern9(); reltol, abstol, dense = true)
      end
   # elseif solver == "symplectic" 
   #    # high accuracy symplectic
   #    if dt == 0 
   #       sol = solve(prob, IRKGaussLegendre.IRKGL16(maxtrials=4); adaptive = true, reltol, abstol, dense = true)
   #       #@show sol.retcode
   #       if sol.retcode == :Failure
   #          sol = solve(prob, IRKGaussLegendre.IRKGL16(); adaptive = false, reltol, abstol, dense = true, dt = abs(tf-t0)/100)
   #       end
   #    else
   #         sol = solve(prob, IRKGaussLegendre.IRKGL16(); adaptive = false, reltol, abstol, dense = true)
   #    end
   else 
      if reltol > 1.e-4  
         # low accuracy automatic selection
         sol = solve(prob, AutoTsit5(Rosenbrock23()) ; reltol, abstol, dense = true)
      else
         # high accuracy automatic selection
         sol = solve(prob, AutoVern9(Rodas5(),nonstifftol = 11/10); reltol, abstol, dense = true)
      end
   end
   return sol
end

function pclyap_intpol(A::PM1, C::PM2, W0::PeriodicTimeSeriesMatrix; N::Int = length(W0), adj = false, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7) where
   {PM1 <: FourierFunctionMatrix, PM2 <: FourierFunctionMatrix} 
   K = length(W0)
   N < K || (N = K)
   while rem(K,N) !== 0
      N += 1
   end
   tsub = W0.period/W0.nperiod
   Ts = tsub/N
   Y = similar(Vector{Any},N)
   Ni = div(K,N)
   if adj 
      for k = N:-1:1   
          iw = mod(k*Ni-1,K)+2; 
          iw > K && (iw = 1)
          Y[k]  = PeriodicMatrixEquations.tvclyap_sol(A, C, (k-1)*Ts, k*Ts, W0.values[iw]; solver = "auto", adj = true, reltol = 1.e-10, abstol = 1.e-10) 
      end
   else
      for k = 1:N
          Y[k]  = PeriodicMatrixEquations.tvclyap_sol(A, C, k*Ts, (k-1)*Ts, W0.values[(k-1)*Ni+1]; adj, solver, reltol, abstol) 
      end
   end

   return PeriodicFunctionMatrix(t-> 
   begin
      tf = mod(t,tsub)
      ind = round(Int,tf/Ts-0.5)+1
      MatrixEquations.vec2triu(Y[ind](tf), her=true)   
   end, W0.period; W0.nperiod)  
end
function PeriodicMatrixEquations.tvclyap(A::PM1, C::PM2, ts::AbstractVector, W0::AbstractMatrix; adj = false, solver = "auto", reltol = 1e-4, abstol = 1e-7) where
   {PM1 <: FourierFunctionMatrix, PM2 <:FourierFunctionMatrix} 
   """
      tvclyap(A, C, ts, W0; adj, solver, reltol, abstol) -> W::Matrix

   Compute the solution at the time values ts of the differential matrix Lyapunov equation 
            .
            W(t) = A(t)*W(t)+W(t)*A'(t)+C(t), W(ts[1]) = W0, if adj = false

   or 
            .
            W(t) = -A(t)'*W(t)-W(t)*A(t)-C(t), W(ts[end]) = W0, tf < t0, if adj = true. 

   The ODE solver to be employed can be specified using the keyword argument `solver`, 
   together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`) and  
   absolute accuracy `abstol` (default: `abstol = 1.e-7`). 
   
   """
   n = size(A,1)
   n == size(A,2) || error("the periodic matrix A must be square")
   (n,n) == size(C) || error("the periodic matrix C must have same dimensions as A")
   all(diff(ts) .>= 0) || throw(ArgumentError("the elements of ts must be non-decreasing"))
   T = eltype(ts)
   # using OrdinaryDiffEq
   u0 = MatrixEquations.triu2vec(W0)
   if adj
      period = promote_period(A,C)
      n = round(ts[end]/period)+1
      tspan = (n*period, ts[1]) 
      tval = reverse(ts)
   else
      tspan = (0, ts[end]) 
      tval = ts
   end
   fclyap!(du,u,p,t) = adj ? PeriodicMatrixEquations.muladdcsym!(du, u, -1, tpmeval(A,t)', tpmeval(C,t)) : PeriodicMatrixEquations.muladdcsym!(du, u, 1, tpmeval(A,t), tpmeval(C,t))
   prob = ODEProblem(fclyap!, u0, tspan)
   if solver == "stiff" 
      if reltol > 1.e-4  
         # standard stiff
         sol = solve(prob, Rodas4(); reltol, abstol, saveat = tval)
      else
         # high accuracy stiff
         sol = solve(prob, KenCarp58(); reltol, abstol, saveat = tval)
      end
   elseif solver == "non-stiff" 
      if reltol > 1.e-4  
         # standard non-stiff
         sol = solve(prob, Tsit5(); reltol, abstol, saveat = tval)
      else
         # high accuracy non-stiff
         sol = solve(prob, Vern9(); reltol, abstol, saveat = tval)
      end
   # elseif solver == "symplectic" 
   #    # high accuracy symplectic
   #    if dt == 0 
   #       #sol = solve(prob, IRKGaussLegendre.IRKGL16(maxtrials=4); adaptive = true, reltol, abstol, saveat = ts)
   #       sol = solve(prob, IRKGaussLegendre.IRKGL16(maxtrials=4); adaptive = true, reltol, abstol, dense = true)
   #       if sol.retcode == :Failure
   #          #sol = solve(prob, IRKGaussLegendre.IRKGL16(); adaptive = false, reltol, abstol, saveat = ts, dt = abs(tf-t0)/100)
   #          sol = solve(prob, IRKGaussLegendre.IRKGL16(); adaptive = false, reltol, abstol, dense = true, dt = abs(tf-t0)/100)
   #       end
   #    else
   #       #sol = solve(prob, IRKGaussLegendre.IRKGL16(); adaptive = false, reltol, abstol, saveat = ts)
   #       sol = solve(prob, IRKGaussLegendre.IRKGL16(); adaptive = false, reltol, abstol, dense = true)
   #    end
   #    return MatrixEquations.vec2triu.(sol(ts).u, her=true)     
   else 
      if reltol > 1.e-4  
         # low accuracy automatic selection
         sol = solve(prob, AutoTsit5(Rosenbrock23()) ; reltol, abstol, saveat = tval)
      else
         # high accuracy automatic selection
         sol = solve(prob, AutoVern9(Rodas5(),nonstifftol = 11/10); reltol, abstol, saveat = tval)
      end
   end
   W = MatrixEquations.vec2triu.(sol.u, her=true)
   return adj ? reverse(W) : W    
end
function PeriodicMatrixEquations.tvclyap(A::PM1, C::PM2, W0::AbstractMatrix; adj = false, solver = "auto", reltol = 1e-4, abstol = 1e-7) where
   {PM1 <: FourierFunctionMatrix, PM2 <: FourierFunctionMatrix} 
   #{PM1 <: Union{PeriodicFunctionMatrix,PeriodicSymbolicMatrix,HarmonicArray,FourierFunctionMatrix,PeriodicSwitchingMatrix}, PM2 <: Union{PeriodicFunctionMatrix,PeriodicSymbolicMatrix,HarmonicArray,FourierFunctionMatrix,PeriodicSwitchingMatrix}} 
   """
      tvclyap(A, C, ts, W0; adj, solver, reltol, abstol) -> W::Matrix

   Compute the solution at the time values ts of the differential matrix Lyapunov equation 
            .
            W(t) = A(t)*W(t)+W(t)*A'(t)+C(t), W(ts[1]) = W0, if adj = false

   or 
            .
            W(t) = -A(t)'*W(t)-W(t)*A(t)-C(t), W(ts[end]) = W0, tf < t0, if adj = true. 

   The ODE solver to be employed can be specified using the keyword argument `solver`, 
   together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`) and  
   absolute accuracy `abstol` (default: `abstol = 1.e-7`). 
   Depending on the desired relative accuracy `reltol`, lower order solvers are employed for `reltol >= 1.e-4`, 
   which are generally very efficient, but less accurate. If `reltol < 1.e-4`,
   higher order solvers are employed able to cope with high accuracy demands. 

   The following solvers from the [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) package can be selected:

   `solver = "non-stiff"` - use a solver for non-stiff problems (`Tsit5()` or `Vern9()`);

   `solver = "stiff"` - use a solver for stiff problems (`Rodas4()` or `KenCarp58()`);

   `solver = "auto"` - use the default solver, which automatically detects stiff problems (`AutoTsit5(Rosenbrock23())` or `AutoVern9(Rodas5())`). 
   """
   n = size(A,1)
   n == size(A,2) || error("the periodic matrix A must be square")
   (n,n) == size(C) || error("the periodic matrix C must have same dimensions as A")
   period = A.period
   T = eltype(period)
   # using OrdinaryDiffEq
   u0 = MatrixEquations.triu2vec(W0)
   tspan = adj ? (period, zero(T)) : (zero(T), period) 
   fclyap!(du,u,p,t) = adj ? PeriodicMatrixEquations.muladdcsym!(du, u, -1, tpmeval(A,t)', tpmeval(C,t)) : PeriodicMatrixEquations.muladdcsym!(du, u, 1, tpmeval(A,t), tpmeval(C,t))
   prob = ODEProblem(fclyap!, u0, tspan)
   if solver == "stiff" 
      if reltol > 1.e-4  
         # standard stiff
         sol = solve(prob, Rodas4(); reltol, abstol, save_everystep = true)
      else
         # high accuracy stiff
         sol = solve(prob, KenCarp58(); reltol, abstol, save_everystep = true)
      end
   elseif solver == "non-stiff" 
      if reltol > 1.e-4  
         # standard non-stiff
         sol = solve(prob, Tsit5(); reltol, abstol, save_everystep = true)
      else
         # high accuracy non-stiff
         sol = solve(prob, Vern9(); reltol, abstol, save_everystep = true)
      end
   # elseif solver == "symplectic" 
   #    # high accuracy symplectic
   #    if dt == 0 
   #       sol = solve(prob, IRKGaussLegendre.IRKGL16(maxtrials=4); adaptive = true, reltol, abstol, save_everystep = true)
   #       #@show sol.retcode
   #       if sol.retcode == :Failure
   #          sol = solve(prob, IRKGaussLegendre.IRKGL16(); adaptive = false, reltol, abstol, save_everystep = true, dt = abs(tf-t0)/100)
   #       end
   #    else
   #         sol = solve(prob, IRKGaussLegendre.IRKGL16(); adaptive = false, reltol, abstol, save_everystep = true)
   #    end
   else 
      if reltol > 1.e-4  
         # low accuracy automatic selection
         sol = solve(prob, AutoTsit5(Rosenbrock23()) ; reltol, abstol, save_everystep = true)
      else
         # high accuracy automatic selection
         sol = solve(prob, AutoVern9(Rodas5(),nonstifftol = 11/10); reltol, abstol, save_everystep = true)
      end
   end
   return PeriodicFunctionMatrix(t-> MatrixEquations.vec2triu(sol(t), her=true),period)     
end
function PeriodicMatrixEquations.tvclyap(A::PM1, C::PM2, W::PeriodicTimeSeriesMatrix; adj = false, solver = "auto", reltol = 1e-4, abstol = 1e-7) where
   {PM1 <: FourierFunctionMatrix, PM2 <: FourierFunctionMatrix} 
   """
      tvclyap(A, C, G; adj, solver, reltol, abstol) -> S(t)

   Compute the solution at the time values between [0, period] of the differential matrix Lyapunov equation 
            .
            W(t) = A(t)*W(t)+W(t)*A'(t)+C(t), W(0) = G(0), if adj = false

   or 
            .
            W(t) = -A(t)'*W(t)-W(t)*A(t)-C(t), W(tsub) = G(0), if adj = true,
            
   where `G` is a periodic generator computed by `pgclyap`. 
   The resulting periodic function matrix `W(t)` provides the values 
   of the solution at an arbitrary time value `t` using built-in interpolation formulas
   withing the `OrdinaryDiffEq` package.       

   The ODE solver to be employed can be specified using the keyword argument `solver`, 
   together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`) and  
   absolute accuracy `abstol` (default: `abstol = 1.e-7`). 
   Depending on the desired relative accuracy `reltol`, lower order solvers are employed for `reltol >= 1.e-4`, 
   which are generally very efficient, but less accurate. If `reltol < 1.e-4`,
   higher order solvers are employed able to cope with high accuracy demands. 

   The following solvers from the [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) package can be selected:

   `solver = "non-stiff"` - use a solver for non-stiff problems (`Tsit5()` or `Vern9()`);

   `solver = "stiff"` - use a solver for stiff problems (`Rodas4()` or `KenCarp58()`);

   `solver = "auto"` - use the default solver, which automatically detects stiff problems (`AutoTsit5(Rosenbrock23())` or `AutoVern9(Rodas5())`). 
   """
   n = size(A,1)
   n == size(A,2) || error("the periodic matrix A must be square")
   (n,n) == size(C) || error("the periodic matrix C must have same dimensions as A")
   (n,n) == size(W) || error("the periodic time series matrix W must have same dimensions as A")
   period = W.period
   nperiod = W.nperiod
   (period, nperiod) == promote_period2(A,C) || error("inconsistent period and nperiod between (A, C) and W")
   T = eltype(period)
   # using OrdinaryDiffEq
   u0 = MatrixEquations.triu2vec(W(0))
   tsub = period/nperiod
   tspan = adj ? (tsub, zero(T)) : (zero(T), tsub) 
   fclyap!(du,u,p,t) = adj ? PeriodicMatrixEquations.muladdcsym!(du, u, -1, tpmeval(A,t)', tpmeval(C,t)) : PeriodicMatrixEquations.muladdcsym!(du, u, 1, tpmeval(A,t), tpmeval(C,t))
   prob = ODEProblem(fclyap!, u0, tspan)
   if solver == "stiff" 
      if reltol > 1.e-4  
         # standard stiff
         sol = solve(prob, Rodas4(); reltol, abstol, dense = true)
      else
         # high accuracy stiff
         sol = solve(prob, KenCarp58(); reltol, abstol, dense = true)
      end
   elseif solver == "non-stiff" 
      if reltol > 1.e-4  
         # standard non-stiff
         sol = solve(prob, Tsit5(); reltol, abstol, dense = true)
      else
         # high accuracy non-stiff
         sol = solve(prob, Vern9(); reltol, abstol, dense = true)
      end
   # elseif solver == "symplectic" 
   #    # high accuracy symplectic
   #    if dt == 0 
   #       sol = solve(prob, IRKGaussLegendre.IRKGL16(maxtrials=4); adaptive = true, reltol, abstol, dense = true)
   #       #@show sol.retcode
   #       if sol.retcode == :Failure
   #          sol = solve(prob, IRKGaussLegendre.IRKGL16(); adaptive = false, reltol, abstol, dense = true, dt = abs(tf-t0)/100)
   #       end
   #    else
   #         sol = solve(prob, IRKGaussLegendre.IRKGL16(); adaptive = false, reltol, abstol, dense = true)
   #    end
   else 
      if reltol > 1.e-4  
         # low accuracy automatic selection
         sol = solve(prob, AutoTsit5(Rosenbrock23()) ; reltol, abstol, dense = true)
      else
         # high accuracy automatic selection
         sol = solve(prob, AutoVern9(Rodas5(),nonstifftol = 11/10); reltol, abstol, dense = true)
      end
   end
   return PeriodicFunctionMatrix(t-> MatrixEquations.vec2triu(sol(mod(t,tsub)), her=true),period; nperiod)     
end
function PeriodicMatrixEquations.pcplyap(A::FourierFunctionMatrix, C::FourierFunctionMatrix; K::Int = 10, adj = false, intpol = true, solver = "non-stiff", reltol = 1.e-7, abstol = 1.e-7)
   # if intpol
   #    W0 = PeriodicMatrixEquations.pgclyap(A, adj ? C'*C : C*C', K;  adj, solver, reltol, abstol, stability_check = true)
   #    return PeriodicMatrixEquations.tvcplyap(A, C, W0; adj, solver, reltol, abstol)
   # else
   #    U = PeriodicMatrixEquations.pgcplyap(A, C, K;  adj, solver, reltol, abstol)
   #    PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvcplyap_eval(t, U, A, C; solver, adj, reltol, abstol), U.period; nperiod = U.nperiod)
   # end
   if intpol
      if PeriodicMatrices.isconstant(A) && PeriodicMatrices.isconstant(C) 
         U = PeriodicMatrixEquations.pgcplyap(A,  C, K;  adj, solver, reltol, abstol)
         return PeriodicFunctionMatrix(U(0),U.period,nperiod=U.nperiod)
      end
      W0 = PeriodicMatrixEquations.pgclyap(A, adj ? C'*C : C*C', K;  adj, solver, reltol, abstol, stability_check = true)
      return PeriodicMatrixEquations.tvcplyap(A, C, W0; adj, solver, reltol, abstol)
   else
      U = PeriodicMatrixEquations.pgcplyap(A,  C, K;  adj, solver, reltol, abstol)
      if PeriodicMatrices.isconstant(U) 
         (K > 1 || (PeriodicMatrices.isconstant(A) && PeriodicMatrices.isconstant(C) )) && (return PeriodicFunctionMatrix(U(0),U.period,nperiod=U.nperiod))
      end
      PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvcplyap_eval(t, U, A, C; solver, adj, reltol, abstol), U.period; nperiod = U.nperiod)
   end
end
# function PeriodicMatrixEquations.pcplyap(A::FourierFunctionMatrix, C::FourierFunctionMatrix; K::Int = 10, adj = false, solver = "non-stiff", reltol = 1.e-7, abstol = 1.e-7)
#    convert(FourierFunctionMatrix, PeriodicMatrixEquations.pgcplyap(A,  C, K;  adj, solver, reltol, abstol))
# end
for PM in (:FourierFunctionMatrix, )
   @eval begin
      function PeriodicMatrixEquations.prcplyap(A::$PM, C::$PM; K::Int = 10, solver = "non-stiff", reltol = 1.e-7, abstol = 1.e-7) 
         PeriodicMatrixEquations.pcplyap(A, C; K, adj = true, solver, reltol, abstol)
      end
      function PeriodicMatrixEquations.prcplyap(A::$PM, C::AbstractMatrix; kwargs...)
         #PeriodicMatrixEquations.prcplyap(A, $PM(C, A.period; nperiod = A.nperiod); kwargs...)
         PeriodicMatrixEquations.prcplyap(A, $PM(C, A.period); kwargs...)
      end
      function PeriodicMatrixEquations.pfcplyap(A::$PM, C::$PM; K::Int = 10, solver = "non-stiff", reltol = 1.e-7, abstol = 1.e-7) 
         PeriodicMatrixEquations.pcplyap(A, C; K, adj = false, solver, reltol, abstol)
      end
      function PeriodicMatrixEquations.pfcplyap(A::$PM, C::AbstractMatrix; K::Int = 10, solver = "non-stiff", reltol = 1.e-7, abstol = 1.e-7) 
         #PeriodicMatrixEquations.pfcpyap(A, $PM(C, A.period; nperiod = A.nperiod); kwargs...)
         PeriodicMatrixEquations.pcplyap(A, $PM(C, A.period); K, adj = false, solver, reltol, abstol)
      end
   end
end
"""
    pgcplyap(A, C[, K = 1]; adj = false, solver, reltol, abstol) -> U

Compute upper triangular periodic generators `U(t)` of the solution `X(t) = U(t)U(t)'` of the 
periodic Lyapunov differential equation

    .
    X(t) = A(t)X(t) + X(t)A(t)' + C(t)C(t)' , if adj = false,

or of the solution `X(t) = U(t)'U(t)` of the periodic Lyapunov differential equation

     .
    -X(t) = A(t)'X(t) + X(t)A(t) + C(t)'C(t) , if adj = true.
    
The periodic matrices `A` and `C` must have the same type, commensurate periods and `A` must be stable.
The resulting `U` is a collection of periodic generator matrices determined 
as a periodic time-series matrix with `N` components, where `N = 1` if `A` and `C` are constant matrices
and `N = K` otherwise. 
The period of `U` is set to the least common commensurate period of `A` and `C` and the number of subperiods
is adjusted accordingly. Any component matrix of `U` is a valid initial value to be used to generate the  
solution over a full period by integrating the appropriate differential equation. 
An extension of the multiple-shooting method of [1] is employed, first, to convert the continuous-time periodic Lyapunov 
into a discrete-time periodic Lyapunov equation satisfied by 
the generator solution in `K` time grid points and then to compute the solution by solving an appropriate discrete-time periodic Lyapunov 
equation using the iterative method (Algorithm 5) of [2]. 

The ODE solver to be employed to convert the continuous-time problem into a discrete-time problem can be specified using the keyword argument `solver`, 
together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`) and  
absolute accuracy `abstol` (default: `abstol = 1.e-7`).
Depending on the desired relative accuracy `reltol`, lower order solvers are employed for `reltol >= 1.e-4`, 
which are generally very efficient, but less accurate. If `reltol < 1.e-4`,
higher order solvers are employed able to cope with high accuracy demands. 

The following solvers from the [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) package can be selected:

`solver = "non-stiff"` - use a solver for non-stiff problems (`Tsit5()` or `Vern9()`);

`solver = "stiff"` - use a solver for stiff problems (`Rodas4()` or `KenCarp58()`);

`solver = "auto"` - use the default solver, which automatically detects stiff problems (`AutoTsit5(Rosenbrock23())` or `AutoVern9(Rodas5())`). 

For large values of `K`, parallel computation of the matrices of the discrete-time problem can be alternatively performed 
by starting Julia with several execution threads. 
The number of execution threads is controlled either by using the `-t/--threads` command line argument 
or by using the `JULIA_NUM_THREADS` environment variable.  

_References_

[1] A. Varga. On solving periodic differential matrix equations with applications to periodic system norms computation. 
    Proc. IEEE CDC/ECC, Seville, 2005. 

[2] A. Varga. Periodic Lyapunov equations: some applications and new algorithms. 
    Int. J. Control, vol, 67, pp, 69-87, 1997.
    
"""
function PeriodicMatrixEquations.pgcplyap(A::PM1, C::PM2, K::Int = 1; adj = false, solver = "auto", reltol = 1e-7, abstol = 1e-7) where
      {PM1 <: FourierFunctionMatrix, PM2 <: FourierFunctionMatrix} 
   period = promote_period(A, C)
   na = Int(round(period/A.period))
   nc = Int(round(period/C.period))
   nperiod = gcd(na*A.nperiod, nc*C.nperiod)
   n = size(A,1)
   Ts = period/K/nperiod
   
   if PeriodicMatrices.isconstant(A) && PeriodicMatrices.isconstant(C)
      U = adj ? plyapc(tpmeval(A,0)', tpmeval(C,0)') :  plyapc(tpmeval(A,0), tpmeval(C,0))
   else
      T = promote_type(eltype(A),eltype(C),Float64)
      T == Num && (T = Float64)
      Ka = PeriodicMatrices.isconstant(A) ? 1 : max(1,Int(round(A.period/A.nperiod/Ts)))
      Ad = Array{T,3}(undef, n, n, Ka) 
      Cd = Array{T,3}(undef, n, n, K) 
      Threads.@threads for i = 1:Ka
         @inbounds Ad[:,:,i] = tvstm(A, i*Ts, (i-1)*Ts; solver, reltol, abstol) 
      end
      if adj
         #Threads.@threads for i = K:-1:1
         for i = K:-1:1
             Xd = PeriodicMatrixEquations.tvclyap(A, C'*C, (i-1)*Ts, i*Ts; adj, solver, reltol, abstol) 
             Fd = cholesky(Xd, RowMaximum(), check = false)
             Cd[:,:,i] = [Fd.U[1:Fd.rank, invperm(Fd.p)]; zeros(T,n-Fd.rank,n)]
         end
         U = psplyapd(Ad, Cd; adj)
      else
         #Threads.@threads for i = 1:K
         for i = 1:K
             Xd = PeriodicMatrixEquations.tvclyap(A, C*C', i*Ts, (i-1)*Ts; adj, reltol, abstol) 
             Fd = cholesky(Xd, RowMaximum(), check = false)
             Cd[:,:,i] = [Fd.L[invperm(Fd.p), 1:Fd.rank] zeros(T,n,n-Fd.rank)]
         end
         U = psplyapd(Ad, Cd; adj)
      end
   end
   return PeriodicTimeSeriesMatrix([U[:,:,i] for i in 1:size(U,3)], period; nperiod)
end
"""
     tvcplyap_eval(t, U, A, C; adj = false, solver, reltol, abstol) -> Uval


Compute the time value `Uval := U(t)` of the upper triangular periodic generators 
`U(t)` of the solution `X(t) = U(t)U(t)'` of the periodic Lyapunov differential equation

      .
      X(t) = A(t)X(t) + X(t)A(t)' + C(t)C(t)' , X(t0) = U(t0)U(t0)', t > t0, if adj = false,

or of the solution `X(t) = U(t)'U(t)` of the periodic Lyapunov differential equation

      .
     -X(t) = A(t)'X(t) + X(t)A(t) + C(t)'C(t) , X(t0) = U(t0)'U(t0), t < t0, if adj = true,

using the periodic generator `U` determined with the function [`pgcplyap`](@ref) for the same periodic matrices `A` and `C`
and the same value of the keyword argument `adj`. 
The initial time `t0` is the nearest time grid value to `t`, from below, if `adj = false`, or from above, if `adj = true`. 

The above ODE is solved by employing the integration method specified via the keyword argument `solver`, 
together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`) and  
absolute accuracy `abstol` (default: `abstol = 1.e-7`).
Depending on the desired relative accuracy `reltol`, lower order solvers are employed for `reltol >= 1.e-4`, 
which are generally very efficient, but less accurate. If `reltol < 1.e-4`,
higher order solvers are employed able to cope with high accuracy demands. 

The following solvers from the [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) package can be selected:

`solver = "non-stiff"` - use a solver for non-stiff problems (`Tsit5()` or `Vern9()`);

`solver = "stiff"` - use a solver for stiff problems (`Rodas4()` or `KenCarp58()`);

`solver = "auto"` - use the default solver, which automatically detects stiff problems (`AutoTsit5(Rosenbrock23())` or `AutoVern9(Rodas5())`). 
"""
function PeriodicMatrixEquations.tvcplyap_eval(t::Real,U::PeriodicTimeSeriesMatrix,A::PM1, C::PM2; adj = false, solver = "non-stiff", reltol = 1e-4, abstol = 1e-7) where
   {PM1 <: FourierFunctionMatrix, PM2 <: FourierFunctionMatrix} 
   tsub = U.period/U.nperiod
   ns = length(U.values)
   Δ = tsub/ns
   tf = mod(t,tsub)
   tf == 0 && (return U.values[1])
   if adj 
      ind = round(Int,tf/Δ)
      if ind == ns
         t0 = ind*Δ; ind = 1
      else
         t0 = (ind+1)*Δ; ind = ind+2; 
         ind > ns && (ind = 1) 
     end 
   else
      ind = round(Int,tf/Δ)
      ind == 0 && (ind = 1) 
      t0 = (ind-1)*Δ
   end
   n = size(A,1)
   T = eltype(U)
   # use fallback method
   Q = adj ? C'*C : C*C'
   X0 = adj ? U.values[ind]'*U.values[ind] : U.values[ind]*U.values[ind]' 
   Xd = PeriodicMatrixEquations.tvclyap(A, Q, tf, t0, X0; adj, solver, reltol, abstol) 
   if adj
      Fd = cholesky(Xd, RowMaximum(), check = false)
      return PeriodicMatrixEquations.makesp!([qr(Fd.U[1:Fd.rank, invperm(Fd.p)]).R; zeros(T,n-Fd.rank,n)];adj)
   else
      Fd = cholesky(Xd, RowMaximum(), check = false)
      return PeriodicMatrixEquations.makesp!(triu(LAPACK.gerqf!([Fd.L[invperm(Fd.p), 1:Fd.rank] zeros(T,n,n-Fd.rank)], similar(Xd,n))[1]);adj)
   end
end
function PeriodicMatrixEquations.tvcplyap(A::PM1, C::PM2, U::PeriodicTimeSeriesMatrix; adj = false, solver = "non-stiff", reltol = 1e-4, abstol = 1e-7) where
   {PM1 <: FourierFunctionMatrix, PM2 <: FourierFunctionMatrix} 
   # tsub = U.period/U.nperiod
   # ns = length(U.values)
   # Δ = tsub/ns
   # tf = mod(t,tsub)
   # tf == 0 && (return U.values[1])
   # if adj 
   #    ind = round(Int,tf/Δ)
   #    if ind == ns
   #       t0 = ind*Δ; ind = 1
   #    else
   #       t0 = (ind+1)*Δ; ind = ind+2; 
   #       ind > ns && (ind = 1) 
   #   end 
   # else
   #    ind = round(Int,tf/Δ)
   #    ind == 0 && (ind = 1) 
   #    t0 = (ind-1)*Δ
   # end
   n = size(A,1)
   T = eltype(U)
   # use fallback method
   Q = adj ? C'*C : C*C'
   #X0 = adj ? U.values[ind]'*U.values[ind] : U.values[ind]*U.values[ind]' 
   Xd = PeriodicMatrixEquations.tvclyap(A, Q, U; adj, solver, reltol, abstol) 
   return PeriodicFunctionMatrix(t-> 
   begin
      if adj
         Fd = cholesky(Xd(t), RowMaximum(), check = false)
         PeriodicMatrixEquations.makesp!([qr(Fd.U[1:Fd.rank, invperm(Fd.p)]).R; zeros(T,n-Fd.rank,n)];adj)
      else
         Fd = cholesky(Xd(t), RowMaximum(), check = false)
         PeriodicMatrixEquations.makesp!(triu(LAPACK.gerqf!([Fd.L[invperm(Fd.p), 1:Fd.rank] zeros(T,n,n-Fd.rank)], similar(Vector{T},n))[1]);adj)
      end
   end, Xd.period; Xd.nperiod)  
end

    