"""
    pcsylv(A, C; K = 10, N = 1, adj = false, solver, reltol, abstol, intpol -> X

Solve the periodic Sylvester differential equation

    .
    X(t) = A(t)X(t) + X(t)A(t)' + C(t) , if adj = false,

or 

     .
    -X(t) = A(t)'X(t) + X(t)A(t) + C(t) , if adj = true.               

The periodic matrices `A`, `B` and `C` must have the same type, compatible dimensions and commensurate periods. 
Additionally, `A` and `B` must be square. 
The resulting symmetric periodic solution `X` has the type `PeriodicFunctionMatrix` and 
`X(t)` can be used to evaluate the value of `X` at time `t`. 
`X` has the period set to the least common commensurate period of `A`, `B`  and `C` and the number of subperiods
is adjusted accordingly. 

The multiple-shooting method of [1] is employed to convert the (continuous-time) periodic differential Sylvester equation 
into a discrete-time periodic Sylvester equation satisfied by a multiple point generator of the solution. 
The keyword argument `K` specifies the number of grid points to be used
for the discretization of the continuous-time problem (default: `K = 10`). 
If  `A`, `B`  and `C` are of types `PeriodicTimeSeriesMatrix` or `PeriodicSwitchingMatrix`, then `K` specifies the number of grid points used between two consecutive switching time values (default: `K = 1`).  
The multiple point periodic generator is computed  by solving the appropriate discrete-time periodic Sylvester 
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

[2] R. Byers and N. Rhee, “Cyclic Schur and Hessenberg-Schur numerical
    methods for solving periodic Sylvester and Sylvester equations,” Dept. of Mathematics, 
    Univ. of Missouri at Kansas City, Tech. Rep., June 1995.        
"""
function PeriodicMatrixEquations.pcsylv(A::PM, B::PM, C::PM; K::Int = 10, N::Int = 1, adj = false, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7, intpol = true) where {PM <: FourierFunctionMatrix}
   W0 = PeriodicMatrixEquations.pgcsylv(A, B, C, K;  adj, solver, reltol, abstol)
   if PeriodicMatrices.isconstant(W0) 
      (K > 1 || (PeriodicMatrices.isconstant(A) && PeriodicMatrices.isconstant(B) && PeriodicMatrices.isconstant(C) )) && (return PeriodicFunctionMatrix(W0(0),W0.period,nperiod=W0.nperiod))
   end
   if intpol
      return PeriodicMatrixEquations.pcsylv_intpol(A, B, C, W0; N, adj, solver, reltol, abstol)
   else
      return PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvcsylv_eval(t, W0, A, B, C; solver, adj, reltol, abstol),A.period)
   end
end


for PM in (FourierFunctionMatrix, )
   @eval begin
      function PeriodicMatrixEquations.prcsylv(A::$PM, B::$PM, C::$PM; K::Int = 10, N::Int = 1, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7, intpol = true) 
         PeriodicMatrixEquations.pcsylv(A, B, C; K, N, adj = true, solver, reltol, abstol, intpol)
      end
      function PeriodicMatrixEquations.prcsylv(A::$PM, B::$PM, C::AbstractMatrix; kwargs...)
         PeriodicMatrixEquations.prcsylv(A, B, $PM(C, A.period); kwargs...)
      end
      function PeriodicMatrixEquations.pfcsylv(A::$PM, B::$PM, C::$PM; K::Int = 10, N::Int = 1, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7, intpol = true) 
         PeriodicMatrixEquations.pcsylv(A, B, C; K, N, adj = false, solver, reltol, abstol, intpol)
      end
      function PeriodicMatrixEquations.pfcsylv(A::$PM, B::$PM, C::AbstractMatrix; kwargs...)
         #PeriodicMatrixEquations.pfcsylv(A, $PM(C, A.period; nperiod = A.nperiod); kwargs...)
         PeriodicMatrixEquations.pfcsylv(A, B, $PM(C, A.period); kwargs...)
      end
   end
end

"""
    pgcsylv(A, B, C[, K = 1]; adj = false, solver, reltol, abstol) -> X

Compute periodic generators for the periodic Sylvester differential equation

    .
    X(t) = A(t)X(t) + X(t)B(t) + C(t) , if adj = false,

or 

     .
    -X(t) = A(t)X(t) + X(t)B(t) + C(t) , if adj = true.
    
The periodic matrices `A`, `B`  and `C` must have the same type, compatible dimensions and commensurate periods, 
and additionally `C` must be symmetric. 
The resulting `X` is a collection of periodic generator matrices determined 
as a periodic time-series matrix with `N` components, where `N = 1` if `A`, `B`  and `C` are constant matrices
and `N = K` otherwise. 
The period of `X` is set to the least common commensurate period of `A`, `B`  and `C` and the number of subperiods
is adjusted accordingly. Any component matrix of `X` is a valid initial value to be used to generate the  
solution over a full period by integrating the appropriate differential equation. 
The multiple-shooting method of [1] is employed, first, to convert the continuous-time periodic Sylvester differential equation 
into a discrete-time periodic Sylvester equation satisfied by 
the generator solution in the grid points and then to compute the solution by solving an appropriate discrete-time periodic Sylvester 
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

[2] R. Byers and N. Rhee, “Cyclic Schur and Hessenberg-Schur numerical
    methods for solving periodic Sylvester and Sylvester equations,” Dept. of Mathematics, 
    Univ. of Missouri at Kansas City, Tech. Rep., June 1995.        
    
"""
function PeriodicMatrixEquations.pgcsylv(A::PM1, B::PM2, C::PM3, K::Int = 1; adj = false, solver = "non-stiff", reltol = 1e-4, abstol = 1e-7) where
      {PM1 <: FourierFunctionMatrix, PM2 <:FourierFunctionMatrix, PM3 <:FourierFunctionMatrix} 
   K > 0 || throw(ArgumentError("number of grid ponts K must be greater than 0, got K = $K"))    
   period = promote_period(A, C)
   na = Int(round(period/A.period))
   nb = Int(round(period/B.period))
   nc = Int(round(period/C.period))
   nperiod = gcd(na*A.nperiod, nb*B.nperiod, nc*C.nperiod)
   m = size(A,1)
   n = size(B,1)
   Ts = period/K/nperiod
   
   T = promote_type(eltype(A),eltype(B),eltype(C),Float64)
   if PeriodicMatrices.isconstant(A) && PeriodicMatrices.isconstant(B) && PeriodicMatrices.isconstant(C)
      X = sylvc(tpmeval(A,0), tpmeval(B,0), -tpmeval(C,0)) 
   else
      Ka = PeriodicMatrices.isconstant(A) ? 1 : max(1,Int(round(A.period/A.nperiod/Ts)))
      Kb = PeriodicMatrices.isconstant(B) ? 1 : max(1,Int(round(B.period/B.nperiod/Ts)))
      Ad = Array{T,3}(undef, m, m, Ka) 
      Bd = Array{T,3}(undef, n, n, Kb) 
      Cd = Array{T,3}(undef, m, n, K) 
      if adj
         Threads.@threads for i = 1:Ka
            @inbounds Ad[:,:,i] = tvstm(transpose(A), i*Ts, (i-1)*Ts; solver, reltol, abstol) 
         end
         Threads.@threads for i = 1:Kb
            @inbounds Bd[:,:,i] = tvstm(B, i*Ts, (i-1)*Ts; solver, reltol, abstol) 
         end
         Threads.@threads for i = K:-1:1
            @inbounds Cd[:,:,i]  = tvcsylv(A, B, C, (i-1)*Ts, i*Ts; adj, solver, reltol, abstol) 
         end
      else
         Threads.@threads for i = 1:Ka
            @inbounds Ad[:,:,i] = tvstm(A, i*Ts, (i-1)*Ts; solver, reltol, abstol) 
         end
         Threads.@threads for i = 1:Kb
            @inbounds Bd[:,:,i] = tvstm(transpose(B), i*Ts, (i-1)*Ts; solver, reltol, abstol) 
         end
         Threads.@threads for i = 1:K
               @inbounds Cd[:,:,i]  = tvcsylv(A, B, C, i*Ts, (i-1)*Ts; adj, solver, reltol, abstol) 
         end
      end
      X = pssylvd(Ad, Bd, -Cd; isgn = -1, rev = adj)
   end
   return PeriodicTimeSeriesMatrix([X[:,:,i] for i in 1:size(X,3)], period; nperiod)
end
"""
      tvcsylv_eval(t, W, A, B, C; adj = false, solver, reltol, abstol) -> Xval

Compute the time value `Xval := X(t)` of the solution of the periodic Sylvester differential equation

       .
       X(t) = A(t)X(t) + X(t)B(t) + C(t) ,  X(t0) = W(t0), t > t0, if adj = false

or 

       .
      -X(t) = A(t)X(t) + X(t)B(t) + C(t) ,  X(t0) = W(t0), t < t0, if adj = true,  

using the periodic generator `W` determined with the function [`pgcsylv`](@ref) for the same periodic matrices `A`, `B`  and `C`
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
function PeriodicMatrixEquations.tvcsylv_eval(t::Real,X::PeriodicTimeSeriesMatrix, A::PM1, B::PM2, C::PM3; adj = false, solver = "non-stiff", reltol = 1e-4, abstol = 1e-7) where
   {PM1 <: FourierFunctionMatrix, PM2 <: FourierFunctionMatrix, PM3 <: FourierFunctionMatrix} 
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
   return PeriodicMatrixEquations.tvcsylv(A, B, C, tf, t0, X.values[ind]; adj, solver, reltol, abstol) 
end
function PeriodicMatrixEquations.tvcsylv_eval(t::Real,X::PeriodicTimeSeriesMatrix,A::PM1,B::PM2; solver = "non-stiff", reltol = 1e-4, abstol = 1e-7) where
   {PM1 <: FourierFunctionMatrix, PM2 <: FourierFunctionMatrix} 
   tsub = X.period/X.nperiod
   ns = length(X.values)
   Δ = tsub/ns
   tf = mod(t,tsub)
   tf == 0 && (return X.values[1])
      ind = round(Int,tf/Δ)
      ind == 0 && (ind = 1) 
      t0 = (ind-1)*Δ
   #@show tf, t0
   return PeriodicMatrixEquations.tvcsylv(A, B, FourierFunctionMatrix(zeros(eltype(A),size(A)...),A.period), tf, t0, X.values[ind]; adj = false, solver, reltol, abstol) 
end

function PeriodicMatrixEquations.tvcsylv(A::PM1, B::PM2, C::PM3, tf, t0, W0::Union{AbstractMatrix,Missing} = missing; adj = false, solver = "auto", reltol = 1e-4, abstol = 1e-7) where
   {PM1 <: FourierFunctionMatrix, PM2 <: FourierFunctionMatrix, PM3 <: FourierFunctionMatrix} 
   """
      tvcsylv(A, B, C, tf, t0; adj, solver, reltol, abstol) -> W::Matrix

   Compute the solution at tf > t0 of the differential matrix Sylvester equation 
            .
            W(t) = A(t)*W(t)+W(t)*B(t)+C(t), W(t0) = 0, tf > t0, if adj = false

   or 
            .
            W(t) = -A(t)*W(t)-W(t)*B(t)-C(t), W(t0) = 0, tf < t0, if adj = true. 

   The ODE solver to be employed to convert the continuous-time problem into a discrete-time 
   problem can be specified using the keyword argument `solver`, 
   together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`) and  
   absolute accuracy `abstol` (default: `abstol = 1.e-7`).
   """
   m = size(A,1)
   m == size(A,2) || error("the periodic matrix A must be square")
   n = size(B,1)
   n == size(B,2) || error("the periodic matrix B must be square")
   m == size(C,1) || error("the periodic matrix C must have same row dimension as A")
   n == size(C,2) || error("the periodic matrix C must have same column dimension as B")
   T = promote_type(typeof(t0), typeof(tf))
   # using OrdinaryDiffEq
   if ismissing(W0) 
      u0 = zeros(T,m,n) 
   else
      (m,n) == size(W0) || throw(ArgumentError("W0 must be an $m × $n matrix"))
      u0 = copy(W0)
   end
   tspan = (T(t0),T(tf))
   fcsylv!(du,u,p,t) = du .= adj ? -(tpmeval(A,t)*u + u*tpmeval(B,t)+tpmeval(C,t)) : tpmeval(A,t)*u + u*tpmeval(B,t)+tpmeval(C,t)
   prob = ODEProblem(fcsylv!, u0, tspan)
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
   else 
      if reltol > 1.e-4  
         # low accuracy automatic selection
         sol = solve(prob, AutoTsit5(Rosenbrock23()) ; reltol, abstol, save_everystep = false)
      else
         # high accuracy automatic selection
         sol = solve(prob, AutoVern9(Rodas5(),nonstifftol = 11/10); reltol, abstol, save_everystep = false)
      end
   end
   return sol.u[end]    
end
function tvcsylv_sol(A::PM1, B::PM2, C::PM3, tf, t0, W0::Union{AbstractMatrix,Missing} = missing; adj = false, solver = "auto", reltol = 1e-4, abstol = 1e-7) where
   {PM1 <: FourierFunctionMatrix, PM2 <: FourierFunctionMatrix, PM3 <: FourierFunctionMatrix} 
   """
      tvcsylv(A, B, C, tf, t0; adj, solver, reltol, abstol) -> W::Matrix

   Compute the solution on the interval `[t0,tf]` of the differential matrix Sylvester equation 
            .
            W(t) = A(t)*W(t)+W(t)*B(t)+C(t), W(t0) = W0, tf > t0, if adj = false

   or on the interval `[tf,t0]` of
            .
            W(t) = -A(t)*W(t)-W(t)*B(t)-C(t), W(t0) = W0, tf < t0, if adj = true. 

   The computed solution `W(t)`, results using the solvers from the 
   [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) package with the option `dense = true`.             

   The ODE solver to be employed to convert the continuous-time problem into a discrete-time 
   problem can be specified using the keyword argument `solver`, 
   together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`) and  
   absolute accuracy `abstol` (default: `abstol = 1.e-7`).
   """
   m = size(A,1)
   m == size(A,2) || error("the periodic matrix A must be square")
   n = size(B,1)
   n == size(B,2) || error("the periodic matrix B must be square")
   m == size(C,1) || error("the periodic matrix C must have same row dimension as A")
   n == size(C,2) || error("the periodic matrix C must have same column dimension as B")
   T = promote_type(typeof(t0), typeof(tf))
   # using OrdinaryDiffEq
   if ismissing(W0) 
      u0 = zeros(T,m,n) 
   else
      (m,n) == size(W0) || throw(ArgumentError("W0 must be an $m × $n matrix"))
      u0 = copy(W0)
   end
   tspan = (T(t0),T(tf))
   fcsylv!(du,u,p,t) = du .= adj ? -(tpmeval(A,t)*u + u*tpmeval(B,t)+tpmeval(C,t)) : tpmeval(A,t)*u + u*tpmeval(B,t)+tpmeval(C,t)
   prob = ODEProblem(fcsylv!, u0, tspan)
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

function pcsylv_intpol(A::PM1, B::PM2, C::PM3, W0::PeriodicTimeSeriesMatrix; N::Int = length(W0), adj = false, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7) where
   {PM1 <: FourierFunctionMatrix, PM2 <: FourierFunctionMatrix, PM3 <: FourierFunctionMatrix} 
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
          Y[k]  = PeriodicMatrixEquations.tvcsylv_sol(A, B, C, (k-1)*Ts, k*Ts, W0.values[iw]; solver = "auto", adj = true, reltol = 1.e-10, abstol = 1.e-10) 
      end
   else
      for k = 1:N
          Y[k]  = PeriodicMatrixEquations.tvcsylv_sol(A, B, C, k*Ts, (k-1)*Ts, W0.values[(k-1)*Ni+1]; adj, solver, reltol, abstol) 
      end
   end

   return PeriodicFunctionMatrix(t-> 
   begin
      tf = mod(t,tsub)
      ind = round(Int,tf/Ts-0.5)+1
      Y[ind](tf)   
   end, W0.period; W0.nperiod)  
end
function PeriodicMatrixEquations.tvcsylv(A::PM1, B::PM2, C::PM3, ts::AbstractVector, W0::AbstractMatrix; adj = false, solver = "auto", reltol = 1e-4, abstol = 1e-7) where
   {PM1 <: FourierFunctionMatrix, PM2 <:FourierFunctionMatrix, PM3 <:FourierFunctionMatrix} 
   """
      tvcsylv(A, B, C, ts, W0; adj, solver, reltol, abstol) -> W::Vector{Matrix}

   Compute the solution at the time values `ts` of the differential matrix Sylvester equation 
            .
            W(t) = A(t)*W(t)+W(t)*B(t)+C(t), W(ts[1]) = W0, if adj = false

   or 
            .
            W(t) = -A(t)*W(t)-W(t)*B(t)-C(t), W(ts[end]) = W0, tf < t0, if adj = true. 

   The ODE solver to be employed can be specified using the keyword argument `solver`, 
   together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`),  
   absolute accuracy `abstol` (default: `abstol = 1.e-7`) and stepsize `dt` (default: `dt = abs(tf-t0)/100`, only used if `solver = "symplectic"`). 
   
   """
   m = size(A,1)
   m == size(A,2) || error("the periodic matrix A must be square")
   n = size(B,1)
   n == size(B,2) || error("the periodic matrix B must be square")
   m == size(C,1) || error("the periodic matrix C must have same row dimension as A")
   n == size(C,2) || error("the periodic matrix C must have same column dimension as B")
   all(diff(ts) .>= 0) || throw(ArgumentError("the elements of ts must be non-decreasing"))
   T = eltype(ts)
   # using OrdinaryDiffEq
   (m,n) == size(W0) || throw(ArgumentError("W0 must be an $m × $n matrix"))
   u0 = copy(W0)
   if adj
      period = promote_period(A,C)
      n = round(ts[end]/period)+1
      tspan = (n*period, ts[1]) 
      tval = reverse(ts)
   else
      tspan = (0, ts[end]) 
      tval = ts
   end
   fcsylv!(du,u,p,t) = du .= adj ? -(tpmeval(A,t)*u + u*tpmeval(B,t)+tpmeval(C,t)) : tpmeval(A,t)*u + u*tpmeval(B,t)+tpmeval(C,t)
   prob = ODEProblem(fcsylv!, u0, tspan)
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
   else 
      if reltol > 1.e-4  
         # low accuracy automatic selection
         sol = solve(prob, AutoTsit5(Rosenbrock23()) ; reltol, abstol, saveat = tval)
      else
         # high accuracy automatic selection
         sol = solve(prob, AutoVern9(Rodas5(),nonstifftol = 11/10); reltol, abstol, saveat = tval)
      end
   end
   W = sol.u
   return adj ? reverse(W) : W    
end
function PeriodicMatrixEquations.tvcsylv(A::PM1, B::PM2, C::PM3, W0::AbstractMatrix; adj = false, solver = "auto", reltol = 1e-4, abstol = 1e-7) where
   {PM1 <: FourierFunctionMatrix, PM2 <: FourierFunctionMatrix, PM3 <: FourierFunctionMatrix} 
   """
      tvcsylv(A, B, C, W0; adj, solver, reltol, abstol) -> W::Matrix

   Compute the solution at the time values between [0, period] of the differential matrix Sylvester equation
            .
            W(t) = A(t)*W(t)+W(t)*B(t)+C(t), W(0) = W0, if adj = false

   or 
            .
            W(t) = -A(t)*W(t)-W(t)*B(t)-C(t), W(period) = W0, tf < t0, if adj = true. 

   The ODE solver to be employed can be specified using the keyword argument `solver`, 
   together with the required relative accuracy `reltol` (default: `reltol = 1.e-4`),  
   absolute accuracy `abstol` (default: `abstol = 1.e-7`) and stepsize `dt` (default: `dt = abs(tf-t0)/100`, only used if `solver = "symplectic"`). 
   Depending on the desired relative accuracy `reltol`, lower order solvers are employed for `reltol >= 1.e-4`, 
   which are generally very efficient, but less accurate. If `reltol < 1.e-4`,
   higher order solvers are employed able to cope with high accuracy demands. 

   The following solvers from the [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) package can be selected:

   `solver = "non-stiff"` - use a solver for non-stiff problems (`Tsit5()` or `Vern9()`);

   `solver = "stiff"` - use a solver for stiff problems (`Rodas4()` or `KenCarp58()`);

   `solver = "auto"` - use the default solver, which automatically detects stiff problems (`AutoTsit5(Rosenbrock23())` or `AutoVern9(Rodas5())`). 
   """
   m = size(A,1)
   m == size(A,2) || error("the periodic matrix A must be square")
   n = size(B,1)
   n == size(B,2) || error("the periodic matrix B must be square")
   m == size(C,1) || error("the periodic matrix C must have same row dimension as A")
   n == size(C,2) || error("the periodic matrix C must have same column dimension as B")
   period, nperiod = promote_period2(A,B,C)
   T = eltype(period)
   # using OrdinaryDiffEq
   u0 = copy(W0)
   tspan = adj ? (period, zero(T)) : (zero(T), period) 
   fcsylv!(du,u,p,t) = du .= adj ? -(tpmeval(A,t)*u + u*tpmeval(B,t)+tpmeval(C,t)) : tpmeval(A,t)*u + u*tpmeval(B,t)+tpmeval(C,t)
   prob = ODEProblem(fcsylv!, u0, tspan)
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
   else 
      if reltol > 1.e-4  
         # low accuracy automatic selection
         sol = solve(prob, AutoTsit5(Rosenbrock23()) ; reltol, abstol, save_everystep = true)
      else
         # high accuracy automatic selection
         sol = solve(prob, AutoVern9(Rodas5(),nonstifftol = 11/10); reltol, abstol, save_everystep = true)
      end
   end
   return PeriodicFunctionMatrix(t-> sol(t),period)     
end
function PeriodicMatrixEquations.tvcsylv(A::PM1, B::PM2, C::PM3, W::PeriodicTimeSeriesMatrix; adj = false, solver = "auto", reltol = 1e-4, abstol = 1e-7) where
   {PM1 <: FourierFunctionMatrix, PM2 <: FourierFunctionMatrix, PM3 <: FourierFunctionMatrix} 
   """
      tvclyap(A, B, C, G; adj, solver, reltol, abstol) -> W::Matrix

   Compute the solution at the time values between [0, period] of the differential matrix Sylvester equation 
            .
            W(t) = A(t)*W(t)+W(t)*B(t)+C(t), W(0) = G(0), if adj = false

   or 
            .
            W(t) = -A(t)*W(t)-W(t)*B(t)-C(t), W(period) = G(0), if adj = true,
            
   where `G` is a periodic generator computed by `pgcsylv` and `tsub = period/nperiod`.
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
   m = size(A,1)
   m == size(A,2) || error("the periodic matrix A must be square")
   n = size(B,1)
   n == size(B,2) || error("the periodic matrix B must be square")
   m == size(C,1) || error("the periodic matrix C must have same row dimension as A")
   n == size(C,2) || error("the periodic matrix C must have same column dimension as B")
   (m,n) == size(W) || error("the periodic time series matrix W must have same dimensions $m × $n")
   period = W.period
   nperiod = W.nperiod
   (period, nperiod) == promote_period2(A,B,C) || error("inconsistent period and nperiod between (A, C) and W")
   T = eltype(period)
   # using OrdinaryDiffEq
   u0 = copy(W(0))
   tsub = period/nperiod
   tspan = adj ? (tsub, zero(T)) : (zero(T), tsub) 
   fcsylv!(du,u,p,t) = du .= adj ? -(tpmeval(A,t)*u + u*tpmeval(B,t)+tpmeval(C,t)) : tpmeval(A,t)*u + u*tpmeval(B,t)+tpmeval(C,t)
   prob = ODEProblem(fcsylv!, u0, tspan)
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
   else 
      if reltol > 1.e-4  
         # low accuracy automatic selection
         sol = solve(prob, AutoTsit5(Rosenbrock23()) ; reltol, abstol, dense = true)
      else
         # high accuracy automatic selection
         sol = solve(prob, AutoVern9(Rodas5(),nonstifftol = 11/10); reltol, abstol, dense = true)
      end
   end
   return PeriodicFunctionMatrix(t-> sol(mod(t,tsub)),period; nperiod)     
end
    