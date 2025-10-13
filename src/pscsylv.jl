"""
    pcsylv(A, C; K = 10, N = 1, adj = false, solver, reltol, abstol, intpol = true) -> X

Solve the periodic Sylvester differential equation

    .
    X(t) = A(t)X(t) + X(t)B(t) + C(t) , if adj = false,

or 

     .
    -X(t) = A(t)X(t) + X(t)B(t) + C(t) , if adj = true.               

The periodic matrices `A`, `B` and `C` must have the same type, compatible dimensions and commensurate periods. 
Additionally, `A` and `B` must be square. 
The resulting periodic solution `X` has the type `PeriodicFunctionMatrix` and 
`X(t)` can be used to evaluate the value of `X` at time `t`. 
`X` has the period set to the least common commensurate period of `A`, `B` and `C` and the number of subperiods
is adjusted accordingly. 

The multiple-shooting method of [1] is employed to convert the (continuous-time) periodic differential Sylvester equation 
into a discrete-time periodic Sylvester equation satisfied by a multiple point periodic generator of the solution. 
The keyword argument `K` specifies the number of grid points to be used
for the discretization of the continuous-time problem (default: `K = 10`). 
If  `A`, `B` and `C` are of types `PeriodicTimeSeriesMatrix` or `PeriodicSwitchingMatrix`, then `K` specifies the number of grid points 
used between two consecutive switching time values (default: `K = 1`).  
The multiple point periodic generator is computed  by solving the appropriate discrete-time periodic Sylvester 
equation using the periodic Schur method of [2]. 
The obtained periodic generator is finally converted into a periodic function matrix which determines for a given `t` 
the function value `X(t)` by interpolating the solution of the appropriate differential equations if `intpol = true` (default)  
or by integrating the appropriate ODE from the nearest grid point value if `intpol = false`.
For the interpolation-based evaluation the integer keyword argument `N` can be used to split the integration domain (i.e., one period) 
into `N` subdomains to perform the interpolations separately in each domain.
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
function pcsylv(A::PeriodicFunctionMatrix, B::PeriodicFunctionMatrix, C::PeriodicFunctionMatrix; K::Int = 10, N::Int = 1, adj = false, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7, intpol = true)
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
pcsylv(A::PeriodicFunctionMatrix, B::PeriodicFunctionMatrix, C::AbstractMatrix; kwargs...) = pcsylv(A, B, PeriodicFunctionMatrix(C, A.period; nperiod = A.nperiod); kwargs...)
function pcsylv(A::PeriodicSymbolicMatrix, B::PeriodicSymbolicMatrix, C::PeriodicSymbolicMatrix; K::Int = 10, N::Int = 1, adj = false, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7, intpol = true)
   At = convert(PeriodicFunctionMatrix,A)
   Bt = convert(PeriodicFunctionMatrix,B)
   Ct = convert(PeriodicFunctionMatrix,C)
   W0 = PeriodicMatrixEquations.pgcsylv(At, Bt, Ct, K;  adj, solver, reltol, abstol)
   if PeriodicMatrices.isconstant(W0) 
      (K > 1 || (PeriodicMatrices.isconstant(A) && PeriodicMatrices.isconstant(B) && PeriodicMatrices.isconstant(C) )) && (return PeriodicFunctionMatrix(W0(0),W0.period,nperiod=W0.nperiod))
   end
   if intpol
      return PeriodicMatrixEquations.pcsylv_intpol(At, Bt, Ct, W0; N, adj, solver, reltol, abstol)
   else
      PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvcsylv_eval(t, W0, At, Bt, Ct; solver, adj, reltol, abstol), W0.period; nperiod = W0.nperiod)
   end
end
function pcsylv(A::HarmonicArray, B::HarmonicArray, C::HarmonicArray; K::Int = 10, N::Int = 1, adj = false, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7, intpol = true)
   W0 = PeriodicMatrixEquations.pgcsylv(A, B, C, K;  adj, solver, reltol, abstol)
   if PeriodicMatrices.isconstant(W0) 
      (K > 1 || (PeriodicMatrices.isconstant(A) && PeriodicMatrices.isconstant(B) && PeriodicMatrices.isconstant(C) )) && (return PeriodicFunctionMatrix(W0(0),W0.period,nperiod=W0.nperiod))
   end
   if intpol
      return PeriodicMatrixEquations.pcsylv_intpol(A, B, C, W0; N, adj, solver, reltol, abstol)
   else
      PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvcsylv_eval(t, W0, A, B, C; solver, adj, reltol, abstol), W0.period; nperiod = W0.nperiod)
   end
end
function pcsylv(A::PeriodicTimeSeriesMatrix, B::PeriodicTimeSeriesMatrix, C::PeriodicTimeSeriesMatrix; K::Int = 10, N::Int = 1, adj = false, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7, intpol = true)
   if intpol
      W0 = PeriodicMatrixEquations.pgcsylv(A, B, C, K;  adj, solver, reltol, abstol)
      if PeriodicMatrices.isconstant(W0) 
         (K > 1 || (PeriodicMatrices.isconstant(A) && PeriodicMatrices.isconstant(B) && PeriodicMatrices.isconstant(C) )) && (return PeriodicFunctionMatrix(W0(0),W0.period,nperiod=W0.nperiod))
      end
      return PeriodicMatrixEquations.pcsylv_intpol(A, B, C, W0; N, adj, solver, reltol, abstol)
   else
      W0 = pgcsylv(convert(HarmonicArray,A), convert(HarmonicArray,B), convert(HarmonicArray,C), K;  adj, solver, reltol, abstol)
      if PeriodicMatrices.isconstant(W0) 
         (K > 1 || (PeriodicMatrices.isconstant(A) && PeriodicMatrices.isconstant(B) && PeriodicMatrices.isconstant(C) )) && (return PeriodicFunctionMatrix(W0(0),W0.period,nperiod=W0.nperiod))
      end
      PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvcsylv_eval(t, W0, A, B, C; solver, adj, reltol, abstol), W0.period; nperiod = W0.nperiod)
   end
end
function pcsylv(A::PeriodicSwitchingMatrix, B::PeriodicSwitchingMatrix, C::PeriodicSwitchingMatrix; K::Int = 1, adj = false, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7)
   At = convert(PeriodicFunctionMatrix,A)
   Bt = convert(PeriodicFunctionMatrix,B)
   Ct = convert(PeriodicFunctionMatrix,C)
   W0 = pgcsylv(At, Bt, Ct, K;  adj, solver, reltol, abstol)
   if PeriodicMatrices.isconstant(W0) 
      (K > 1 || (PeriodicMatrices.isconstant(A) && PeriodicMatrices.isconstant(B) && PeriodicMatrices.isconstant(C) )) && (return PeriodicFunctionMatrix(W0(0),W0.period,nperiod=W0.nperiod))
   end
   PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvcsylv_eval(t, W0, At, Bt, Ct; solver, adj, reltol, abstol), W0.period; nperiod = W0.nperiod)
end


for PM in (:PeriodicFunctionMatrix, :PeriodicSymbolicMatrix, :HarmonicArray, :PeriodicTimeSeriesMatrix)
   @eval begin
      function prcsylv(A::$PM, B::$PM, C::$PM; K::Int = 10, N::Int = 1, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7, intpol = true) 
         pcsylv(A, B, C; K, N, adj = true, solver, reltol, abstol, intpol)
      end
      function prcsylv(A::$PM, B::$PM, C::AbstractMatrix; kwargs...)
         prcsylv(A, B, $PM(C, A.period; nperiod = A.nperiod); kwargs...)
      end
      function pfcsylv(A::$PM, B::$PM, C::$PM; K::Int = 10, N::Int = 1, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7, intpol = true) 
         pcsylv(A, B, C; K, N, adj = false, solver, reltol, abstol, intpol)
      end
      function pfcsylv(A::$PM, B::$PM, C::AbstractMatrix; kwargs...)
         pfcsylv(A, B, $PM(C, A.period; nperiod = A.nperiod); kwargs...)
      end
   end
end
function prcsylv(A::PM, B::PM, C::PM; K::Int = 1, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7) where {PM <: PeriodicSwitchingMatrix}
   pcsylv(A, B, C; K, adj = true, solver, reltol, abstol)
end
function prcsylv(A::PM, B::PM, C::AbstractMatrix; kwargs...) where {PM <: PeriodicSwitchingMatrix}
   prcsylv(A, B, PeriodicSwitchingMatrix(C, A.period; nperiod = A.nperiod); kwargs...)
end
function pfcsylv(A::PM, B::PM, C::PM; K::Int = 1, solver = "non-stiff", reltol = 1.e-4, abstol = 1.e-7) where {PM <: PeriodicSwitchingMatrix}
   pcsylv(A, B, C; K, adj = false, solver, reltol, abstol)
end
function pfcsylv(A::PM, B::PM, C::AbstractMatrix; kwargs...) where {PM <: PeriodicSwitchingMatrix}
   pfcsylv(A, B, PeriodicSwitchingMatrix(C, A.period; nperiod = A.nperiod); kwargs...)
end

"""
    pfcsylv(A, B, C; K = 10, N = 1, solver, reltol, abstol, intpol) -> X

Solve the periodic forward-time Sylvester differential equation

    .
    X(t) = A(t)X(t) + X(t)B(t) + C(t) .               

The periodic matrices `A`, `B` and `C` must have the same type, compatible dimensions and commensurate periods, 
Additionally, `A` and `B` must be square. The resulting periodic solution `X` has the period 
set to the least common commensurate period of `A`, `B` and `C` and the number of subperiods
is adjusted accordingly. 

This function is merely an interface to [`pcsylv`](@ref) (see this function for the description of keyword parameters). 
"""
pfcsylv(A::PeriodicFunctionMatrix, B::PeriodicFunctionMatrix, C::PeriodicFunctionMatrix)
"""
    prcsylv(A, B, C; K = 10, N = 1, solver, reltol, abstol, intpol) -> X

Solve the periodic reverse-time Sylvester differential equation

     .
    -X(t) = A(t)'X(t) + X(t)A(t) + C(t).               

The periodic matrices `A`, `B` and `C` must have the same type, compatible dimensions and commensurate periods. 
Additionally, `A` and `B` must be square. The resulting periodic solution `X` has the period 
set to the least common commensurate period of `A`, `B` and `C` and the number of subperiods
is adjusted accordingly. 

This function is merely an interface to [`pcsylv`](@ref) (see this function for the description of keyword parameters). 
"""
prcsylv(A::PeriodicFunctionMatrix, B::PeriodicFunctionMatrix, C::PeriodicFunctionMatrix)

"""
    pgsylv(A, B, C[, K = 1]; adj = false, solver, reltol, abstol) -> X

Compute periodic generators for the periodic Sylvester differential equation

    .
    X(t) = A(t)X(t) + X(t)B(t) + C(t) , if adj = false,

or 

     .
    -X(t) = A(t)X(t) + X(t)B(t) + C(t) , if adj = true.
    
The periodic matrices `A`, `B` and `C` must have the same type, compatible dimensions and commensurate periods.
`K` specifies the number of grid points to be used
for the discretization of the continuous-time problem (default: `K = 1`). 
If  `A`, `B` and `C` are of types `PeriodicTimeSeriesMatrix` or `PeriodicSwitchingMatrix`, then `K` specifies the number of grid points 
used between two consecutive switching time values (default: `K = 1`).  

If `A`, `B` and `C` have the types `PeriodicFunctionMatrix`, `HarmonicArray`, `FourierFunctionMatrix` or `PeriodicTimeSeriesMatrix`, 
then the resulting `X` is a collection of periodic generator matrices determined 
as a _periodic time-series matrix_ with `N` components, where `N = 1` if `A`, `B` and `C` are constant matrices
and `N = K` otherwise. 
If `A`, `B` and `C` have the type `PeriodicSwitchingMatrix`, then `X` is a collection of periodic generator matrices 
determined as a _periodic switching matrix_,
whose switching times are the unique switching times contained in the union of the switching times of `A`, `B` and `C`. 
If `K > 1`, a refined grid of `K` equidistant values is used for each two consecutive 
switching times in the union.      

The period of `X` is set to the least common commensurate period of `A`, `B` and `C` and the number of subperiods
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
function pgcsylv(A::PM1, B::PM2, C::PM3, K::Int = 1; adj = false, solver = "non-stiff", reltol = 1e-4, abstol = 1e-7) where
   {PM1 <: Union{PeriodicFunctionMatrix,PeriodicSymbolicMatrix,HarmonicArray}, 
    PM2 <: Union{PeriodicFunctionMatrix,PeriodicSymbolicMatrix,HarmonicArray},
    PM3 <: Union{PeriodicFunctionMatrix,PeriodicSymbolicMatrix,HarmonicArray}} 
   K > 0 || throw(ArgumentError("number of grid ponts K must be greater than 0, got K = $K"))    
   period = promote_period(A, B, C)
   na = Int(round(period/A.period))
   nb = Int(round(period/B.period))
   nc = Int(round(period/C.period))
   nperiod = gcd(na*A.nperiod, nb*B.nperiod, nc*C.nperiod)
   m = size(A,1)
   n = size(B,1)
   Ts = period/K/nperiod
   
   T = promote_type(eltype(A),eltype(B),eltype(C),Float64)
   T == Num && (T = Float64)
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
function pgcsylv(A::PM1, B::PM2, C::PM3, K::Int = 1; adj = false, solver = "non-stiff", reltol = 1e-4, abstol = 1e-7) where
                {PM1 <: PeriodicTimeSeriesMatrix, PM2 <: PeriodicTimeSeriesMatrix, PM3 <: PeriodicTimeSeriesMatrix} 
   K > 0 || throw(ArgumentError("number of grid ponts K must be greater than 0, got K = $K"))    
   m = size(A,1)
   m == size(A,2) || error("the periodic matrix A must be square")
   n = size(B,1)
   n == size(B,2) || error("the periodic matrix B must be square")
   m == size(C,1) || error("the periodic matrix C must have same row dimension as A")
   n == size(C,2) || error("the periodic matrix C must have same column dimension as B")
   period = promote_period(A, B, C)
   na = round(Int,period/A.period)
   nb = round(Int,period/B.period)
   nc = round(Int,period/C.period)
   nperiod = gcd(na*A.nperiod, nb*B.nperiod, nc*C.nperiod)
   tsub = period/nperiod
   if A.period == B.period == C.period
      nperiod = gcd(A.nperiod,B.nperiod,C.nperiod)
      ns = div(lcm(A.nperiod*length(A),B.nperiod*length(B),C.nperiod*length(C)),nperiod)
      Δ = A.period/nperiod/ns
      δ = Δ/2
   else       
      Tsub = A.period/A.nperiod
      Tsub ≈ B.period/B.nperiod ≈ C.period/C.nperiod || error("periods or subperiods must be equal for addition")
      nperiod = lcm(A.nperiod,B.nperiod,C.nperiod)
      period = Tsub*nperiod
      ns = lcm(length(A),length(B),length(C))
      Δ = Tsub/ns
      δ = Δ/2
   end 
   Kc = ns
   Tsd = Δ/K
   Ts = Δ
   δ = Tsd/2
   T = promote_type(eltype(A),eltype(C),Float64)
   if PeriodicMatrices.isconstant(A) && PeriodicMatrices.isconstant(B) && PeriodicMatrices.isconstant(C)
      X = sylvc(tpmeval(A,0), tpmeval(B,0), -tpmeval(C,0)) 
      return PeriodicTimeSeriesMatrix([X], period; nperiod)
   else
      Ka = PeriodicMatrices.isconstant(A) ? 1 : Kc
      Kb = PeriodicMatrices.isconstant(B) ? 1 : Kc
      Kc1 = Kc*K
      Ad = Array{T,3}(undef, n, n, Ka*K) 
      Bd = Array{T,3}(undef, n, n, Kb*K) 
      Cd = Array{T,3}(undef, n, n, Kc1) 
      if adj
         Threads.@threads for i = 1:Ka
            k = (i-1)*K+1
            @inbounds Ad[:,:,k] = transpose(exp(tpmeval(A,(i-1)*Ts+δ)*Tsd)) 
            K == 1 || [Ad[:,:,j] = Ad[:,:,k] for j in k+1:k+K-1]  
         end
         Threads.@threads for i = 1:Kb
             k = (i-1)*K+1
             @inbounds Bd[:,:,k] = exp(tpmeval(B,(i-1)*Ts+δ)*Tsd) 
             K == 1 || [Bd[:,:,j] = Bd[:,:,k] for j in k+1:k+K-1]  
         end
         k = Kc1
         Threads.@threads for i = Kc:-1:1
            tf = (i-1)*Ts
            k = (i-1)*K+1
            @inbounds Cd[:,:,k]  = tvcsylv(A, B, C, tf, tf+Tsd; adj, solver, reltol, abstol) 
            #K == 1 || [Cd[:,:,j] = Cd[:,:,k] for j in k-K+1:k-1]  
            K == 1 || [Cd[:,:,j] = Cd[:,:,k] for j in k+1:k+K-1]  
         end
      else
         Threads.@threads for i = 1:Ka
             k = (i-1)*K+1
             @inbounds Ad[:,:,k] = exp(tpmeval(A,(i-1)*Ts+δ)*Tsd) 
             K == 1 || [Ad[:,:,j] = Ad[:,:,k] for j in k+1:k+K-1]  
         end
         Threads.@threads for i = 1:Kb
             k = (i-1)*K+1
             @inbounds Bd[:,:,k] = transpose(exp(tpmeval(B,(i-1)*Ts+δ)*Tsd)) 
             K == 1 || [Bd[:,:,j] = Bd[:,:,k] for j in k+1:k+K-1]  
         end
         Threads.@threads for i = 1:Kc
            t0 = (i-1)*Ts
            k = (i-1)*K+1
            @inbounds Cd[:,:,k]  = tvcsylv(A, B, C, t0+Tsd, t0; adj, solver, reltol, abstol) 
            K == 1 || [Cd[:,:,j] = Cd[:,:,k] for j in k+1:k+K-1]  
         end
      end
      X = pssylvd(Ad, Bd, -Cd; isgn = -1, rev = adj)
   end
   return PeriodicTimeSeriesMatrix([X[:,:,i] for i in 1:size(X,3)], period; nperiod)
end

function pgcsylv(A::PM1, B::PM2, C::PM3, K::Int = 1; adj = false, solver = "non-stiff", reltol = 1e-4, abstol = 1e-7) where
      {PM1 <: PeriodicSwitchingMatrix, PM2 <: PeriodicSwitchingMatrix, PM3 <: PeriodicSwitchingMatrix} 
   K > 0 || throw(ArgumentError("number of grid ponts K must be greater than 0, got K = $K"))    
   m = size(A,1)
   m == size(A,2) || error("the periodic matrix A must be square")
   n = size(B,1)
   n == size(B,2) || error("the periodic matrix B must be square")
   m == size(C,1) || error("the periodic matrix C must have same row dimension as A")
   n == size(C,2) || error("the periodic matrix C must have same column dimension as B")
   period = promote_period(A, B, C)
   na = round(Int,period/A.period)
   nb = round(Int,period/B.period)
   nc = round(Int,period/C.period)
   nperiod = gcd(na*A.nperiod, nb*B.nperiod, nc*C.nperiod)
   tsub = period/nperiod

   ts = unique(sort([A.ts;B.ts;C.ts]))
   Kc = length(ts)
  
   T = promote_type(eltype(A),eltype(B),eltype(C),Float64)
   if PeriodicMatrices.isconstant(A) && PeriodicMatrices.isconstant(B) && PeriodicMatrices.isconstant(C)
      X = sylvc(tpmeval(A,0), tpmeval(B,0), -tpmeval(C,0)) 
      return PeriodicSwitchingMatrix([X], ts, period; nperiod)
   else
      Ka = PeriodicMatrices.isconstant(A) ? 1 : Kc
      Kb = PeriodicMatrices.isconstant(B) ? 1 : Kc
      Kc1 = Kc*K
      Ad = Array{T,3}(undef, n, n, Ka*K) 
      Bd = Array{T,3}(undef, n, n, Kb*K) 
      Cd = Array{T,3}(undef, n, n, Kc1) 
      if adj
         Threads.@threads for i = 1:Ka
            tf = i == Ka ? tsub : ts[i+1]
            k = (i-1)*K+1
            @inbounds Ad[:,:,k] = transpose(exp(tpmeval(A,ts[i])*(tf-ts[i])/K))
            K == 1 || [Ad[:,:,j] = Ad[:,:,k] for j in k+1:k+K-1]  
         end
         Threads.@threads for i = 1:Kb
             tf = i == Kb ? tsub : ts[i+1]
             k = (i-1)*K+1
             @inbounds Bd[:,:,k] = exp(tpmeval(B,ts[i])*(tf-ts[i])/K) 
             K == 1 || [Bd[:,:,j] = Bd[:,:,k] for j in k+1:k+K-1]  
         end
         Threads.@threads for i = Kc:-1:1
            t0 = i == Kc ? tsub : ts[i+1]
            k = (i-1)*K+1
            @inbounds Cd[:,:,k]  = tvcsylv(A, B, C, ts[i], ts[i]+(t0-ts[i])/K; adj, solver, reltol, abstol) 
            K == 1 || [Cd[:,:,j] = Cd[:,:,k] for j in k+1:k+K-1]  
         end
      else
         Threads.@threads for i = 1:Ka
             tf = i == Ka ? tsub : ts[i+1]
             k = (i-1)*K+1
             @inbounds Ad[:,:,k] = exp(tpmeval(A,ts[i])*(tf-ts[i])/K) 
             K == 1 || [Ad[:,:,j] = Ad[:,:,k] for j in k+1:k+K-1]  
         end
         Threads.@threads for i = 1:Kb
             tf = i == Kb ? tsub : ts[i+1]
             k = (i-1)*K+1
             @inbounds Bd[:,:,k] = transpose(exp(tpmeval(B,ts[i])*(tf-ts[i])/K)) 
             K == 1 || [Bd[:,:,j] = Bd[:,:,k] for j in k+1:k+K-1]  
         end
         Threads.@threads for i = 1:Kc
            tf = i == Kc ? tsub : ts[i+1]
            k = (i-1)*K+1
            @inbounds Cd[:,:,k]  = tvcsylv(A, B, C, ts[i]+(tf-ts[i])/K, ts[i]; adj, solver, reltol, abstol) 
            K == 1 || [Cd[:,:,j] = Cd[:,:,k] for j in k+1:k+K-1]  
         end
      end
      X = pssylvd(Ad, Bd, -Cd; isgn = -1, rev = adj)
   end
   if K == 1
      return PeriodicSwitchingMatrix([X[:,:,i] for i in 1:size(X,3)], ts, period; nperiod)
   else
      tt = T[]
      for i = 1:Kc
          tf = i == Kc ? tsub : ts[i+1]
          Δ = (tf-ts[i])/K
          push!(tt,(ts[i] .+ collect(T,0:K-1)*Δ)...)
      end
      return PeriodicSwitchingMatrix([X[:,:,i] for i in 1:size(X,3)], tt, period; nperiod)
   end
end
"""
      tvcsylv_eval(t, W, A, C; adj = false, solver, reltol, abstol) -> Xval

Compute the time value `Xval := X(t)` of the solution of the periodic Sylvester differential equation

       .
       X(t) = A(t)X(t) + X(t)B(t) + C(t) ,  X(t0) = W(t0), t > t0, if adj = false

or 

       .
      -X(t) = A(t)X(t) + X(t)B(t) + C(t) ,  X(t0) = W(t0), t < t0, if adj = true,  

using the periodic generator `W` determined with the function [`pgcsylv`](@ref) for the same periodic matrices `A`, `B` and `C`
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
function tvcsylv_eval(t::Real,X::PeriodicTimeSeriesMatrix,A::PM1, B::PM2, C::PM3; adj = false, solver = "non-stiff", reltol = 1e-4, abstol = 1e-7) where
   {PM1 <: Union{PeriodicFunctionMatrix,PeriodicSymbolicMatrix,HarmonicArray,PeriodicTimeSeriesMatrix}, 
    PM2 <: Union{PeriodicFunctionMatrix,PeriodicSymbolicMatrix,HarmonicArray,PeriodicTimeSeriesMatrix},
    PM3 <: Union{PeriodicFunctionMatrix,PeriodicSymbolicMatrix,HarmonicArray,PeriodicTimeSeriesMatrix}} 
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
   return tvcsylv(A, B, C, tf, t0, X.values[ind]; adj, solver, reltol, abstol) 
end
function tvcsylv_eval(t::Real,X::PeriodicSwitchingMatrix, A::PM1, B::PM2, C::PM3; adj = false, solver = "non-stiff", reltol = 1e-4, abstol = 1e-7) where
                     {PM1 <: PeriodicSwitchingMatrix, PM2 <: PeriodicSwitchingMatrix, PM3 <: PeriodicSwitchingMatrix}   
   tsub = X.period/X.nperiod
   tf = mod(t,tsub)
   tf == 0 && (return X.values[1])
   if adj 
      ind = findfirst(X.ts .> tf*(1+10*eps()))
      isnothing(ind) ? (ind = 1; t0 = tsub) : t0 = X.ts[ind]; 
   else
      ind = findfirst(X.ts .> tf*(1+10*eps()))
      isnothing(ind) ? ind = length(X) : ind -= 1
      t0 = X.ts[ind]
   end
   return tvcsylv(A, B, C, tf, t0, X.values[ind]; adj, solver, reltol, abstol) 
end

function tvcsylv(A::PM1, B::PM2, C::PM3, tf, t0, W0::Union{AbstractMatrix,Missing} = missing; adj = false, solver = "auto", reltol = 1e-4, abstol = 1e-7) where
   {T1, T2, T3, PM1 <: AbstractPeriodicArray{:c,T1}, PM2 <: AbstractPeriodicArray{:c,T2}, PM3 <: AbstractPeriodicArray{:c,T3}} 
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

function tvcsylv(A::PM1, B::PM2, C::PM3, ts::AbstractVector, W0::AbstractMatrix; adj = false, solver = "auto", reltol = 1e-4, abstol = 1e-7) where
   {T1, T2, T3, PM1 <: AbstractPeriodicArray{:c,T1}, PM2 <: AbstractPeriodicArray{:c,T2}, PM3 <: AbstractPeriodicArray{:c,T3}} 
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
      period = promote_period(A,B,C)
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
function tvcsylv(A::PM1, B::PM2, C::PM3, W0::AbstractMatrix; adj = false, solver = "auto", reltol = 1e-4, abstol = 1e-7) where
   {T1, T2, T3, PM1 <: AbstractPeriodicArray{:c,T1}, PM2 <: AbstractPeriodicArray{:c,T2}, PM3 <: AbstractPeriodicArray{:c,T3}} 
   """
      tvcsylv(A, B, C, ts, W0; adj, solver, reltol, abstol) -> W::Matrix

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
   {T1, T2, T3, PM1 <: AbstractPeriodicArray{:c,T1}, PM2 <: AbstractPeriodicArray{:c,T2}, PM3 <: AbstractPeriodicArray{:c,T3}} 
   """
      tvclyap(A, B, C, G; adj, solver, reltol, abstol) -> W::Matrix

   Compute the solution at the time values between [0, period] of the differential matrix Sylvester equation 
            .
            W(t) = A(t)*W(t)+W(t)*B(t)+C(t), W(0) = G(0), if adj = false

   or 
            .
            W(t) = -A(t)*W(t)-W(t)*B(t)-C(t), W(tsub) = G(0), if adj = true,
            
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
   (period, nperiod) == promote_period2(A,B,C) || error("inconsistent period and nperiod between (A, B, C) and W")
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


function tvcsylv_sol(A::PM1, B::PM2, C::PM3, tf, t0, W0::Union{AbstractMatrix,Missing} = missing; adj = false, solver = "auto", reltol = 1e-4, abstol = 1e-7) where
   {T1, T2, T3, PM1 <: AbstractPeriodicArray{:c,T1}, PM2 <: AbstractPeriodicArray{:c,T2}, PM3 <: AbstractPeriodicArray{:c,T3}} 
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
   {T1, T2, T3, PM1 <: AbstractPeriodicArray{:c,T1}, PM2 <: AbstractPeriodicArray{:c,T2}, PM3 <: AbstractPeriodicArray{:c,T3}}  
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
      Threads.@threads for k = N:-1:1   
          #iw = mod(k*Ni-2,K)+3; 
          iw = mod(k*Ni-1,K)+2; 
          iw > K && (iw = 1)
          Y[k]  = PeriodicMatrixEquations.tvcsylv_sol(A, B, C, (k-1)*Ts, k*Ts, W0.values[iw]; solver = "auto", adj = true, reltol = 1.e-10, abstol = 1.e-10) 
      end
   else
      Threads.@threads for k = 1:N
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
