module Test_psclyap

using PeriodicMatrixEquations
using ApproxFun
using Symbolics
using Test
using LinearAlgebra
using Symbolics

println("Test_psclyap")

@testset "pclyap, pcplyap" begin

tt = Vector((1:500)*2*pi/500) 
ts = [0.4459591888577492, 1.2072325802972004, 1.9910835248218244, 2.1998199838900527, 2.4360161318589695, 
     2.9004720268745463, 2.934294124172935, 4.149208861412936, 4.260935465730602, 5.956614157549958]

# generate symbolic periodic matrices
@variables t
A1 = [0  1; -10*cos(t)-1 -24-19*sin(t)]
As = PeriodicSymbolicMatrix(A1,2*pi)
X1 =  [1+cos(t) 0; 0 1+sin(t)] 
Xdots = [-sin(t) 0; 0 cos(t)] 
# Xdots = Symbolics.derivative.(X1,t)
Xs = PeriodicSymbolicMatrix(X1, 2*pi)
Cds = PeriodicSymbolicMatrix(-(A1'*X1+X1*A1+Xdots),2*pi)
Cs = PeriodicSymbolicMatrix(Xdots - A1*X1-X1*A1', 2*pi)

# @time Ys = pclyap(As,Cs; K = 100, reltol = 1.e-10, abstol = 1.e-10);
# @test Xs ≈ Ys && norm(As*Ys+Ys*As'+Cs - pmderiv(Ys)) < 1.e-6

# @time Ys = pclyap(As,Cds; adj = true, K = 100, reltol = 1.e-10, abstol = 1.e-10);
# @test Xs ≈ Ys && norm(As'*Ys+Ys*As+Cds + pmderiv(Ys)) < 1.e-6

# @time Ys = pfclyap(As,Cs; K = 100, reltol = 1.e-10);
# @test Xs ≈ Ys && norm(As*Ys+Ys*As'+Cs - pmderiv(Ys)) < 1.e-6

# @time Ys = prclyap(As,Cds; K = 100, reltol = 1.e-10);
# @test Xs ≈ Ys && norm(As'*Ys+Ys*As+Cds + pmderiv(Ys)) < 1.e-6


@time Yt = pclyap(As,Cs; K = 512, reltol = 1.e-10, abstol = 1.e-10,intpol=false);
@time Yt1 = pclyap(As,Cs; K = 512, reltol = 1.e-10, abstol = 1.e-10,intpol=true);
Ys = convert(PeriodicSymbolicMatrix,Yt) 
Ys1 = convert(PeriodicSymbolicMatrix,Yt1) 
@test Ys ≈ Xs && Ys1 ≈ Xs

@time Yt = pclyap(As,Cds; adj = true, K = 512, reltol = 1.e-10, abstol = 1.e-10,intpol=true); 
Ys = convert(PeriodicSymbolicMatrix,Yt) 
@test Xs ≈ Ys 

@time Yt = pfclyap(As,Cs; K = 512, reltol = 1.e-10, abstol = 1.e-10,intpol=true); 
Ys = convert(PeriodicSymbolicMatrix,Yt) 
@test Ys ≈ Xs 

@time Yt = prclyap(As,Cds; K = 512, reltol = 1.e-10, abstol = 1.e-10,intpol=false); 
Ys = convert(PeriodicSymbolicMatrix,Yt) 
@test Xs ≈ Ys 


@time Yt = pclyap(As,Cs'*Cs; adj = true, K = 100, reltol = 1.e-10, abstol = 1.e-10,intpol=true);
@time Ut = pcplyap(As,Cs; adj = true, K = 100, reltol = 1.e-10, abstol = 1.e-10,intpol=true);
@test norm(Ut'.(ts).*Ut.(ts).-Yt.(ts),Inf) < 1.e-7 

# generate periodic function matrices
A(t) = [0  1; -10*cos(t)-1 -24-19*sin(t)]
X(t) = [1+cos(t) 0; 0 1+sin(t)]  # desired solution
Xdot(t) = [-sin(t) 0; 0 cos(t)]  # pmderiv of the desired solution
C(t) = [ -sin(t)  -1-sin(t)-(-1-10cos(t))*(1+cos(t));
-1-sin(t)-(-1-10cos(t))*(1+cos(t))   cos(t)- 2(-24 - 19sin(t))*(1 + sin(t)) ]  # corresponding C
Cd(t) = [ sin(t)  -1-cos(t)-(-1-10cos(t))*(1+sin(t));
-1-cos(t)-(-1-10cos(t))*(1+sin(t))    -cos(t)-2(-24-19sin(t))*(1 + sin(t)) ] # corresponding Cd
At = PeriodicFunctionMatrix(A,2*pi)
Ct = PeriodicFunctionMatrix(C,2*pi)
Cdt = PeriodicFunctionMatrix(Cd,2*pi)
Xt = PeriodicFunctionMatrix(X,2*pi)
Xd = PeriodicFunctionMatrix(Xdot,2*pi)
Ac = PeriodicFunctionMatrix(A(0),2*pi)
Cc = PeriodicFunctionMatrix(C(0),2*pi)
Cdc = PeriodicFunctionMatrix(Cd(0),2*pi)

Ad = monodromy(At,100)
@test PeriodicMatrixEquations.pseig3(Ad.M) ≈ PeriodicMatrixEquations.pseig3(Ad.M,fast=true) &&
      PeriodicMatrixEquations.pseig3(Ad.M,rev=false) ≈ PeriodicMatrixEquations.pseig3(Ad.M,fast=true,rev = false)

@time Yt = pclyap(At, Ct, K = 512, intpol = true, reltol = 1.e-12, abstol=1.e-12);
@time Yt1 = pclyap(At,Ct; K = 512, reltol = 1.e-12, abstol = 1.e-12,intpol=false);
@test norm(Xt.(ts).-Yt.(ts),Inf) < 1.e-7 && norm(Xt.(ts).-Yt1.(ts),Inf) < 1.e-7  

@time Yt = pclyap(Ac, Cc, stability_check = true, K = 512, intpol = true, reltol = 1.e-12, abstol=1.e-12);
@test norm(Yt.(ts).*Ac'.(ts).+Ac.(ts).*Yt.(ts).+Cc.(ts),Inf) < 1.e-7

@time Yt = pclyap(Ac, Cdc, stability_check = true, adj = true, K = 512, intpol = true, reltol = 1.e-12, abstol=1.e-12);
@test norm(Yt.(ts).*Ac.(ts).+Ac'.(ts).*Yt.(ts).+Cdc.(ts),Inf) < 1.e-7

@time Yt = pclyap(At, Ct(0), K = 512, intpol = true, reltol = 1.e-12, abstol=1.e-12);
@time Yt1 = pclyap(At,Ct(0); K = 512, reltol = 1.e-12, abstol = 1.e-12,intpol=false);
@test norm(Yt.(ts).-Yt1.(ts),Inf)  < 1.e-5  


@time Yt1 = pclyap(At, Ct, K = 1, reltol = 1.e-12, abstol=1.e-12, intpol = false);
@test Xt ≈ Yt1 && Xd ≈ pmderiv(Yt1) 

@time Yt1 = pclyap(At, Cdt, K = 1, adj = true, reltol = 1.e-12, abstol=1.e-12, intpol = false)
@test Xt ≈ Yt1 && Xd ≈ pmderiv(Yt1) 

@time Yt = pfclyap(At, Ct, K = 500, reltol = 1.e-12, abstol=1.e-12);
@test Xt ≈ Yt 

@time Yt = pfclyap(At, Ct(0), K = 500, intpol = true, reltol = 1.e-12, abstol=1.e-12);
@time Yt1 = pfclyap(At, Ct(0), K = 500, intpol = false, reltol = 1.e-12, abstol=1.e-12);
@test norm(Yt1.(ts) .- Yt.(ts),Inf) < 1.e-5 

@time Yt = prclyap(At, Cdt, K = 500, reltol = 1.e-12, abstol=1.e-12)
@test Xt ≈ Yt 

@time Yt = prclyap(At, Cdt(0), K = 500, intpol = true, reltol = 1.e-12, abstol=1.e-12);
@time Yt1 = prclyap(At, Cdt(0), K = 500, intpol = false, reltol = 1.e-12, abstol=1.e-12);
@test norm(Yt1.(ts) .- Yt.(ts),Inf) < 1.e-5 


@time Yt = pclyap(At,Ct*Ct'; adj = false, K = 512, reltol = 1.e-14, abstol = 1.e-14, intpol=false);
@time Ut = pcplyap(At,Ct; adj = false, K = 512, reltol = 1.e-14, abstol = 1.e-14, intpol=false);
@test Ut*Ut' ≈ Yt 
@time Ut1 = pfcplyap(At,Ct; K = 512, reltol = 1.e-14, abstol = 1.e-14);
@test Ut1*Ut1' ≈ Yt 

N = 2; 
@time Ut = pcplyap(At,Ct; adj = false, K = 512, reltol = 1.e-14, abstol = 1.e-14, intpol=false);
for N in (1,2,5,10,100,512)
    @time Yt = pclyap(At,Ct*Ct'; N, adj = false, K = 512, reltol = 1.e-14, abstol = 1.e-14, intpol=true);
    @test norm(Ut.(ts).*Ut'.(ts).-Yt.(ts),Inf) < 1.e-5
end

N = 2; 
@time Ut = pcplyap(At,Cdt; adj = true, K = 512, reltol = 1.e-14, abstol = 1.e-14, intpol=true);
for N in (1,2,5,10,100,512)
    @time Yt = pclyap(At,Cdt'*Cdt; N, adj = true, K = 512, reltol = 1.e-14, abstol = 1.e-14, intpol=true);
    @test norm(Ut'.(ts).*Ut.(ts).-Yt.(ts),Inf) < 1.e-5
end


@time Yt = pclyap(At,Ct'*Ct; adj = true, K = 1000, reltol = 1.e-14, abstol = 1.e-14, intpol=false);
@time Ut = pcplyap(At,Ct; adj = true, K = 1000, reltol = 1.e-14, abstol = 1.e-14, intpol=false);
@test Ut'*Ut ≈ Yt 
@time Ut1 = prcplyap(At,Ct; K = 512, reltol = 1.e-14, abstol = 1.e-14);
@test Ut1'*Ut1 ≈ Yt 

@time Yt = pclyap(At,Ct'*Ct; adj = true, K = 1000, reltol = 1.e-14, abstol = 1.e-14, intpol=true);
@time Ut = pcplyap(At,Ct; adj = true, K = 1000, reltol = 1.e-14, abstol = 1.e-14, intpol=true);
@test norm(Ut'.(ts).*Ut.(ts).-Yt.(ts),Inf) < 1.e-5 

@time Yt = pclyap(Ac,Cc'*Cc; adj = true, K = 1000, reltol = 1.e-14, abstol = 1.e-14, intpol=true);
@time Ut = pcplyap(Ac,Cc; adj = true, K = 1000, reltol = 1.e-14, abstol = 1.e-14, intpol=true);
@test norm(Ut'.(ts).*Ut.(ts).-Yt.(ts),Inf) < 1.e-5

@time Yt = pclyap(At,(Ct*Ct')(0); adj = false, K = 1000, reltol = 1.e-14, abstol = 1.e-14, intpol=true);
@time Ut = pcplyap(At,Ct(0); adj = false, K = 1000, reltol = 1.e-14, abstol = 1.e-14, intpol=true);
@test norm(Ut.(ts).*Ut'.(ts).-Yt.(ts),Inf) < 1.e-5
@time Ut1 = pfcplyap(At,Ct(0); K = 512, reltol = 1.e-14, abstol = 1.e-14);
@test norm(Ut1.(ts).*Ut1'.(ts).-Yt.(ts),Inf) .< 1.e-5 

@time Yt = pclyap(At,(Ct'*Ct)(0); adj = true, K = 1000, reltol = 1.e-14, abstol = 1.e-14, intpol=true);
@time Ut = pcplyap(At,Ct(0); adj = true, K = 1000, reltol = 1.e-14, abstol = 1.e-14, intpol=true);
@test norm(Ut'.(ts).*Ut.(ts).-Yt.(ts),Inf) < 1.e-5 
@time Ut1 = prcplyap(At,Ct(0); K = 512, reltol = 1.e-14, abstol = 1.e-14);
@test norm(Ut1'.(ts).*Ut1.(ts).-Yt.(ts),Inf) < 1.e-5 



# # check implicit solver based solution 
# @time Yt = pclyap(At,Ct*Ct'; adj = false, K = 512, reltol = 1.e-14, abstol = 1.e-14, intpol=false);
# @time Ut = pcplyap(At,Ct; adj = false, K = 512, reltol = 1.e-10, abstol = 1.e-10, intpol=false, implicit = true);
# Xt2 = Ut*Ut'; 
# @test norm(At*Yt+Yt*At'+Ct*Ct' - pmderiv(Yt)) < 1.e-4
# @test Xt2 ≈ Yt && norm(At*Xt2+Xt2*At'+Ct*Ct' - pmderiv(Ut)*Ut'-Ut*pmderiv(Ut)') < 1.e-4*norm(Xt2)


# @time Yt = pclyap(At,Ct'*Ct; adj = true, K = 512, reltol = 1.e-14, abstol = 1.e-14, intpol=false);
# @time Ut = pcplyap(At,Ct; adj = true, K = 512, reltol = 1.e-10, abstol = 1.e-10, intpol=false, implicit = true);
# Xt2 = Ut'*Ut; 
# @test norm(At'*Yt+Yt*At+Ct'*Ct + pmderiv(Yt)) < 1.e-4
# @test Xt2 ≈ Yt && norm(At'*Xt2+Xt2*At+Ct'*Ct + pmderiv(Ut)'*Ut+Ut'*pmderiv(Ut)) < 1.e-4*norm(Xt2)

# singular factors
At1 = [[At zeros(2,2)]; [zeros(2,2) At]]; Ct1 = [Ct; -Ct];
Ut1 = pgcplyap(At1, Ct1, 64; adj = false, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
@test all(cond.(Ut1.values) .> 1.e6)
@time Ut2 = pcplyap(At1,Ct1; adj = false, K = 64, reltol = 1.e-10, abstol = 1.e-10, intpol=false); 
ti = collect((0:63)*(2pi/64)).+eps(20.);
@test all(cond.(Ut2.(ti)) .> 1.e6)
@test all(isapprox.(Ut1.(ti),Ut2.(ti),rtol=1.e-6))


# # check explicit solver based solution  # fails but still works without errors
# @time Yt = pclyap(At,Ct*Ct'; adj = false, K = 512, reltol = 1.e-14, abstol = 1.e-14, intpol=false);
# @time Ut = pcplyap(At,Ct; adj = false, K = 512, solver = "non-stiff", reltol = 1.e-10, abstol = 1.e-6, intpol=false, implicit = false);
# Xt2 = Ut*Ut'; 
# @test norm(At*Yt+Yt*At'+Ct*Ct' - pmderiv(Yt)) < 1.e-4
# @test Xt2 ≈ Yt && norm(At*Xt2+Xt2*At'+Ct*Ct' - pmderiv(Ut)*Ut'-Ut*pmderiv(Ut)') < 1.e-4*norm(Xt2)


# @time Yt = pclyap(At,Ct'*Ct; adj = true, K = 512, reltol = 1.e-14, abstol = 1.e-14, intpol=false);
# @time Ut = pcplyap(At,Ct; adj = true, K = 512, reltol = 1.e-5, abstol = 1.e-5, intpol=false, implicit =false);
# Xt2 = Ut'*Ut; 
# @test norm(At'*Yt+Yt*At+Ct'*Ct + pmderiv(Yt)) < 1.e-4
# @test Xt2 ≈ Yt && norm(At'*Xt2+Xt2*At+Ct'*Ct + pmderiv(Ut)'*Ut+Ut'*pmderiv(Ut)) < 1.e-4*norm(Xt2)


K = 500
@time W0 = pgclyap(At, Ct, K; adj = false, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
Ts = 2pi/K
success = true
for i = 1:K
    Y  = PeriodicMatrixEquations.tvclyap(At, Ct, i*Ts, (i-1)*Ts, W0.values[mod(i+K-1,K)+1]; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10) 
    iw = i+1; iw > K && (iw = 1)
    success = success && norm(Y-W0.values[iw]) < 1.e-7
end
@test success

@time W1 = pgclyap(At, Cdt, K; adj = true, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
@time X0, Y0  = pgclyap2(At, Ct, Cdt, K; solver = "auto", reltol = 1.e-10, abstol = 1.e-10, stability_check = true);
@test X0 ≈ W0 && Y0 ≈ W1

@time W0 = pgclyap(Ac, Cc, K; adj = false, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
@time W1 = pgclyap(Ac, Cdc, K; adj = true, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
@time X0, Y0  = pgclyap2(Ac, Cc, Cdc, K; solver = "auto", reltol = 1.e-10, abstol = 1.e-10, stability_check = true);
@test X0 ≈ W0 && Y0 ≈ W1

@time W1 = pgclyap(At, Cdt, K; adj = true, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
@time X0, Y0  = pgclyap2(At, Ct(0), Cdt, K; solver = "auto", reltol = 1.e-10, abstol = 1.e-10, stability_check = true);
YYt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, Y0, At, Cdt; solver = "auto", adj = true, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((At'*YYt+YYt*At+Cdt+pmderiv(YYt)).(ts),Inf) < 1.e-5  
XXt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, X0, At; solver = "auto", reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((At*XXt+XXt*At'-pmderiv(XXt)).(ts),Inf) < 1.e-5  
@test Y0 ≈ W1


@time W1 = pgclyap(Ac, Cdc, K; adj = true, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
@time X0, Y0  = pgclyap2(Ac, Cc, Cdc, K; solver = "auto", reltol = 1.e-10, abstol = 1.e-10, stability_check = true);
@test Y0 ≈ W1

@time W1 = pgclyap(Ac, Cdc, K; adj = true, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
@time X0, Y0  = pgclyap2(Ac, Cc(0), Cdc, K; solver = "auto", reltol = 1.e-10, abstol = 1.e-10, stability_check = true);
XXt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, X0, Ac; solver = "auto", reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((Ac*XXt+XXt*Ac'-pmderiv(XXt)).(ts),Inf) < 1.e-5  
@test Y0 ≈ W1


K = 10
@time W0 = pgclyap(At, Ct, K; adj = false, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
ns = 10; ts = sort([0.; rand(ns-1)*2*pi; 2*pi])
success = true
for t in ts
    Xtval = PeriodicMatrixEquations.tvclyap_eval(t, W0, At, Ct; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10) 
    success = success && norm(Xtval-Xt(t)) < 1.e-7
end
@test success
@time XXt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, W0, At, Ct; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((At*XXt+XXt*At'+Ct-pmderiv(XXt)).(ts),Inf) < 1.e-5   



K = 100
solver = "non-stiff"
for solver in ("non-stiff", "stiff", "auto")
    println("solver = $solver")
    @time W0 = pgclyap(At, Ct, K; adj = false, solver, reltol = 1.e-10, abstol = 1.e-10);
    Y  = PeriodicMatrixEquations.tvclyap(At, Ct, ts, W0.values[1]; solver, adj = false, reltol = 1.e-10, abstol = 1.e-10) 
    Xtval = [PeriodicMatrixEquations.tvclyap_eval(t, W0, At, Ct; solver, adj = false, reltol = 1.e-10, abstol = 1.e-10)  for t in ts]
    @test norm(Y-Xtval) < 1.e-5
end

K= 100
solver = "non-stiff"
for solver in ("non-stiff", "stiff", "auto")
    println("solver = $solver")
    @time W0 = pgclyap(At, Cdt, K; adj = true, solver, reltol = 1.e-10, abstol = 1.e-10);
    Xtval = [PeriodicMatrixEquations.tvclyap_eval(t, W0, At, Cdt; solver, adj = true, reltol = 1.e-10, abstol = 1.e-10)  for t in 2*ts]
    Y  = PeriodicMatrixEquations.tvclyap(At, Cdt, 2*ts, W0.values[1]; solver, adj = true, reltol = 1.e-10, abstol = 1.e-10) 
    @test norm(Y-Xtval) < 1.e-5
end



K = 100
solver = "non-stiff"
for solver in ("non-stiff", "stiff", "auto")
    println("solver = $solver")
    @time W0 = pgclyap(At, Ct, K; adj = false, solver, reltol = 1.e-10, abstol = 1.e-10);
    Y  = PeriodicMatrixEquations.tvclyap(At, Ct, W0.values[1]; solver, adj = false, reltol = 1.e-10, abstol = 1.e-10); 
    Xtval = [PeriodicMatrixEquations.tvclyap_eval(t, W0, At, Ct; solver, adj = false, reltol = 1.e-10, abstol = 1.e-10)  for t in ts]
    @test norm(Y.(ts)-Xtval) < 1.e-5
end

K = 100
solver = "non-stiff"
for solver in ("non-stiff", "stiff", "auto")
    println("solver = $solver")
    @time W0 = pgclyap(At, Ct, K; adj = false, solver, reltol = 1.e-10, abstol = 1.e-10);
    Y  = PeriodicMatrixEquations.tvclyap(At, Ct, W0; solver, adj = false, reltol = 1.e-10, abstol = 1.e-10); 
    Xtval = [PeriodicMatrixEquations.tvclyap_eval(t, W0, At, Ct; solver, adj = false, reltol = 1.e-10, abstol = 1.e-10)  for t in ts]
    @test norm(Y.(ts)-Xtval) < 1.e-5
end


@time W0 = pgclyap(At, Ct, K; adj = false, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
Y  = PeriodicMatrixEquations.tvclyap(At, Ct, ts, W0.values[1]; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10) 
Xtval = [PeriodicMatrixEquations.tvclyap_eval(t, W0, At, Ct; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10)  for t in ts]
@test norm(Y-Xtval) < 1.e-7

success = true
for t in ts
    Xtval = PeriodicMatrixEquations.tvclyap_eval(t, W0, At, Ct; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10) 
    success = success && norm(Xtval-Xt(t)) < 1.e-7
end
@test success
@time XXt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, W0, At, Ct; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((At*XXt+XXt*At'+Ct-pmderiv(XXt)).(ts),Inf) < 1.e-5   


K = 500
W0 = pgclyap(At,  Cdt, K; adj = true, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
Ts = 2pi/K
success = true
for i = K:-1:1
    iw = i+1; iw > K && (iw = 1)
    Y  = PeriodicMatrixEquations.tvclyap(At, Cdt, (i-1)*Ts, i*Ts, W0.values[iw]; solver = "auto", adj = true, reltol = 1.e-10, abstol = 1.e-10) 
    success = success && norm(Y-W0.values[i]) < 1.e-7
end
@test success

K = 10
W0 = pgclyap(At, Cdt, K; adj = true, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
ns = 10; ts = sort([0.; rand(ns-1)*2*pi; 2*pi])
success = true
for t in ts
    Xtval = PeriodicMatrixEquations.tvclyap_eval(t, W0, At, Cdt; solver = "auto", adj = true, reltol = 1.e-10, abstol = 1.e-10) 
    success = success && norm(Xtval-Xt(t)) < 1.e-7
end
@test success

XXt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, W0, At, Cdt; solver = "auto", adj = true, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((At'*XXt+XXt*At+Cdt+pmderiv(XXt)).(ts),Inf) < 1.e-5  


solver = "non-stiff"
for solver in ("non-stiff", "stiff", "auto")
    println("solver = $solver")
    @time Yt = pclyap(At, Ct; solver, K = 500, reltol = 1.e-10, abstol = 1.e-10,intpol=true);
    #@test Xt ≈ Yt 
    @test norm(Xt.(ts).-Yt.(ts),Inf) < 1.e-7
    @time Yt = pclyap(At, Ct; solver, K = 500, reltol = 1.e-10, abstol = 1.e-10,intpol=false);
    #@test Xt ≈ Yt 
    @test norm(Xt.(ts).-Yt.(ts),Inf) < 1.e-7

end


# solve using harmonic arrays
Ah = convert(HarmonicArray,At);
Ch = convert(HarmonicArray,Ct);
Cdh = convert(HarmonicArray,Cdt);
Xh = convert(HarmonicArray,Xt);
Xdh = convert(HarmonicArray,Xd);

@time Yt = pclyap(Ah, Ch, K = 500, reltol = 1.e-10, abstol = 1.e-10);
@time Yt1 = pclyap(Ah,Ch; K = 512, reltol = 1.e-10, abstol = 1.e-10,intpol=true);
Yh = convert(HarmonicArray,Yt);
Yh1 = convert(HarmonicArray,Yt1);
@test Xh ≈ Yh && Xh ≈ Yh1

@time Yt = pclyap(Ah, Cdh, K = 500, adj = true, reltol = 1.e-10, abstol = 1.e-10);
Yh = convert(HarmonicArray,Yt);
@test Xh ≈ Yh && Xdh ≈ pmderiv(Yh) 

@time Yt = pclyap(Ah,Ch'*Ch; adj = true, K = 1000, reltol = 1.e-14, abstol = 1.e-14);
@time Ut = pcplyap(Ah,Ch; adj = true, K = 1000, reltol = 1.e-14, abstol = 1.e-14);
@test all(norm.(Ut'.(ts).*Ut.(ts).-Yt.(ts),Inf) .< 1.e-5) 


@time Yt = pfclyap(Ah, Ch, K = 500, reltol = 1.e-10, abstol = 1.e-10);
Yh = convert(HarmonicArray,Yt);
@test Xh ≈ Yh 

@time Yt = prclyap(Ah, Cdh, K = 500, reltol = 1.e-10, abstol = 1.e-10);
Yh = convert(HarmonicArray,Yt);
@test Xh ≈ Yh 

# solve using Fourier function matrices
Af = convert(FourierFunctionMatrix,At);
Cf = convert(FourierFunctionMatrix,Ct); 
Cdf = convert(FourierFunctionMatrix,Cdt)
Xf = convert(FourierFunctionMatrix,Xt);
Xdf = convert(FourierFunctionMatrix,Xd);
Afc = convert(FourierFunctionMatrix,Ac);
Cfc = convert(FourierFunctionMatrix,Cc); 
Cdfc = convert(FourierFunctionMatrix,Cdc);
ts = [0.4459591888577492, 1.2072325802972004, 1.9910835248218244, 2.1998199838900527, 2.4360161318589695, 
     2.9004720268745463, 2.934294124172935, 4.149208861412936, 4.260935465730602, 5.956614157549958]


@time Yt = pclyap(Af, Cf, K = 500, reltol = 1.e-10, abstol = 1.e-10, intpol=false, stability_check = true);
@time Yt1 = pclyap(Af,Cf; K = 512, reltol = 1.e-10, abstol = 1.e-10,intpol=true);
Yf = convert(FourierFunctionMatrix,Yt);
@test Xf ≈ Yf && Xdf ≈ pmderiv(Yf) && 
      norm(Af*Yf+Yf*Af'+Cf-pmderiv(Yf)) < 1.e-7 && norm(Yt1.(ts).-Yt.(ts),Inf) < 1.e-7

@time Yt = pclyap(Af, Cdf, K = 500, adj = true, reltol = 1.e-10, abstol = 1.e-10)
Yf = convert(FourierFunctionMatrix,Yt);
@test Xf ≈ Yf && Xdf ≈ pmderiv(Yf) && norm(Af'*Yf+Yf*Af+Cdf+pmderiv(Yf)) < 1.e-7

@time Yt = pfclyap(Af, Cf, K = 500, reltol = 1.e-10, abstol = 1.e-10);
Yf = convert(FourierFunctionMatrix,Yt);
@test Xf ≈ Yf && norm(Af*Yf+Yf*Af'+Cf-pmderiv(Yf)) < 1.e-7

@time Yt = prclyap(Af, Cdf, K = 500, reltol = 1.e-10, abstol = 1.e-10)
Yf = convert(FourierFunctionMatrix,Yt);
@test Xf ≈ Yf && norm(Af'*Yf+Yf*Af+Cdf+pmderiv(Yf)) < 1.e-7

@time Yt = pclyap(Af,Cf'*Cf; adj = true, K = 500, reltol = 1.e-10, abstol = 1.e-10, intpol=true);
@time Ut = pcplyap(Af,Cf; adj = true, K = 500, reltol = 1.e-10, abstol = 1.e-10);
@test norm(Ut'.(ts).*Ut.(ts).-Yt.(ts),Inf) < 1.e-5

@time Yt = pclyap(Af,Cf'*Cf; adj = true, K = 500, reltol = 1.e-10, abstol = 1.e-10, intpol=true);
@time Ut = pcplyap(Af,Cf; adj = true, K = 500, reltol = 1.e-10, abstol = 1.e-10, intpol=true);
@test norm(Ut'.(ts).*Ut.(ts).-Yt.(ts),Inf) < 1.e-5


@time Yt = pclyap(Afc,Cfc'*Cfc; adj = true, K = 500, reltol = 1.e-10, abstol = 1.e-10, intpol=true);
@time Ut = pcplyap(Afc,Cfc; adj = true, K = 500, reltol = 1.e-10, abstol = 1.e-10);
@test norm(Ut'.(ts).*Ut.(ts).-Yt.(ts),Inf) < 1.e-5


@time Yt = prclyap(Af,Cf(0)'*Cf(0); K = 500, reltol = 1.e-10, abstol = 1.e-10);
@time Ut = prcplyap(Af,Cf(0); K = 500, reltol = 1.e-10, abstol = 1.e-10);
@test all(norm.(Ut'.(ts).*Ut.(ts).-Yt.(ts),Inf) .< 1.e-5) 

@time Yt = pclyap(Af,Cdf*Cdf'; adj = false, K = 500, reltol = 1.e-10, abstol = 1.e-10, intpol=true);
@time Ut = pcplyap(Af,Cdf; adj = false, K = 500, reltol = 1.e-10, abstol = 1.e-10, intpol=true);
@show norm(Ut.(ts).*Ut'.(ts).-Yt.(ts),Inf)
@test norm(Ut.(ts).*Ut'.(ts).-Yt.(ts),Inf) < 1.e-7

@time Yt = pclyap(Af,Cdf*Cdf'; adj = false, K = 500, reltol = 1.e-10, abstol = 1.e-10, intpol=false);
@time Ut = pcplyap(Af,Cdf; adj = false, K = 500, reltol = 1.e-10, abstol = 1.e-10, intpol=false);
@show norm(Ut.(ts).*Ut'.(ts).-Yt.(ts),Inf)
@test norm(Ut.(ts).*Ut'.(ts).-Yt.(ts),Inf) < 1.e-7

@time Yt = pfclyap(Af,Cdf*Cdf'; K = 500, reltol = 1.e-14, abstol = 1.e-14);
@time Ut = pfcplyap(Af,Cdf; K = 500, reltol = 1.e-14, abstol = 1.e-14);
#@show norm(Ut.(ts).*Ut'.(ts).-Yt.(ts),Inf)
@test norm(Ut.(ts).*Ut'.(ts).-Yt.(ts),Inf) < 1.e-7

@time Yt = pfclyap(Af,(Cdf*Cdf')(0); K = 500, reltol = 1.e-14, abstol = 1.e-14);
@time Ut = pfcplyap(Af,Cdf(0); K = 500, reltol = 1.e-14, abstol = 1.e-14);
#@show norm(Ut.(ts).*Ut'.(ts).-Yt.(ts),Inf)
@test norm(Ut.(ts).*Ut'.(ts).-Yt.(ts),Inf) < 1.e-7

K = 500
@time W0 = pgclyap(Af, Cf, K; adj = false, solver = "auto", reltol = 1.e-10, abstol = 1.e-10, stability_check = true);
Ts = 2pi/K
success = true
for i = 1:K
    Y  = PeriodicMatrixEquations.tvclyap(Af, Cf, i*Ts, (i-1)*Ts, W0.values[mod(i+K-1,K)+1]; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10) 
    iw = i+1; iw > K && (iw = 1)
    success = success && norm(Y-W0.values[iw]) < 1.e-7
end
@test success

@time W1 = pgclyap(Af, Cdf, K; adj = true, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
@time X0, Y0  = pgclyap2(Af, Cf, Cdf, K; solver = "auto", reltol = 1.e-10, abstol = 1.e-10, stability_check = true);
@test X0 ≈ W0 && Y0 ≈ W1

@time W0 = pgclyap(Afc, Cfc, K; adj = false, solver = "auto", reltol = 1.e-10, abstol = 1.e-10, stability_check = true);
@time W1 = pgclyap(Afc, Cdfc, K; adj = true, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
@time X0, Y0  = pgclyap2(Afc, Cfc, Cdfc, K; solver = "auto", reltol = 1.e-10, abstol = 1.e-10, stability_check = true);
@test X0 ≈ W0 && Y0 ≈ W1

K = 100
@time W1 = pgclyap(Af, Cdf, K; adj = true, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
@time X0, Y0  = pgclyap2(Af, Cf(0), Cdf, K; solver = "auto", reltol = 1.e-10, abstol = 1.e-10, stability_check = true);
YYt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, Y0, Af, Cdf; solver = "auto", adj = true, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((Af'.(ts).*YYt.(ts).+YYt.(ts).*Af.(ts).+Cdf.(ts).+pmderiv(YYt).(ts)),Inf) < 1.e-5  
XXt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, X0, Af; solver = "auto", reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test  norm((Af'.(ts).*YYt.(ts).+YYt.(ts).*Af.(ts).+Cdf.(ts).+pmderiv(YYt).(ts)),Inf) < 1.e-5
@test Y0 ≈ W1
@time X0, Y0  = pgclyap2(Afc, Cf(0), Cdfc, K; solver = "auto", reltol = 1.e-10, abstol = 1.e-10, stability_check = true);
YYt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, Y0, Afc, Cdfc; solver = "auto", adj = true, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((Afc'.(ts).*YYt.(ts).+YYt.(ts).*Afc.(ts).+Cdfc.(ts).+pmderiv(YYt).(ts)),Inf) < 1.e-5 


K = 100
solver = "non-stiff"
for solver in ("non-stiff", "stiff", "auto")
    println("solver = $solver")
    @time W0 = pgclyap(Af, Cf, K; adj = false, solver, reltol = 1.e-10, abstol = 1.e-10);
    Y  = PeriodicMatrixEquations.tvclyap(Af, Cf, ts, W0.values[1]; solver, adj = false, reltol = 1.e-10, abstol = 1.e-10) 
    Xtval = [PeriodicMatrixEquations.tvclyap_eval(t, W0, Af, Cf; solver, adj = false, reltol = 1.e-10, abstol = 1.e-10)  for t in ts]
    @test norm(Y-Xtval) < 1.e-5
end

K = 100
solver = "non-stiff"
for solver in ("non-stiff", "stiff", "auto")
    println("solver = $solver")
    @time W0 = pgclyap(Af, Cdf, K; adj = true, solver, reltol = 1.e-10, abstol = 1.e-10);
    Xtval = [PeriodicMatrixEquations.tvclyap_eval(t, W0, Af, Cdf; solver, adj = true, reltol = 1.e-10, abstol = 1.e-10)  for t in 2*ts]
    Y  = PeriodicMatrixEquations.tvclyap(Af, Cdf, 2*ts, W0.values[1]; solver, adj = true, reltol = 1.e-14, abstol = 1.e-14) 
    @test norm(Y-Xtval) < 1.e-5
end


K = 100
solver = "non-stiff"
for solver in ("non-stiff", "stiff", "auto")
    println("solver = $solver")
    @time W0 = pgclyap(Af, Cf, K; adj = false, solver, reltol = 1.e-10, abstol = 1.e-10);
    Y  = PeriodicMatrixEquations.tvclyap(Af, Cf, W0.values[1]; solver, adj = false, reltol = 1.e-10, abstol = 1.e-10); 
    Xtval = [PeriodicMatrixEquations.tvclyap_eval(t, W0, Af, Cf; solver, adj = false, reltol = 1.e-10, abstol = 1.e-10)  for t in ts]
    @test norm(Y.(ts)-Xtval) < 1.e-5 
end
K = 100
solver = "non-stiff"
for solver in ("non-stiff", "stiff", "auto")
    println("solver = $solver")
    @time W0 = pgclyap(Af, Cf, K; adj = false, solver, reltol = 1.e-10, abstol = 1.e-10);
    Y  = PeriodicMatrixEquations.tvclyap(Af, Cf, W0; solver, adj = false, reltol = 1.e-10, abstol = 1.e-10); 
    Xtval = [PeriodicMatrixEquations.tvclyap_eval(t, W0, Af, Cf; solver, adj = false, reltol = 1.e-10, abstol = 1.e-10)  for t in ts]
    @test norm(Y.(ts)-Xtval) < 1.e-5 
end

@time Yt = pclyap(Afc, Cfc, stability_check = true, K = 512, intpol = true, reltol = 1.e-12, abstol=1.e-12);
@test norm(Yt.(ts).*Afc'.(ts).+Afc.(ts).*Yt.(ts).+Cfc.(ts),Inf) < 1.e-7

K = 100
@time W1 = pgclyap(Af, Cdf, K; adj = true, solver = "auto", reltol = 1.e-10, abstol = 1.e-10)
@time X0, Y0  = pgclyap2(Af, Cf(0), Cdf, K; solver = "auto", reltol = 1.e-10, abstol = 1.e-10, stability_check = true);
@test Y0 ≈ W1
YYt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, Y0, Af, Cdf; solver = "auto", adj = true, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((Af'*YYt+YYt*Af+Cdf+pmderiv(YYt,method="4d")).(ts),Inf) < 1.e-5  
XXt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, X0, Af; solver = "auto", reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm(Af.(ts).*XXt.(ts).+XXt.(ts).*Af'.(ts).-pmderiv(XXt).(ts),Inf) < 1.e-5  

K = 100
@time W1 = pgclyap(Afc, Cdfc, K; adj = true, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
@time X0, Y0  = pgclyap2(Afc, Cf(0), Cdfc, K; solver = "auto", reltol = 1.e-10, abstol = 1.e-10, stability_check = true);
@test Y0 ≈ W1
YYt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, Y0, Afc, Cdfc; solver = "auto", adj = true, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((Afc'.(ts).*YYt.(ts).+YYt.(ts).*Afc.(ts).+Cdfc.(ts).+pmderiv(YYt).(ts)),Inf) < 1.e-5  
XXt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, X0, Afc; solver = "auto", reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm(Afc.(ts).*XXt.(ts).+XXt.(ts).*Afc'.(ts).-pmderiv(XXt).(ts),Inf) < 1.e-5  


# solve using periodic time-series matrices
K = 512
Ats = convert(PeriodicTimeSeriesMatrix,At;ns = K);
Cts = convert(PeriodicTimeSeriesMatrix,Ct;ns = K);
Cdts = convert(PeriodicTimeSeriesMatrix,Cdt;ns = K);
Atsc = convert(PeriodicTimeSeriesMatrix,Ac;ns = K);
Ctsc = convert(PeriodicTimeSeriesMatrix,Cc;ns = K);

ts = [0.4459591888577492, 1.2072325802972004, 1.9910835248218244, 2.1998199838900527, 2.4360161318589695, 
     2.9004720268745463, 2.934294124172935, 4.149208861412936, 4.260935465730602, 5.956614157549958]


@time Yt1 = pclyap(Ats, Cts; K = 1, reltol = 1.e-14, abstol = 1.e-14); 
@test norm((Ats.(ts).*Yt1.(ts)).+(Yt1.(ts).*(Ats').(ts)).+Cts.(ts).-pmderiv(Yt1).(ts),Inf) < 1.e-5 
#@test norm((Ats*Yt1+Yt1*Ats'+Cts-pmderiv(Yt1)).(ts)) < 1.e-5 

@time Yt = pclyap(Ats,Cts; K, reltol = 1.e-14, abstol = 1.e-14,intpol=true);
@test norm((Ats*Yt+Yt*Ats'+Cts-pmderiv(Yt)).(ts)) < 1.e-5 

Xt1 = pclyap(Ats, Cts; K = 128, reltol = 1.e-14, abstol = 1.e-14,intpol=true);
@test norm((Ats*Xt1+Xt1*Ats'+Cts-pmderiv(Xt1)).(ts)) < 1.e-5 

Xt2 = pclyap(Ats, Cts; K = 256, reltol = 1.e-14, abstol = 1.e-14,intpol=true);
@test norm((Ats*Xt2+Xt2*Ats'+Cts-pmderiv(Xt2)).(ts)) < 1.e-5 
@test norm((Xt1-Xt2).(ts)) < 1.e-7

Ka = 10; Kc = 1; 
Ats1 = convert(PeriodicTimeSeriesMatrix,At;ns = Ka);
Cts1 = convert(PeriodicTimeSeriesMatrix,Ct;ns = Kc);
Xt1 = pclyap(Ats1, Cts1; K = 10, reltol = 1.e-14, abstol = 1.e-14);
@test norm((Ats1*Xt1+Xt1*Ats1'+Cts1-pmderiv(Xt1)).(ts)) < 1.e-5 

Xt2 = pclyap(Ats1, Cts1; K = 2000, reltol = 1.e-14, abstol = 1.e-14);
@test norm((Ats1*Xt2+Xt2*Ats1'+Cts1-pmderiv(Xt2)).(ts)) < 1.e-5 
#@test norm((Xt1-Xt2).(ts)) < 1.e-7

# @time Yt = pclyap(convert(HarmonicArray,Ats),convert(HarmonicArray,Cts)'*convert(HarmonicArray,Cts); adj = true, K = 1000, reltol = 1.e-14, abstol = 1.e-14);
# @time Ut = pcplyap(Ats,Cts; adj = true, K = 1000, reltol = 1.e-14, abstol = 1.e-14);
# @test all(norm.(Ut'.(ts).*Ut.(ts).-Yt.(ts),Inf) .< 1.e-5) 

K = 512
W0 = pgclyap(Ats, Cts, K; adj = false, solver = "auto", reltol = 1.e-10, abstol = 1.e-10, stability_check = true);
Xts = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, W0, Ats, Cts; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((Ats*Xts+Xts*Ats'+Cts-pmderiv(Xts)).(ts)) < 1.e-6

W0 = pgclyap(Ats, Cts, K; adj = true, solver = "auto", reltol = 1.e-10, abstol = 1.e-10, stability_check = true);
Xts = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, W0, Ats, Cts; solver = "auto", adj =true, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((Ats'*Xts+Xts*Ats+Cts+pmderiv(Xts)).(ts)) < 1.e-6

Cts2=set_period(Cts,4pi)
W0 = pgclyap(Ats, Cts2, K; adj = false, solver = "auto", reltol = 1.e-10, abstol = 1.e-10, stability_check = true);
Xts = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, W0, Ats, Cts2; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10),4*pi)
@test norm((Ats*Xts+Xts*Ats'+Cts2-pmderiv(Xts)).(ts)) < 1.e-6


W0 = pgclyap(Atsc, Ctsc, K; adj = false, solver = "auto", reltol = 1.e-10, abstol = 1.e-10, stability_check = true);
Xts = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, W0, Atsc, Ctsc; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((Atsc*Xts+Xts*Atsc'+Ctsc-pmderiv(Xts)).(ts)) < 1.e-6

# K = 1000;
# Ats2 = convert(PeriodicTimeSeriesMatrix,At;ns = K);
# Cts2 = convert(PeriodicTimeSeriesMatrix,Ct;ns = K);
# Xt1 = pclyap(Ats2, Cts2; K = 1, reltol = 1.e-14, abstol = 1.e-14);
# @time Xt2 = pfclyap(Ats2, Cts2; reltol = 1.e-14, abstol = 1.e-14);
# @test norm((Xt1-Xt).(ts)) < 1.0e-3 &&
#       norm((Xt2-Xt).(ts)) < 1.0e-3

# Xt1 = pclyap(Ats, Cdts; K = 100, adj = true, reltol = 1.e-14, abstol = 1.e-14,intpol=true);
# Xt2 = pclyap(Ats, Cdts; K = 200, adj = true, reltol = 1.e-14, abstol = 1.e-14,intpol=true);
# @test norm((Xt1-Xt2).(ts)) < 1.e-7


# Cdts1 = convert(PeriodicTimeSeriesMatrix,Cdt;ns = Kc);
# Xt1 = pclyap(Ats1, Cdts1; K = 100, adj = true, reltol = 1.e-14, abstol = 1.e-14,intpol=true);
# Xt2 = pclyap(Ats1, Cdts1; K = 200, adj = true, reltol = 1.e-14, abstol = 1.e-14,intpol=true);
# @test norm((Xt1-Xt2).(ts)) < 1.e-5

# Cdts2 = convert(PeriodicTimeSeriesMatrix,Cdt;ns = 10000);
# Xt1 = pclyap(Ats2, Cdts2; K = 1, adj = true, reltol = 1.e-14, abstol = 1.e-14);
# @time Xt2 = prclyap(Ats2, Cdts2; K = 1, reltol = 1.e-14, abstol = 1.e-14);
# @test norm((Xt1-Xt2).(ts)) < 1.e-5

# @time Yt = pfclyap(Ats, Cts; reltol = 1.e-14, abstol = 1.e-14);
# @test norm((Ats*Yt+Yt*Ats'+Cts-pmderiv(Yt)).(ts)) < 1.e-5 

# @time Yt = prclyap(Ats, Cdts; reltol = 1.e-14, abstol = 1.e-14);
# @test norm((Ats'*Yt+Yt*Ats+Cdts+pmderiv(Yt)).(ts)) < 1.e-5


# solve using periodic switching matrices
Asw = convert(PeriodicSwitchingMatrix,Ats)
Csw = convert(PeriodicSwitchingMatrix,Cts)
Cdsw = convert(PeriodicSwitchingMatrix,Cdts)
Aswc = convert(PeriodicSwitchingMatrix,Ac)
Cswc = convert(PeriodicSwitchingMatrix,Cc)
Cdswc = convert(PeriodicSwitchingMatrix,Cdc)

ts = [0.4459591888577492, 1.2072325802972004, 1.9910835248218244, 2.1998199838900527, 2.4360161318589695, 
     2.9004720268745463, 2.934294124172935, 4.149208861412936, 4.260935465730602, 5.956614157549958]

@time Yt = pclyap(Asw, Csw; K = 100, reltol = 1.e-14, abstol = 1.e-14,stability_check=true);
@test norm((Asw.(ts).*Yt.(ts).+Yt.(ts).*Asw'.(ts).+Csw.(ts).-pmderiv(Yt).(ts)),Inf) < 1.e-5 

@time Yt = pclyap(Aswc, Cswc; K = 100, reltol = 1.e-14, abstol = 1.e-14,stability_check=true);
@test norm((Aswc.(ts).*Yt.(ts).+Yt.(ts).*Aswc'.(ts).+Cswc.(ts).-pmderiv(Yt).(ts)),Inf) < 1.e-5 

# @time Yt = pclyap(Asw, Cdsw; K = 2, adj = true, reltol = 1.e-14, abstol = 1.e-14)
# @test norm((Asw'*Yt+Yt*Asw+Cdsw+pmderiv(Yt)).(ts)) < 1.e-5  

@time Yt = pfclyap(Asw, Csw; K = 100, reltol = 1.e-14, abstol = 1.e-14);
@test norm((Asw.(ts).*Yt.(ts).+Yt.(ts).*Asw'.(ts).+Csw.(ts).-pmderiv(Yt).(ts)),Inf) < 1.e-5 

@time Yt = pfclyap(Asw, Csw(0); K = 200, reltol = 1.e-14, abstol = 1.e-14);
@test norm((Asw.(ts).*Yt.(ts).+Yt.(ts).*Asw'.(ts).+Csw.(0).-pmderiv(Yt).(ts)),Inf) < 1.e-5 

@time Yt = prclyap(Asw, Cdsw; K = 100, reltol = 1.e-14, abstol = 1.e-14);
@test norm((Asw'.(ts).*Yt.(ts).+Yt.(ts).*Asw.(ts).+Cdsw.(ts).+pmderiv(Yt).(ts)),Inf) < 1.e-5 

@time Yt = prclyap(Asw, Cdsw(0); K = 200, reltol = 1.e-14, abstol = 1.e-14);
@test norm((Asw'.(ts).*Yt.(ts).+Yt.(ts).*Asw.(ts).+Cdsw.(0).+pmderiv(Yt).(ts)),Inf) < 1.e-5 



A4(t) = [0  1; -cos(t)-1 -2-sin(t)]
C4(t) = [ -sin(t)  -1-sin(t)-(-1-10cos(t))*(1+cos(t));
-1-sin(t)-(-1-10cos(t))*(1+cos(t))   cos(t)- 2(-24 - 19sin(t))*(1 + sin(t)) ]  
# C(t) = [ -sin(t)  -1-sin(t);
# -1-sin(t)   cos(t)]  
#C(t) = [ 1 0;0 1. ]  
tsa = [0., pi/4, pi/2, 3pi/2]; Ats =  [A4(t) for t in tsa]
tsc = [0., 3pi/4, pi]; Cts =  [C4(t) for t in tsc]
Asw = PeriodicSwitchingMatrix(Ats,tsa,2pi)
Csw = PeriodicSwitchingMatrix(Cts,tsc,2pi)
Ast = convert(PeriodicFunctionMatrix,Asw)
Cst = convert(PeriodicFunctionMatrix,Csw)
ts = [0.4459591888577492, 1.2072325802972004, 1.9910835248218244, 2.1998199838900527, 2.4360161318589695, 
     2.9004720268745463, 2.934294124172935, 4.149208861412936, 4.260935465730602, 5.956614157549958]



K = 500
W0 = pgclyap(Ast, Cst, K; adj = false, solver = "auto", reltol = 1.e-10, abstol = 1.e-10, stability_check = true);
Ts = 2pi/K
success = true
for i = 1:K
    Y  = PeriodicMatrixEquations.tvclyap(Ast, Cst, i*Ts, (i-1)*Ts, W0.values[mod(i+K-1,K)+1]; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10) 
    iw = i+1; iw > K && (iw = 1)
    success = success && norm(Y-W0.values[iw]) < 1.e-3
end
@test success
Xst = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, W0, Ast, Cst; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((Ast*Xst+Xst*Ast'+Cst-pmderiv(Xst)).(ts)) < 1.e-6


K = 100;
#K = 1
W1 = pgclyap(Asw, Csw, K; adj = false, solver = "auto", reltol = 1.e-10, abstol = 1.e-10, stability_check = true);
W2 = pgclyap(Asw, Csw, 1; adj = false, solver = "auto", reltol = 1.e-10, abstol = 1.e-10, stability_check = true);
success = true
Ks = length(W1.ts)
for i = 1:Ks
    tf = i == Ks ? W1.period/W1.nperiod : W1.ts[i+1]
    Y  = PeriodicMatrixEquations.tvclyap(Asw, Csw, tf, W1.ts[i], W1.values[mod(i+Ks-1,Ks)+1]; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10) 
    iw = i+1; iw > Ks && (iw = 1)
    success = success && norm(Y-W1.values[iw]) < 1.e-3
end
@test success

XXt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, W1, Asw, Csw; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((Ast*XXt+XXt*Ast'+Cst-pmderiv(XXt)).(rand(10)*2*pi)) < 1.e-6
@test norm((Xst-XXt).(ts)) < 1.e-3
XXt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, W2, Asw, Csw; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((Ast*XXt+XXt*Ast'+Cst-pmderiv(XXt)).(rand(10)*2*pi)) < 1.e-6
@test norm((Xst-XXt).(ts)) < 1.e-3

W1 = pgclyap(Aswc, Cswc, K; adj = false, solver = "auto", reltol = 1.e-10, abstol = 1.e-10, stability_check = true);
success = true
Ks = length(W1.ts)
for i = 1:Ks
    tf = i == Ks ? W1.period/W1.nperiod : W1.ts[i+1]
    Y  = PeriodicMatrixEquations.tvclyap(Aswc, Cswc, tf, W1.ts[i], W1.values[mod(i+Ks-1,Ks)+1]; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10) 
    iw = i+1; iw > Ks && (iw = 1)
    success = success && norm(Y-W1.values[iw]) < 1.e-3
end
@test success


# using Plots
# t = [0;sort(rand(200)*2*pi);2*pi]; n = length(t)
# y1 = Xt.(t);
# y2 = XXt.(t);

# # y1 = pmderiv(Xt).(t);
# # y2 = pmderiv(XXt).(t);


# plot(t,[y1[i][1,1] for i in 1:n])
# plot!(t,[y2[i][1,1] for i in 1:n])

# t2 = [W1.ts;2*pi]
# ns2 = length(t2)
# x2 = W1.(t2)
# plot!(t2,[x2[i][1,1] for i in 1:ns2])

K = 500
@time W0 = pgclyap(Ast, Cst, K; adj = true, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
Xst = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, W0, Ast, Cst; solver = "auto", adj = true, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((Ast'*Xst+Xst*Ast+Cst+pmderiv(Xst)).(ts)) < 1.e-6

K = 100;
K = 1
W1 = pgclyap(Asw, Csw, K; adj = true, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
Ks = length(W1.ts)
success = true
for i = Ks:-1:1
    iw = i+1; iw > Ks && (iw = 1)
    t0 = i == Ks ? W1.period/W1.nperiod : W1.ts[i+1]
    Y  = PeriodicMatrixEquations.tvclyap(Asw, Csw, W1.ts[i], t0, W1.values[iw]; solver = "auto", adj = true, reltol = 1.e-10, abstol = 1.e-10) 
    success = success && norm(Y-W1.values[i]) < 1.e-7
end
@test success

XXt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvclyap_eval(t, W1, Asw, Csw; solver = "auto", adj = true, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((Ast'*XXt+XXt*Ast+Cst+pmderiv(XXt)).(ts)) < 1.e-6
@test norm(Xst-XXt) < 1.e-3 



@time Xt = pclyap(Asw, Csw; K = 10, reltol = 1.e-10, abstol=1.e-10);
@test norm((Asw*Xt+Xt*Asw'+Csw-pmderiv(Xt)).(ts)) < 1.e-6

@time Xt1 = pclyap(Asw, Csw; adj = true, K = 10, reltol = 1.e-10, abstol=1.e-10);
@test norm((Asw'*Xt1+Xt1*Asw+Csw+pmderiv(Xt1)).(ts)) < 1.e-6




# generate periodic function matrices
# A(t) = [0  1; -10*cos(t)-1 -24-19*sin(t)]
# X(t) = [1+cos(t) 0; 0 1+sin(t)]  # desired solution
# Xdot(t) = [-sin(t) 0; 0 cos(t)]  # pmderiv of the desired solution
# C(t) = [ -sin(t)  -1-sin(t)-(-1-10cos(t))*(1+cos(t));
# -1-sin(t)-(-1-10cos(t))*(1+cos(t))   cos(t)- 2(-24 - 19sin(t))*(1 + sin(t)) ]  # corresponding C
# Cd(t) = [ sin(t)  -1-cos(t)-(-1-10cos(t))*(1+sin(t));
# -1-cos(t)-(-1-10cos(t))*(1+sin(t))    -cos(t)-2(-24-19sin(t))*(1 + sin(t)) ] # corresponding Cd
At = PeriodicFunctionMatrix(A,2*pi)
Ct = PeriodicFunctionMatrix(C,2*pi)
Cdt = PeriodicFunctionMatrix(Cd,2*pi)
Xt = PeriodicFunctionMatrix(X,2*pi)
Xd = PeriodicFunctionMatrix(Xdot,2*pi)


for K in (1, 16, 64, 128, 256)
    @time Y = pclyap(At,Ct; K, reltol = 1.e-10);    
    tt = Vector((1:K)*2*pi/K) 
    println(" Acc = $(maximum(norm.(Y.f.(tt).-Xt.f.(tt)))) ")
    @test maximum(norm.(Y.f.(tt).-Xt.f.(tt))) < 1.e-7
end 

for K in (1, 16, 64, 128, 256)
    @time Y = pclyap(At,Cdt; K, adj = true, reltol = 1.e-10);   
    tt = Vector((1:K)*2*pi/K) 
    println(" Acc = $(maximum(norm.(Y.f.(tt).-Xt.f.(tt)))) ")
    @test maximum(norm.(Y.f.(tt).-Xt.f.(tt))) < 1.e-7
end 

for (na,nc) in ((1,1),(1,2),(2,1),(2,2))
   At = PeriodicFunctionMatrix(A,4*pi; nperiod = na)
   Ct = PeriodicFunctionMatrix(C,4*pi; nperiod = nc)
   Cdt = PeriodicFunctionMatrix(Cd,4*pi; nperiod = nc)
   Xt = PeriodicFunctionMatrix(X,4*pi; nperiod = gcd(na,nc))

   @time Y = pclyap(At,Ct,K = 500, reltol = 1.e-10);
   
   tt = Vector((1:500)*4*pi/500/Y.nperiod) 
   @test maximum(norm.(Y.f.(tt).-Xt.f.(tt))) < 1.e-7

   @time Y = pclyap(At,Cdt,K = 500, adj = true, reltol = 1.e-10);
   
   tt = Vector((1:500)*4*pi/500/Y.nperiod) 
   @test maximum(norm.(Y.f.(tt).-Xt.f.(tt))) < 1.e-7

end

# # generate symbolic periodic matrices
# using Symbolics
# @variables t
# As = [0  1; -1 -24]
# Xs =  [1+cos(t) 0; 0 1+sin(t)] 
# Xdots = [-sin(t) 0; 0 cos(t)] 
# Cds = -(As'*Xs+Xs*As+Xdots)
# Cs = Xdots - As*Xs-Xs*As'

A1 = [0  1; -1 -24]
X1(t) = [1+cos(t) 0; 0 1+sin(t)]  # desired solution
X1dot(t) = [-sin(t) 0; 0 cos(t)]  # pmderiv of the desired solution
C1(t) = [ -sin(t)   cos(t)-sin(t);
cos(t)-sin(t)  48+48sin(t)+cos(t) ]  # corresponding C
Cd1(t) = [ sin(t)   -cos(t)+sin(t);
-cos(t)+sin(t)  48+48sin(t)-cos(t)  ] # corresponding Cd


At = PeriodicFunctionMatrix(A1,2*pi)
Ct = PeriodicFunctionMatrix(C1,2*pi)
Cdt = PeriodicFunctionMatrix(Cd1,2*pi)
Xt = PeriodicFunctionMatrix(X1,2*pi)

@time Y = pclyap(At,Ct,K = 500, reltol = 1.e-10);

tt = Vector((1:500)*2*pi/500) 
@test maximum(norm.(Y.f.(tt).-Xt.f.(tt))) < 1.e-7

@time Y = pclyap(At,Cdt,K = 500, adj = true, reltol = 1.e-10)

tt = Vector((1:500)*2*pi/500) 
@test maximum(norm.(Y.f.(tt).-Xt.f.(tt))) < 1.e-7


A2 = [0  1; -1 -24]
X2 = [1 0; 0 1]  # desired solution
X2dot = [0 0; 0 0]  # pmderiv of the desired solution
C2 = [ 0   0; 0  48 ]  # corresponding C
Cd2 = [ 0   0; 0  48 ] # corresponding Cd

At = PeriodicFunctionMatrix(A2,2*pi)
Ct = PeriodicFunctionMatrix(C2,2*pi)
Cdt = PeriodicFunctionMatrix(Cd2,2*pi)
Xt = PeriodicFunctionMatrix(X2,2*pi)

@time Y = pclyap(At,Ct,K = 500, reltol = 1.e-10, intpol=false);

tt = Vector((1:500)*2*pi/500) 
@test maximum(norm.(Y.f.(tt).-Xt.f.(tt))) < 1.e-7  

@time Y = pclyap(At,Cdt,K = 500, adj = true, reltol = 1.e-10)

tt = Vector((1:500)*2*pi/500) 
@test maximum(norm.(Y.f.(tt).-Xt.f.(tt))) < 1.e-7


# Perturbed Pitelkau's example - singular Lyapunov equations
ω = 0.00103448
period = 2*pi/ω
β = 0.01; 
a = [  0            0     5.318064566757217e-02                         0
       0            0                         0      5.318064566757217e-02
       -1.352134256362805e-03   0             0    -7.099273035392090e-02
       0    -7.557182037479544e-04     3.781555288420663e-02             0
];
b = t->[0; 0; 0.1389735*10^-6*sin(ω*t); -0.3701336*10^-7*cos(ω*t)];
c = [1 0 0 0;0 1 0 0];
d = zeros(2,1);
#psysc = ps(a-β*I,PeriodicFunctionMatrix(b,period),c,d);
A = PeriodicFunctionMatrix(a-0.002I,period); B = PeriodicFunctionMatrix(b,period); C = PeriodicFunctionMatrix(c,period); D = PeriodicFunctionMatrix(d,period);

# observability Gramian
@time Qc = prclyap(A,transpose(C)*C,K = 120, abstol = 1.e-12, reltol = 1.e-12, intpol = true);
# controlability Gramian
@time Rc = pfclyap(A,B*transpose(B), K = 120, abstol = 1.e-10, reltol = 1.e-10, intpol = true);
@test norm(A'*Qc+Qc*A+C'*C + pmderiv(Qc)) < 1.e-7 &&
      norm(A*Rc+Rc*A'+B*B' - pmderiv(Rc)) < 1.e-7 

# # singular case
# psysc = ps(a,PeriodicFunctionMatrix(b,period),c,d);
# @time Qc = prclyap(psysc.A,transpose(psysc.C)*psysc.C,K = 120, reltol = 1.e-10);
# @time Rc = pfclyap(psysc.A,psysc.B*transpose(psysc.B), K = 120, reltol = 1.e-10);
# @test norm(Qc) > 1.e3 && norm(Rc) > 1.e2

# K = 120;
# @time psys = psc2d(psysc,period/K,reltol = 1.e-10);

# @time Qd = prdlyap(psys.A,transpose(psys.C)*psys.C);
# @time Rd = pfdlyap(psys.A, psys.B*transpose(psys.B));
# @test norm(Qd) > 1.e3 && norm(Rd) > 1.e3




A0(t) = [-cos(t)-1]
C0(t) = [ 1-sin(t)]  # corresponding C
At = PeriodicFunctionMatrix(A0,2*pi)
Ct = PeriodicFunctionMatrix(C0,2*pi)

# @time Yt = pclyap(At,Ct*Ct'; adj = false, K = 500, reltol = 1.e-12, abstol = 1.e-12);
@time Yt = pclyap(At,Ct*Ct'; adj = false, K = 100, intpol=false, reltol = 1.e-10, abstol = 1.e-10);

@time Ut = pcplyap(At,Ct; adj = false, K = 100, reltol = 1.e-10, abstol = 1.e-10);
@test norm(At*Yt+Yt*At'+Ct*Ct' - pmderiv(Yt)) < 1.e-6
@test norm(Yt-Ut*Ut') < 1.e-6

@time Yt = pclyap(At,Ct'*Ct; adj = true, K = 500, reltol = 1.e-12, abstol = 1.e-12);
@time Ut = pcplyap(At,Ct; adj = true, K = 100, reltol = 1.e-10, abstol = 1.e-10);
@test norm(At'*Yt+Yt*At+Ct'*Ct + pmderiv(Yt)) < 1.e-6
@test norm(Yt-Ut'*Ut) < 1.e-6


# generate periodic function matrices
A3(t) = [0  1; -1 -2-sin(t)]
#A(t) = [0  -1; 1 2+sin(t)]
# A(t) = [0  1; -1 -2]
# A(t) = [0  -1; 1 2]
B3(t) = [ 1-0.9*sin(t);  -1]  # corresponding B
C3(t) = [ 10-sin(t)  -1-sin(t)]  # corresponding C
At = PeriodicFunctionMatrix(A3,2*pi)
Bt = PeriodicFunctionMatrix(B3,2*pi)
Ct = PeriodicFunctionMatrix(C3,2*pi)

#@time Yt = pclyap(At,Bt*Bt'; adj = false, K = 500, reltol = 1.e-12, abstol = 1.e-12);
@time Yt = pclyap(At,Bt*Bt'; adj = false, K = 100, intpol = false, reltol = 1.e-10, abstol = 1.e-10);
@test norm(At*Yt+Yt*At'+Bt*Bt' - pmderiv(Yt)) < 1.e-6
@time Ut = pcplyap(At,Bt; adj = false, K = 100, reltol = 1.e-10, abstol = 1.e-10);
Xt = Ut*Ut'; 
@test norm(At*Xt+Xt*At'+Bt*Bt' - pmderiv(Xt)) < 1.e-6
@test norm(Yt-Xt) < 1.e-6
# @time Ut = pcplyap(At,Bt; adj = false, K = 100, implicit = true, reltol = 1.e-13, abstol = 1.e-13);
# Xt = Ut*Ut'; 
# @test norm(At*Xt+Xt*At'+Bt*Bt' - pmderiv(Xt)) < 1.e-6
# @test norm(Yt-Xt) < 1.e-6
# @time Ut = pcplyap(At,Bt; adj = false, K = 100, implicit = false, solver = "non-stiff", reltol = 1.e-7, abstol = 1.e-7);
# Xt = Ut*Ut'; 
# @test norm(At*Xt+Xt*At'+Bt*Bt' - pmderiv(Xt)) < 1.e-6
# @test norm(Yt-Xt) < 1.e-6


# @time Yt = pclyap(At,Ct'*Ct; adj = true, K = 500, reltol = 1.e-12, abstol = 1.e-12);
tt = Vector((1:20)*2*pi/20)
@time Yt = pclyap(At,Ct'*Ct; adj = true, K = 100, intpol = false, reltol = 1.e-10, abstol = 1.e-10);
@test norm((At'*Yt+Yt*At+Ct'*Ct + pmderiv(Yt)).(tt),Inf) < 1.e-5
@time Ut = pcplyap(At,Ct; adj = true, K = 100, reltol = 1.e-10, abstol = 1.e-10);
Xt = Ut'*Ut; 
@test norm((At'*Xt+Xt*At+Ct'*Ct + pmderiv(Xt)).(tt),Inf) < 1.e-4  # fails
@test norm((Yt-Xt).(tt)) < 1.e-6
# @time Ut = pcplyap(At,Ct; adj = true, K = 100, implicit = true, reltol = 1.e-13, abstol = 1.e-13);
# Xt = Ut'*Ut; 
# @test norm(At'*Xt+Xt*At+Ct'*Ct + pmderiv(Xt)) < 1.e-5
# @test norm(Yt-Xt) < 1.e-5
# @time Ut = pcplyap(At,Ct; adj = true, K = 100, implicit = false, solver = "non-stiff", reltol = 1.e-5, abstol = 1.e-5);
# Xt = Ut'*Ut; 
# @test norm(At'*Xt+Xt*At+Ct'*Ct + pmderiv(Xt)) < 1.e-6
# @test norm(Yt-Xt) < 1.e-6






# Ah = convert(HarmonicArray,At);
# Bh = convert(HarmonicArray,Bt);
# Ch = convert(HarmonicArray,Ct);
# @time Yh = pclyap(Ah,Bh*Bh'; adj = false, K = 200, reltol = 1.e-10, abstol = 1.e-10);
# @test norm(Ah*Yh+Yh*Ah'+Bh*Bh' - pmderiv(Yh)) < 1.e-6
# @time Uh = pcplyap(Ah,Bh; adj = false, K = 200, reltol = 1.e-10, abstol = 1.e-10);
# Xh = Uh*Uh';
# @test norm(Ah*Xh+Xh*Ah'+Bh*Bh' - pmderiv(Xh)) < 1.e-6
# @test norm(Yh-Uh*Uh') < 1.e-6
# @time Uh = pcplyap(Ah,Bh; adj = false, K = 200, implicit=true,reltol = 1.e-10, abstol = 1.e-10);
# Xh = Uh*Uh';
# @test norm(Ah*Xh+Xh*Ah'+Bh*Bh' - pmderiv(Xh)) < 1.e-6
# @test norm(Yh-Uh*Uh') < 1.e-6

# @time Yh = pclyap(Ah,Ch'*Ch; adj = true, K = 100, reltol = 1.e-10, abstol = 1.e-10);
# @test norm(Ah'*Yh+Yh*Ah+Ch'*Ch + pmderiv(Yh)) < 1.e-6
# @time Uh = pcplyap(Ah,Ch; adj = true, K = 100, reltol = 1.e-10, abstol = 1.e-10);
# Xh = Uh'*Uh; 
# @test norm(Ah'*Xh+Xh*Ah+Ch'*Ch + pmderiv(Xh)) < 1.e-6
# @test norm(Yh-Xh) < 1.e-6
# @time Uh = pcplyap(Ah,Ch; adj = true, K = 100, implicit = true, reltol = 1.e-13, abstol = 1.e-13);
# Xh = Uh'*Uh; 
# @test norm(Ah'*Xh+Xh*Ah+Ch'*Ch + pmderiv(Xh)) < 1.e-6
# @test norm(Yh-Xh) < 1.e-6




# A(t) = [-1;;]
# #A(t) = [0  -1; 1 2+sin(t)]
# # A(t) = [0  1; -1 -2]
# # A(t) = [0  -1; 1 2]
# B(t) = [ 0.5;;]  # corresponding B
# C(t) = [ 1.0;;]  # corresponding C
# At = PeriodicFunctionMatrix(A,2*pi)
# Bt = PeriodicFunctionMatrix(B,2*pi)
# Ct = PeriodicFunctionMatrix(C,2*pi)

# @time Yt = pclyap(At,Bt*Bt'; adj = false, K = 100, reltol = 1.e-10, abstol = 1.e-10);
# @test norm(At*Yt+Yt*At'+Bt*Bt' - pmderiv(Yt)) < 1.e-6
# @time Ut = pcplyap(At,Bt; adj = false, K = 100, reltol = 1.e-10, abstol = 1.e-10);
# Xt = Ut*Ut'; 
# @test norm(At*Xt+Xt*At'+Bt*Bt' - pmderiv(Xt)) < 1.e-6
# @test norm(Yt-Xt) < 1.e-6
# @time Ut = pcplyap(At,Bt; adj = false, K = 100, implicit = true, reltol = 1.e-13, abstol = 1.e-13);
# Xt = Ut*Ut'; 
# @test norm(At*Xt+Xt*At'+Bt*Bt' - pmderiv(Xt)) < 1.e-6
# @test norm(Yt-Xt) < 1.e-6
# @time Ut = pcplyap(At,Bt; adj = false, K = 100, implicit = false, reltol = 1.e-10, abstol = 1.e-10);
# Xt = Ut*Ut'; 
# @test norm(At*Xt+Xt*At'+Bt*Bt' - pmderiv(Xt)) < 1.e-6
# @test norm(Yt-Xt) < 1.e-6


end # psclyap

end
