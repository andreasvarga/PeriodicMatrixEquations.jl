module Test_pscsylv

using PeriodicMatrixEquations
using ApproxFun
using Symbolics
using Test
using LinearAlgebra


println("Test_pscsylv")

@testset "pcsylv" begin

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


@time Yt = pcsylv(As,As',Cs; K = 512, reltol = 1.e-10, abstol = 1.e-10,intpol=false);
@time Yt1 = pcsylv(As,As',Cs; K = 512, reltol = 1.e-10, abstol = 1.e-10,intpol=true);
Ys = convert(PeriodicSymbolicMatrix,Yt) 
Ys1 = convert(PeriodicSymbolicMatrix,Yt1) 
@test Ys ≈ Xs && Ys1 ≈ Xs

@time Yt = pcsylv(As',As,Cds; adj = true, K = 512, reltol = 1.e-10, abstol = 1.e-10,intpol=true); 
Ys = convert(PeriodicSymbolicMatrix,Yt) 
@test Xs ≈ Ys 

@time Yt = pfcsylv(As,As',Cs; K = 512, reltol = 1.e-10, abstol = 1.e-10,intpol=true); 
Ys = convert(PeriodicSymbolicMatrix,Yt) 
@test Ys ≈ Xs 

@time Yt = prcsylv(As',As,Cds; K = 512, reltol = 1.e-10, abstol = 1.e-10,intpol=false); 
Ys = convert(PeriodicSymbolicMatrix,Yt) 
@test Xs ≈ Ys 

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

@time Yt = pcsylv(At, At', Ct, K = 512, intpol = true, reltol = 1.e-12, abstol=1.e-12);
@time Yt1 = pcsylv(At,At', Ct; K = 512, reltol = 1.e-12, abstol = 1.e-12,intpol=false);
@test norm(Xt.(ts).-Yt.(ts),Inf) < 1.e-7 && norm(Xt.(ts).-Yt1.(ts),Inf) < 1.e-7  

@time Yt = pcsylv(Ac, Ac', Cc, K = 512, intpol = true, reltol = 1.e-12, abstol=1.e-12);
@test norm(Yt.(ts).*Ac'.(ts).+Ac.(ts).*Yt.(ts).+Cc.(ts),Inf) < 1.e-7

@time Yt = pcsylv(Ac', Ac, Cdc, adj = true, K = 512, intpol = true, reltol = 1.e-12, abstol=1.e-12);
@test norm(Yt.(ts).*Ac.(ts).+Ac'.(ts).*Yt.(ts).+Cdc.(ts),Inf) < 1.e-7

@time Yt = pcsylv(At, At', Ct(0), K = 512, intpol = true, reltol = 1.e-12, abstol=1.e-12);
@time Yt1 = pcsylv(At,At', Ct(0); K = 512, reltol = 1.e-12, abstol = 1.e-12,intpol=false);
@test norm(Yt.(ts).-Yt1.(ts),Inf)  < 1.e-5  


@time Yt1 = pcsylv(At, At', Ct, K = 1, reltol = 1.e-12, abstol=1.e-12, intpol = false);
@test Xt ≈ Yt1 && Xd ≈ pmderiv(Yt1) 

@time Yt1 = pcsylv(At', At, Cdt, K = 1, adj = true, reltol = 1.e-12, abstol=1.e-12, intpol = false)
@test Xt ≈ Yt1 && Xd ≈ pmderiv(Yt1) 

@time Yt = pfcsylv(At, At', Ct, K = 500, reltol = 1.e-12, abstol=1.e-12);
@test Xt ≈ Yt 

@time Yt = pfcsylv(At, At', Ct(0), K = 500, intpol = true, reltol = 1.e-12, abstol=1.e-12);
@time Yt1 = pfcsylv(At, At', Ct(0), K = 500, intpol = false, reltol = 1.e-12, abstol=1.e-12);
@test norm(Yt1.(ts) .- Yt.(ts),Inf) < 1.e-5 

@time Yt = prcsylv(At', At, Cdt, K = 500, reltol = 1.e-12, abstol=1.e-12)
@test Xt ≈ Yt 

@time Yt = prcsylv(At', At, Cdt(0), K = 500, intpol = true, reltol = 1.e-12, abstol=1.e-12);
@time Yt1 = prcsylv(At', At, Cdt(0), K = 500, intpol = false, reltol = 1.e-12, abstol=1.e-12);
@test norm(Yt1.(ts) .- Yt.(ts),Inf) < 1.e-5 


K = 500
@time W0 = pgcsylv(At, At', Ct, K; adj = false, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
Ts = 2pi/K
success = true
for i = 1:K
    Y  = PeriodicMatrixEquations.tvcsylv(At, At',Ct, i*Ts, (i-1)*Ts, W0.values[mod(i+K-1,K)+1]; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10) 
    iw = i+1; iw > K && (iw = 1)
    success = success && norm(Y-W0.values[iw]) < 1.e-7
end
@test success



K = 10
@time W0 = pgcsylv(At, At',Ct, K; adj = false, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
ns = 10; ts = sort([0.; rand(ns-1)*2*pi; 2*pi])
success = true
for t in ts
    Xtval = PeriodicMatrixEquations.tvcsylv_eval(t, W0, At, At',Ct; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10) 
    success = success && norm(Xtval-Xt(t)) < 1.e-7
end
@test success
@time XXt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvcsylv_eval(t, W0, At, At', Ct; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((At*XXt+XXt*At'+Ct-pmderiv(XXt)).(ts),Inf) < 1.e-5   



K = 100
solver = "non-stiff"
for solver in ("non-stiff", "stiff", "auto")
    println("solver = $solver")
    @time W0 = pgcsylv(At, At',Ct, K; adj = false, solver, reltol = 1.e-10, abstol = 1.e-10);
    Y  = PeriodicMatrixEquations.tvcsylv(At, At', Ct, ts, W0.values[1]; solver, adj = false, reltol = 1.e-10, abstol = 1.e-10) 
    Xtval = [PeriodicMatrixEquations.tvcsylv_eval(t, W0, At, At', Ct; solver, adj = false, reltol = 1.e-10, abstol = 1.e-10)  for t in ts]
    @test norm(Y-Xtval) < 1.e-5
end

K= 100
solver = "non-stiff"
for solver in ("non-stiff", "stiff", "auto")
    println("solver = $solver")
    @time W0 = pgcsylv(At', At, Cdt, K; adj = true, solver, reltol = 1.e-10, abstol = 1.e-10);
    Xtval = [PeriodicMatrixEquations.tvcsylv_eval(t, W0, At', At, Cdt; solver, adj = true, reltol = 1.e-10, abstol = 1.e-10)  for t in 2*ts]
    Y  = PeriodicMatrixEquations.tvcsylv(At', At, Cdt, 2*ts, W0.values[1]; solver, adj = true, reltol = 1.e-10, abstol = 1.e-10) 
    @test norm(Y-Xtval) < 1.e-5
end



K = 100
solver = "non-stiff"
for solver in ("non-stiff", "stiff", "auto")
    println("solver = $solver")
    @time W0 = pgcsylv(At, At', Ct, K; adj = false, solver, reltol = 1.e-10, abstol = 1.e-10);
    Y  = PeriodicMatrixEquations.tvcsylv(At, At', Ct, W0.values[1]; solver, adj = false, reltol = 1.e-10, abstol = 1.e-10); 
    Xtval = [PeriodicMatrixEquations.tvcsylv_eval(t, W0, At, At', Ct; solver, adj = false, reltol = 1.e-10, abstol = 1.e-10)  for t in ts]
    @test norm(Y.(ts)-Xtval) < 1.e-5
end

K = 100
solver = "non-stiff"
for solver in ("non-stiff", "stiff", "auto")
    println("solver = $solver")
    @time W0 = pgcsylv(At, At', Ct, K; adj = false, solver, reltol = 1.e-10, abstol = 1.e-10);
    Y  = PeriodicMatrixEquations.tvcsylv(At, At', Ct, W0; solver, adj = false, reltol = 1.e-10, abstol = 1.e-10); 
    Xtval = [PeriodicMatrixEquations.tvcsylv_eval(t, W0, At, At', Ct; solver, adj = false, reltol = 1.e-10, abstol = 1.e-10)  for t in ts]
    @test norm(Y.(ts)-Xtval) < 1.e-5
end


@time W0 = pgcsylv(At, At', Ct, K; adj = false, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
Y  = PeriodicMatrixEquations.tvcsylv(At, At', Ct, ts, W0.values[1]; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10) 
Xtval = [PeriodicMatrixEquations.tvcsylv_eval(t, W0, At, At', Ct; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10)  for t in ts]
@test norm(Y-Xtval) < 1.e-7

success = true
for t in ts
    Xtval = PeriodicMatrixEquations.tvcsylv_eval(t, W0, At, At', Ct; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10) 
    success = success && norm(Xtval-Xt(t)) < 1.e-7
end
@test success
@time XXt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvcsylv_eval(t, W0, At, At', Ct; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((At*XXt+XXt*At'+Ct-pmderiv(XXt)).(ts),Inf) < 1.e-5   


K = 500
W0 = pgcsylv(At', At,  Cdt, K; adj = true, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
Ts = 2pi/K
success = true
for i = K:-1:1
    iw = i+1; iw > K && (iw = 1)
    Y  = PeriodicMatrixEquations.tvcsylv(At', At, Cdt, (i-1)*Ts, i*Ts, W0.values[iw]; solver = "auto", adj = true, reltol = 1.e-10, abstol = 1.e-10) 
    success = success && norm(Y-W0.values[i]) < 1.e-7
end
@test success

K = 10
W0 = pgcsylv(At', At, Cdt, K; adj = true, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
ns = 10; ts = sort([0.; rand(ns-1)*2*pi; 2*pi])
success = true
for t in ts
    Xtval = PeriodicMatrixEquations.tvcsylv_eval(t, W0, At', At, Cdt; solver = "auto", adj = true, reltol = 1.e-10, abstol = 1.e-10) 
    success = success && norm(Xtval-Xt(t)) < 1.e-7
end
@test success

XXt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvcsylv_eval(t, W0, At', At, Cdt; solver = "auto", adj = true, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((At'*XXt+XXt*At+Cdt+pmderiv(XXt)).(ts),Inf) < 1.e-5  


solver = "non-stiff"
for solver in ("non-stiff", "stiff", "auto")
    println("solver = $solver")
    @time Yt = pcsylv(At, At', Ct; solver, K = 500, reltol = 1.e-10, abstol = 1.e-10,intpol=true);
    #@test Xt ≈ Yt 
    @test norm(Xt.(ts).-Yt.(ts),Inf) < 1.e-6
    @time Yt = pcsylv(At, At', Ct; solver, K = 500, reltol = 1.e-10, abstol = 1.e-10,intpol=false);
    #@test Xt ≈ Yt 
    @test norm(Xt.(ts).-Yt.(ts),Inf) < 1.e-6

end


# solve using harmonic arrays
Ah = convert(HarmonicArray,At);
Ch = convert(HarmonicArray,Ct);
Cdh = convert(HarmonicArray,Cdt);
Xh = convert(HarmonicArray,Xt);
Xdh = convert(HarmonicArray,Xd);

@time Yt = pcsylv(Ah, Ah', Ch, K = 500, reltol = 1.e-10, abstol = 1.e-10);
@time Yt1 =pcsylv(Ah, Ah', Ch; K = 512, reltol = 1.e-10, abstol = 1.e-10,intpol=true);
Yh = convert(HarmonicArray,Yt);
Yh1 = convert(HarmonicArray,Yt1);
@test Xh ≈ Yh && Xh ≈ Yh1

@time Yt = pcsylv(Ah', Ah, Cdh, K = 500, adj = true, reltol = 1.e-10, abstol = 1.e-10);
Yh = convert(HarmonicArray,Yt);
@test Xh ≈ Yh && Xdh ≈ pmderiv(Yh) 

@time Yt = pfcsylv(Ah, Ah', Ch, K = 500, reltol = 1.e-10, abstol = 1.e-10);
Yh = convert(HarmonicArray,Yt);
@test Xh ≈ Yh 

@time Yt = prcsylv(Ah', Ah, Cdh, K = 500, reltol = 1.e-10, abstol = 1.e-10);
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


@time Yt = pcsylv(Af, Af', Cf, K = 500, reltol = 1.e-10, abstol = 1.e-10, intpol=false);
@time Yt1 = pcsylv(Af, Af', Cf; K = 512, reltol = 1.e-10, abstol = 1.e-10,intpol=true);
Yf = convert(FourierFunctionMatrix,Yt);
@test Xf ≈ Yf && Xdf ≈ pmderiv(Yf) && 
      norm(Af*Yf+Yf*Af'+Cf-pmderiv(Yf)) < 1.e-7 && norm(Yt1.(ts).-Yt.(ts),Inf) < 1.e-7

@time Yt = pcsylv(Af', Af, Cdf, K = 500, adj = true, reltol = 1.e-10, abstol = 1.e-10)
Yf = convert(FourierFunctionMatrix,Yt);
@test Xf ≈ Yf && Xdf ≈ pmderiv(Yf) && norm(Af'*Yf+Yf*Af+Cdf+pmderiv(Yf)) < 1.e-7

@time Yt = pfcsylv(Af, Af', Cf, K = 500, reltol = 1.e-10, abstol = 1.e-10);
Yf = convert(FourierFunctionMatrix,Yt);
@test Xf ≈ Yf && norm(Af*Yf+Yf*Af'+Cf-pmderiv(Yf)) < 1.e-7

@time Yt = prcsylv(Af', Af, Cdf, K = 500, reltol = 1.e-10, abstol = 1.e-10)
Yf = convert(FourierFunctionMatrix,Yt);
@test Xf ≈ Yf && norm(Af'*Yf+Yf*Af+Cdf+pmderiv(Yf)) < 1.e-7


K = 500
@time W0 = pgcsylv(Af, Af', Cf, K; adj = false, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
Ts = 2pi/K
success = true
for i = 1:K
    Y  = PeriodicMatrixEquations.tvcsylv(Af, Af', Cf, i*Ts, (i-1)*Ts, W0.values[mod(i+K-1,K)+1]; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10) 
    iw = i+1; iw > K && (iw = 1)
    success = success && norm(Y-W0.values[iw]) < 1.e-7
end
@test success

K = 100
solver = "non-stiff"
for solver in ("non-stiff", "stiff", "auto")
    println("solver = $solver")
    @time W0 = pgcsylv(Af, Af', Cf, K; adj = false, solver, reltol = 1.e-10, abstol = 1.e-10);
    Y  = PeriodicMatrixEquations.tvcsylv(Af, Af', Cf, ts, W0.values[1]; solver, adj = false, reltol = 1.e-10, abstol = 1.e-10) 
    Xtval = [PeriodicMatrixEquations.tvcsylv_eval(t, W0, Af, Af', Cf; solver, adj = false, reltol = 1.e-10, abstol = 1.e-10)  for t in ts]
    @test norm(Y-Xtval) < 1.e-5
end

K = 100
solver = "non-stiff"
for solver in ("non-stiff", "stiff", "auto")
    println("solver = $solver")
    @time W0 = pgcsylv(Af', Af, Cdf, K; adj = true, solver, reltol = 1.e-10, abstol = 1.e-10);
    Xtval = [PeriodicMatrixEquations.tvcsylv_eval(t, W0, Af', Af, Cdf; solver, adj = true, reltol = 1.e-10, abstol = 1.e-10)  for t in 2*ts]
    Y  = PeriodicMatrixEquations.tvcsylv(Af', Af, Cdf, 2*ts, W0.values[1]; solver, adj = true, reltol = 1.e-14, abstol = 1.e-14) 
    @test norm(Y-Xtval) < 1.e-5
end


K = 100
solver = "non-stiff"
for solver in ("non-stiff", "stiff", "auto")
    println("solver = $solver")
    @time W0 = pgcsylv(Af, Af', Cf, K; adj = false, solver, reltol = 1.e-10, abstol = 1.e-10);
    Y  = PeriodicMatrixEquations.tvcsylv(Af, Af', Cf, W0.values[1]; solver, adj = false, reltol = 1.e-10, abstol = 1.e-10); 
    Xtval = [PeriodicMatrixEquations.tvcsylv_eval(t, W0, Af, Af', Cf; solver, adj = false, reltol = 1.e-10, abstol = 1.e-10)  for t in ts]
    @test norm(Y.(ts)-Xtval) < 1.e-5 
end

K = 100
solver = "non-stiff"
for solver in ("non-stiff", "stiff", "auto")
    println("solver = $solver")
    @time W0 = pgcsylv(Af, Af', Cf, K; adj = false, solver, reltol = 1.e-10, abstol = 1.e-10);
    Y  = PeriodicMatrixEquations.tvcsylv(Af, Af', Cf, W0; solver, adj = false, reltol = 1.e-10, abstol = 1.e-10); 
    Xtval = [PeriodicMatrixEquations.tvcsylv_eval(t, W0, Af, Af', Cf; solver, adj = false, reltol = 1.e-10, abstol = 1.e-10)  for t in ts]
    @test norm(Y.(ts)-Xtval) < 1.e-5 
end

@time Yt = pcsylv(Afc, Afc', Cfc, K = 512, intpol = true, reltol = 1.e-12, abstol=1.e-12);
@test norm(Yt.(ts).*Afc'.(ts).+Afc.(ts).*Yt.(ts).+Cfc.(ts),Inf) < 1.e-7


# solve using periodic time-series matrices
K = 512
Ats = convert(PeriodicTimeSeriesMatrix,At;ns = K);
Cts = convert(PeriodicTimeSeriesMatrix,Ct;ns = K);
Cdts = convert(PeriodicTimeSeriesMatrix,Cdt;ns = K);
Atsc = convert(PeriodicTimeSeriesMatrix,Ac;ns = K);
Ctsc = convert(PeriodicTimeSeriesMatrix,Cc;ns = K);

ts = [0.4459591888577492, 1.2072325802972004, 1.9910835248218244, 2.1998199838900527, 2.4360161318589695, 
     2.9004720268745463, 2.934294124172935, 4.149208861412936, 4.260935465730602, 5.956614157549958]


@time Yt1 = pcsylv(Ats, Ats', Cts; K = 1, reltol = 1.e-14, abstol = 1.e-14,intpol=false); 
@test norm((Ats.(ts).*Yt1.(ts)).+(Yt1.(ts).*(Ats').(ts)).+Cts.(ts).-pmderiv(Yt1).(ts),Inf) < 1.e-5 
#@test norm((Ats*Yt1+Yt1*Ats'+Cts-pmderiv(Yt1)).(ts)) < 1.e-5 

@time Yt = pcsylv(Ats,Ats', Cts; K, reltol = 1.e-14, abstol = 1.e-14,intpol=true);
@test norm((Ats*Yt+Yt*Ats'+Cts-pmderiv(Yt)).(ts)) < 1.e-5 

Xt1 = pcsylv(Ats, Ats', Cts; K = 128, reltol = 1.e-14, abstol = 1.e-14,intpol=true);
@test norm((Ats*Xt1+Xt1*Ats'+Cts-pmderiv(Xt1)).(ts)) < 1.e-5 

Xt2 = pcsylv(Ats, Ats', Cts; K = 256, reltol = 1.e-14, abstol = 1.e-14,intpol=true);
@test norm((Ats*Xt2+Xt2*Ats'+Cts-pmderiv(Xt2)).(ts)) < 1.e-5 
@test norm((Xt1-Xt2).(ts)) < 1.e-7

Ka = 10; Kc = 1; 
Ats1 = convert(PeriodicTimeSeriesMatrix,At;ns = Ka);
Cts1 = convert(PeriodicTimeSeriesMatrix,Ct;ns = Kc);
Xt1 = pcsylv(Ats1, Ats1', Cts1; K = 10, reltol = 1.e-14, abstol = 1.e-14);
@test norm((Ats1*Xt1+Xt1*Ats1'+Cts1-pmderiv(Xt1)).(ts)) < 1.e-5 

Xt2 = pcsylv(Ats1, Ats1', Cts1; K = 2000, reltol = 1.e-14, abstol = 1.e-14);
@test norm((Ats1*Xt2+Xt2*Ats1'+Cts1-pmderiv(Xt2)).(ts)) < 1.e-5 

K = 512
W0 = pgcsylv(Ats, Ats', Cts, K; adj = false, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
Xts = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvcsylv_eval(t, W0, Ats, Ats', Cts; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((Ats*Xts+Xts*Ats'+Cts-pmderiv(Xts)).(ts)) < 1.e-6

W0 = pgcsylv(Ats', Ats, Cts, K; adj = true, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
Xts = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvcsylv_eval(t, W0, Ats', Ats, Cts; solver = "auto", adj =true, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((Ats'*Xts+Xts*Ats+Cts+pmderiv(Xts)).(ts)) < 1.e-6

Cts2=set_period(Cts,4pi)
W0 = pgcsylv(Ats, Ats', Cts2, K; adj = false, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
Xts = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvcsylv_eval(t, W0, Ats, Ats', Cts2; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10),4*pi)
@test norm((Ats*Xts+Xts*Ats'+Cts2-pmderiv(Xts)).(ts)) < 1.e-6


W0 = pgcsylv(Atsc, Atsc', Ctsc, K; adj = false, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
Xts = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvcsylv_eval(t, W0, Atsc, Atsc', Ctsc; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((Atsc*Xts+Xts*Atsc'+Ctsc-pmderiv(Xts)).(ts)) < 1.e-6


# solve using periodic switching matrices
Asw = convert(PeriodicSwitchingMatrix,Ats)
Csw = convert(PeriodicSwitchingMatrix,Cts)
Cdsw = convert(PeriodicSwitchingMatrix,Cdts)
Aswc = convert(PeriodicSwitchingMatrix,Ac)
Cswc = convert(PeriodicSwitchingMatrix,Cc)
Cdswc = convert(PeriodicSwitchingMatrix,Cdc)

ts = [0.4459591888577492, 1.2072325802972004, 1.9910835248218244, 2.1998199838900527, 2.4360161318589695, 
     2.9004720268745463, 2.934294124172935, 4.149208861412936, 4.260935465730602, 5.956614157549958]

@time Yt = pcsylv(Asw, Asw', Csw; K = 100, reltol = 1.e-14, abstol = 1.e-14);
@test norm((Asw.(ts).*Yt.(ts).+Yt.(ts).*Asw'.(ts).+Csw.(ts).-pmderiv(Yt).(ts)),Inf) < 1.e-5 

@time Yt = pcsylv(Aswc, Aswc', Cswc; K = 100, reltol = 1.e-14, abstol = 1.e-14);
@test norm((Aswc.(ts).*Yt.(ts).+Yt.(ts).*Aswc'.(ts).+Cswc.(ts).-pmderiv(Yt).(ts)),Inf) < 1.e-5 

@time Yt = pfcsylv(Asw, Asw', Csw; K = 100, reltol = 1.e-14, abstol = 1.e-14);
@test norm((Asw.(ts).*Yt.(ts).+Yt.(ts).*Asw'.(ts).+Csw.(ts).-pmderiv(Yt).(ts)),Inf) < 1.e-5 

@time Yt = pfcsylv(Asw, Asw', Csw(0); K = 200, reltol = 1.e-14, abstol = 1.e-14);
@test norm((Asw.(ts).*Yt.(ts).+Yt.(ts).*Asw'.(ts).+Csw.(0).-pmderiv(Yt).(ts)),Inf) < 1.e-5 

@time Yt = prcsylv(Asw', Asw, Cdsw; K = 100, reltol = 1.e-14, abstol = 1.e-14);
@test norm((Asw'.(ts).*Yt.(ts).+Yt.(ts).*Asw.(ts).+Cdsw.(ts).+pmderiv(Yt).(ts)),Inf) < 1.e-5 

@time Yt = prcsylv(Asw', Asw, Cdsw(0); K = 200, reltol = 1.e-14, abstol = 1.e-14);
@test norm((Asw'.(ts).*Yt.(ts).+Yt.(ts).*Asw.(ts).+Cdsw.(0).+pmderiv(Yt).(ts)),Inf) < 1.e-4 


####

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
W0 = pgcsylv(Ast, Ast', Cst, K; adj = false, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
Ts = 2pi/K
success = true
for i = 1:K
    Y  = PeriodicMatrixEquations.tvcsylv(Ast, Ast', Cst, i*Ts, (i-1)*Ts, W0.values[mod(i+K-1,K)+1]; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10) 
    iw = i+1; iw > K && (iw = 1)
    success = success && norm(Y-W0.values[iw]) < 1.e-2
end
@test success
Xst = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvcsylv_eval(t, W0, Ast, Ast', Cst; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((Ast*Xst+Xst*Ast'+Cst-pmderiv(Xst)).(ts)) < 1.e-6


K = 100;
#K = 1
W1 = pgcsylv(Asw, Asw', Csw, K; adj = false, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
W2 = pgcsylv(Asw, Asw', Csw, 1; adj = false, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
success = true
Ks = length(W1.ts)
for i = 1:Ks
    tf = i == Ks ? W1.period/W1.nperiod : W1.ts[i+1]
    Y  = PeriodicMatrixEquations.tvcsylv(Asw, Asw', Csw, tf, W1.ts[i], W1.values[mod(i+Ks-1,Ks)+1]; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10) 
    iw = i+1; iw > Ks && (iw = 1)
    success = success && norm(Y-W1.values[iw]) < 1.e-3
end
@test success

XXt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvcsylv_eval(t, W1, Asw, Asw', Csw; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((Ast*XXt+XXt*Ast'+Cst-pmderiv(XXt)).(rand(10)*2*pi)) < 1.e-6
@test norm((Xst-XXt).(ts)) < 1.e-3
XXt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvcsylv_eval(t, W2, Asw, Asw', Csw; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((Ast*XXt+XXt*Ast'+Cst-pmderiv(XXt)).(rand(10)*2*pi)) < 1.e-6
@test norm((Xst-XXt).(ts)) < 1.e-3

W1 = pgcsylv(Aswc, Aswc', Cswc, K; adj = false, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
success = true
Ks = length(W1.ts)
for i = 1:Ks
    tf = i == Ks ? W1.period/W1.nperiod : W1.ts[i+1]
    Y  = PeriodicMatrixEquations.tvcsylv(Aswc, Aswc', Cswc, tf, W1.ts[i], W1.values[mod(i+Ks-1,Ks)+1]; solver = "auto", adj = false, reltol = 1.e-10, abstol = 1.e-10) 
    iw = i+1; iw > Ks && (iw = 1)
    success = success && norm(Y-W1.values[iw]) < 1.e-3
end
@test success


K = 500
@time W0 = pgcsylv(Ast, Ast', Cst, K; adj = true, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
Xst = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvcsylv_eval(t, W0, Ast', Ast, Cst; solver = "auto", adj = true, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((Ast'*Xst+Xst*Ast+Cst+pmderiv(Xst)).(ts)) < 1.e-6

K = 100;
#K = 1
W1 = pgcsylv(Asw', Asw, Csw, K; adj = true, solver = "auto", reltol = 1.e-10, abstol = 1.e-10);
Ks = length(W1.ts)
success = true
for i = Ks:-1:1
    iw = i+1; iw > Ks && (iw = 1)
    t0 = i == Ks ? W1.period/W1.nperiod : W1.ts[i+1]
    Y  = PeriodicMatrixEquations.tvcsylv(Asw', Asw, Csw, W1.ts[i], t0, W1.values[iw]; solver = "auto", adj = true, reltol = 1.e-10, abstol = 1.e-10) 
    success = success && norm(Y-W1.values[i]) < 1.e-7
end
@test success

XXt = PeriodicFunctionMatrix(t->PeriodicMatrixEquations.tvcsylv_eval(t, W1, Asw', Asw, Csw; solver = "auto", adj = true, reltol = 1.e-10, abstol = 1.e-10),2*pi)
@test norm((Ast'*XXt+XXt*Ast+Cst+pmderiv(XXt)).(ts)) < 1.e-6
#@test norm(Xst-XXt) < 1.e-3 



@time Xt = pcsylv(Asw, Asw', Csw; K = 10, reltol = 1.e-10, abstol=1.e-10);
@test norm((Asw*Xt+Xt*Asw'+Csw-pmderiv(Xt)).(ts)) < 1.e-6

@time Xt1 = pcsylv(Asw', Asw, Csw; adj = true, K = 10, reltol = 1.e-10, abstol=1.e-10);
@test norm((Asw'*Xt1+Xt1*Asw+Csw+pmderiv(Xt1)).(ts)) < 1.e-6




At = PeriodicFunctionMatrix(A,2*pi)
Ct = PeriodicFunctionMatrix(C,2*pi)
Cdt = PeriodicFunctionMatrix(Cd,2*pi)
Xt = PeriodicFunctionMatrix(X,2*pi)
Xd = PeriodicFunctionMatrix(Xdot,2*pi)


for K in (1, 16, 64, 128, 256)
    @time Y = pcsylv(At, At', Ct; K, reltol = 1.e-10);    
    tt = Vector((1:K)*2*pi/K) 
    println(" Acc = $(maximum(norm.(Y.f.(tt).-Xt.f.(tt)))) ")
    @test maximum(norm.(Y.f.(tt).-Xt.f.(tt))) < 1.e-7
end 

for K in (1, 16, 64, 128, 256)
    @time Y = pcsylv(At', At,Cdt; K, adj = true, reltol = 1.e-10);   
    tt = Vector((1:K)*2*pi/K) 
    println(" Acc = $(maximum(norm.(Y.f.(tt).-Xt.f.(tt)))) ")
    @test maximum(norm.(Y.f.(tt).-Xt.f.(tt))) < 1.e-7
end 

for (na,nc) in ((1,1),(1,2),(2,1),(2,2))
   At = PeriodicFunctionMatrix(A,4*pi; nperiod = na)
   Ct = PeriodicFunctionMatrix(C,4*pi; nperiod = nc)
   Cdt = PeriodicFunctionMatrix(Cd,4*pi; nperiod = nc)
   Xt = PeriodicFunctionMatrix(X,4*pi; nperiod = gcd(na,nc))

   @time Y = pcsylv(At,At', Ct,K = 500, reltol = 1.e-10);
   
   tt = Vector((1:500)*4*pi/500/Y.nperiod) 
   @test maximum(norm.(Y.f.(tt).-Xt.f.(tt))) < 1.e-7

   @time Y = pcsylv(At', At,Cdt,K = 500, adj = true, reltol = 1.e-10);
   
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
X2(t) = [1+cos(t) 0; 0 1+sin(t)]  # desired solution
X1dot(t) = [-sin(t) 0; 0 cos(t)]  # pmderiv of the desired solution
C1(t) = [ -sin(t)   cos(t)-sin(t);
cos(t)-sin(t)  48+48sin(t)+cos(t) ]  # corresponding C
Cd1(t) = [ sin(t)   -cos(t)+sin(t);
-cos(t)+sin(t)  48+48sin(t)-cos(t)  ] # corresponding Cd


At = PeriodicFunctionMatrix(A1,2*pi)
Ct = PeriodicFunctionMatrix(C1,2*pi)
Cdt = PeriodicFunctionMatrix(Cd1,2*pi)
Xt = PeriodicFunctionMatrix(X2,2*pi)

@time Y = pcsylv(At,At',Ct,K = 500, reltol = 1.e-10);

tt = Vector((1:500)*2*pi/500) 
@test maximum(norm.(Y.f.(tt).-Xt.f.(tt))) < 1.e-7

@time Y = pcsylv(At', At,Cdt,K = 500, adj = true, reltol = 1.e-10)

tt = Vector((1:500)*2*pi/500) 
@test maximum(norm.(Y.f.(tt).-Xt.f.(tt))) < 1.e-7


A2 = [0  1; -1 -24]
X22 = [1 0; 0 1]  # desired solution
X2dot = [0 0; 0 0]  # pmderiv of the desired solution
C2 = [ 0   0; 0  48 ]  # corresponding C
Cd2 = [ 0   0; 0  48 ] # corresponding Cd

At = PeriodicFunctionMatrix(A2,2*pi)
Ct = PeriodicFunctionMatrix(C2,2*pi)
Cdt = PeriodicFunctionMatrix(Cd2,2*pi)
Xt = PeriodicFunctionMatrix(X22,2*pi)

@time Y = pcsylv(At,At',Ct,K = 500, reltol = 1.e-10, intpol=false);

tt = Vector((1:500)*2*pi/500) 
@test maximum(norm.(Y.f.(tt).-Xt.f.(tt))) < 1.e-7  

@time Y = pcsylv(At',At,Cdt,K = 500, adj = true, reltol = 1.e-10)

tt = Vector((1:500)*2*pi/500) 
@test maximum(norm.(Y.f.(tt).-Xt.f.(tt))) < 1.e-7


# solve a periodic pole assignment problem using reverse-time Sylvester differential equation
# Pitelkau's example 
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
Ap = PeriodicFunctionMatrix(a,period); Bp = PeriodicFunctionMatrix(b,period); Cp = PeriodicFunctionMatrix(c,period); 
Acl = PeriodicFunctionMatrix(Ap-Matrix(0.01*I(4)),period); G = 100*rand(1,4); 

# transformation matrix
@time P = prcsylv(-Ap,Acl,-Bp*G,K = 120, abstol = 1.e-12, reltol = 1.e-12, intpol = true);
F = G*inv(P); 
@test all(abs.(real(psceig(Ap+Bp*F,200)) .+ 0.01) .< 1.e-5)
@test norm(-Ap*P+P*Acl-Bp*G+pmderiv(P)) < 1.e-7  

# forward-time formulation produce inaccurate result
@time P1 = pfcsylv(Ap,-Acl,Bp*G,K = 120, abstol = 1.e-12, reltol = 1.e-12, intpol = true);
@test norm(P1) > 1.e7



A0(t) = [-cos(t)-1]
C0(t) = [ 1-sin(t)]  # corresponding C
At = PeriodicFunctionMatrix(A0,2*pi)
Ct = PeriodicFunctionMatrix(C0,2*pi)

@time Yt = pcsylv(At,At',Ct*Ct'; adj = false, K = 100, intpol=false, reltol = 1.e-10, abstol = 1.e-10);
@time Yt1 = pcsylv(-At,-At',-Ct*Ct'; adj = true, K = 100, intpol=false, reltol = 1.e-10, abstol = 1.e-10);
@test norm(Yt-Yt1) < 1.e-10


# generate periodic function matrices
A3(t) = [0  1; -1 -2-sin(t)]
B3(t) = [ 1-0.9*sin(t);  -1]  # corresponding B
C3(t) = [ 10-sin(t)  -1-sin(t)]  # corresponding C
At = PeriodicFunctionMatrix(A3,2*pi)
Bt = PeriodicFunctionMatrix(B3,2*pi)
Ct = PeriodicFunctionMatrix(C3,2*pi)

@time Yt = pcsylv(At,At',Bt*Bt'; adj = false, K = 100, intpol = false, reltol = 1.e-10, abstol = 1.e-10);
@time Yt1 = pcsylv(-At,-At',-Bt*Bt'; adj = true, K = 100, intpol = false, reltol = 1.e-10, abstol = 1.e-10);
@test norm(Yt-Yt1) < 1.e-10

@time Yt = pcsylv(At',At,Ct'*Ct; adj = false, K = 100, intpol=false, reltol = 1.e-10, abstol = 1.e-10);
@time Yt1 = pcsylv(-At',-At,-Ct'*Ct; adj = true, K = 100, intpol=false, reltol = 1.e-10, abstol = 1.e-10);
@test norm(Yt-Yt1) < 1.e-10

# compute periodic stationary solution for step input
ZM = PeriodicFunctionMatrix(zeros(1,1),2*pi)
@time Xt = pcsylv(At,ZM,Bt; adj = false, K = 100, intpol = false, reltol = 1.e-10, abstol = 1.e-10);
@time Xt1 = pcsylv(-At,ZM,-Bt; adj = true, K = 100, intpol = false, reltol = 1.e-10, abstol = 1.e-10);
@test norm(Xt-Xt1) < 1.e-10
@test norm(pmderiv(Xt)-At*Xt-Bt) < 1.e-7

# no periodic solution (see Brocket, p. 53) 
A3(t) = [0  1; -1 0]
B3(t) = [ 0; 1-0.9*sin(t)]  # B has first order harmonic
C3(t) = [ 1 1]  # corresponding C
At = PeriodicFunctionMatrix(A3,2*pi)
Bt = PeriodicFunctionMatrix(B3,2*pi)
Ct = PeriodicFunctionMatrix(C3,2*pi)

@time Xt = pcsylv(At,ZM,Bt; adj = false, K = 100, intpol = false, reltol = 1.e-10, abstol = 1.e-10);
@time Xt1 = pcsylv(-At,ZM,-Bt; adj = true, K = 100, intpol = false, reltol = 1.e-10, abstol = 1.e-10);
@test norm(Xt) > 1.e10 && norm(Xt1) > 1.e10

B3(t) = [ 0; -0.9*sin(2t)]  # B has no first order harmonic
Bt = PeriodicFunctionMatrix(B3,2pi)
@time Xt = pcsylv(At,ZM,Bt; adj = false, K = 100, intpol = false, reltol = 1.e-10, abstol = 1.e-10);
@test norm(pmderiv(Xt)-At*Xt-Bt) < 1.e-7
@time Xt1 = pcsylv(-At,ZM,-Bt; adj = true, K = 100, intpol = false, reltol = 1.e-10, abstol = 1.e-10);
@test norm(pmderiv(Xt1)-At*Xt1-Bt) < 1.e-7
@test norm(Xt-Xt1) > 1.e-1 # periodic solution is not uniques

end # pscsylv

end
