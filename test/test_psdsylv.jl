module Test_psdsylv

using PeriodicMatrices
using PeriodicMatrixEquations
using Symbolics
using Test
using LinearAlgebra
#using FastLapackInterface



println("Test_psdsylv")


@testset "pdsylvc" begin

A0 = [2.7 0.9;-1.1 2.3]; A1 = [4.2 1.3; -1.9 3.8]; A2 = [6.1 3.8; -3.1 6.3];
B0 = [1.5 -0.2; 0.4 1.]; B1 = [2.1 -0.4; 0.4 2.]; B2 = [3.1 -0.6; 0.7 3.5];
C0 = [13.2 10.6; 0.6 8.4]; C1 = [26.4 21.2; 1.2 16.8]; C2 = [38.6 32.1; 1.6 24.2];

fast = true; isgn = 1
for fast in (true,false)
for isgn in (-1,1)
A = PeriodicArray(reshape([A0 A1 A2],2,2,3),3); 
B = PeriodicArray(reshape([B0 B1 B2],2,2,3),3); 
C = PeriodicArray(reshape([C0 C1 C2],2,2,3),3); 

X = pdsylvc(A,B,C; isgn, fast)
@test norm(A*X+isgn*pmshift(X)*B-C) < 1.e-13

X = pdsylvc(A,B,C;isgn,rev = true, fast)
@test norm(isgn*A*pmshift(X)+X*B-C) < 1.e-13

X = pfdsylvc(A,B,C; isgn, fast)
@test norm(A*X+isgn*pmshift(X)*B-C) < 1.e-13

X = prdsylvc(A,B,C; isgn, fast)
@test norm(isgn*A*pmshift(X)+X*B-C) < 1.e-13

# X = PeriodicArray(psylsolve2(A.M,B.M,C.M; isgn),3)
# @test norm(A*X+isgn*pmshift(X)*B-C) < 1.e-13

# X = PeriodicArray(psylsolve2(A.M,B.M,C.M;isgn,rev = true),3)
# @test norm(isgn*A*pmshift(X)+X*B-C) < 1.e-13

At = PeriodicMatrix([A0, A1, A2],3); 
Bt = PeriodicMatrix([B0, B1, B2],3); 
Ct = PeriodicMatrix([C0, C1, C2],3); 

Xt = pdsylvc(At,Bt,Ct; isgn, fast)
@test norm(At*Xt+isgn*pmshift(Xt)*Bt-Ct) < 1.e-13

Xt = pdsylvc(At,Bt,Ct; isgn, rev = true, fast)
@test norm(isgn*At*pmshift(Xt)+Xt*Bt-Ct) < 1.e-13


A00 = [2.7 0.9;]; A01 = [4.2;;]; A02 = [6.1; -3.1;;];
C00 = [13.2 10.6;]; C01 = [26.4 21.2;]; C02 = [38.6 32.1; 1.6 24.2];
At0 = PeriodicMatrix([A00, A01, A02],3); 
Bt0 = PeriodicMatrix([B0, B1, B2],3); 
Ct0 = PeriodicMatrix([C00, C01, C02],3); 

Xt0 = pdsylvc(At0,Bt0,Ct0; isgn, fast)
@test norm(At0*Xt0+isgn*pmshift(Xt0)*Bt0-Ct0)/norm(Xt0) < 1.e-13

At0 = PeriodicMatrix([A02, A01, A00],3); 
Bt0 = PeriodicMatrix([B2, B1, B0],3); 
Ct0 = PeriodicMatrix([C02, C01, C00],3); 
Xt0 = pdsylvc(At0,Bt0,Ct0; isgn, rev=true, fast)
@test norm(isgn*At0*pmshift(Xt0)+Xt0*Bt0-Ct0) < 1.e-13

# # fails
# At0 = PeriodicMatrix([A00, A01, A02],3); 
# Xt0 = pdsylvc(At0,At0,At0; isgn)
# @test norm(At0*Xt0 + isgn*pmshift(Xt0)*At0 - At0) < 1.e-13

# # failsAt0 = PeriodicMatrix([A00', A01', A02'],3);
# Xt0 = pdsylvc(At0,At0,At0; isgn, rev=true);
# @test norm(isgn*At0*pmshift(Xt0)+Xt0*At0-At0) < 1.e-13


A = PeriodicArray(rand(5,5,3),3);
B = PeriodicArray(rand(3,3,3),3);
C = PeriodicArray(rand(5,3,3),3);
@time X = pdsylvc(A,B,C; isgn, fast)
@test norm(A*X+isgn*pmshift(X)*B-C)/norm(X) < 1.e-13

X = pdsylvc(A,B,C;isgn,rev = true, fast)
@test norm(isgn*A*pmshift(X)+X*B-C)/norm(X) < 1.e-13

A = PeriodicArray(rand(5,5),3,nperiod = 3);
B = PeriodicArray(rand(3,3,3),3);
C = PeriodicArray(rand(5,3,1),3;nperiod=3);
X = pdsylvc(A,B,C; isgn, fast)
@test norm(A*X+isgn*pmshift(X)*B-C)/norm(X) < 1.e-13

X = pdsylvc(A,B,C;isgn,rev = true, fast)
@test norm(isgn*A*pmshift(X)+X*B-C)/norm(X) < 1.e-13

A = PeriodicArray(rand(5,5),3,nperiod = 3);
B = PeriodicArray(rand(3,3),3,nperiod = 3);
C = PeriodicArray(rand(5,3,1),3;nperiod=3);
X = pdsylvc(A,B,C; isgn, fast)
@test norm(A*X+isgn*pmshift(X)*B-C)/norm(X) < 1.e-13

# fails for isgn, fast = (-1, true)
X = pdsylvc(A,B,C;isgn,rev = true, fast)
@test norm(isgn*A*pmshift(X)+X*B-C)/norm(X) < 1.e-13

if fast
n = 50; m = 30; p = 300
A = PeriodicArray(rand(n,n,p),p);
B = PeriodicArray(rand(m,m,p),p);
C = PeriodicArray(rand(n,m,p),p);
@time X = pdsylvc(A,B,C; isgn);
@test norm(A*X+isgn*pmshift(X)*B-C)/norm(X) < 1.e-13

@time X = pdsylvc(A,B,C;isgn,rev = true)
@test norm(isgn*A*pmshift(X)+X*B-C)/norm(X) < 1.e-13
end


# n = 5; m = 3; p = 100
# A = PeriodicArray(rand(n,n,p),p);
# B = PeriodicArray(rand(m,m,p),p);
# C = PeriodicArray(rand(n,m,p),p);
# @time X = PeriodicArray(psylsolve2(A.M,B.M,C.M; isgn),p)
# @test norm(A*X+isgn*pmshift(X)*B-C)/norm(X) < 1.e-13

# @time X = PeriodicArray(psylsolve2(A.M,B.M,C.M;isgn,rev = true),p)
# @test norm(isgn*A*pmshift(X)+X*B-C)/norm(X) < 1.e-13
end

end

end


@testset "pssylvdckr" begin

A0 = [2.7 0.9;-1.1 2.3]; A1 = [4.2 1.3; -1.9 3.8]; A2 = [6.1 3.8; -3.1 6.3];
B0 = [1.5 -0.2; 0.4 1.]; B1 = [2.1 -0.4; 0.4 2.]; B2 = [3.1 -0.6; 0.7 3.5];
C0 = [13.2 10.6; 0.6 8.4]; C1 = [26.4 21.2; 1.2 16.8]; C2 = [38.6 32.1; 1.6 24.2];

A = PeriodicArray(reshape([A0 A1 A2],2,2,3),3); 
B = PeriodicArray(reshape([B0 B1 B2],2,2,3),3); 
C = PeriodicArray(reshape([C0 C1 C2],2,2,3),3); 

Y = pssylvdckr(A.M,B.M,C.M)
X = PeriodicArray(Y,3)
@test norm(A*X+pmshift(X)*B-C) < 1.e-13

Y = pssylvdckr(A.M,B.M,C.M,rev = true)
X = PeriodicArray(Y,3)
@test norm(A*pmshift(X)+X*B-C) < 1.e-13

At = PeriodicMatrix([A0, A1, A2],3); 
Bt = PeriodicMatrix([B0, B1, B2],3); 
Ct = PeriodicMatrix([C0, C1, C2],3); 

Yt = pssylvdckr(At.M,Bt.M,Ct.M)
Xt = PeriodicMatrix(Yt,3)
@test norm(At*Xt+pmshift(Xt)*Bt-Ct) < 1.e-13

Yt = pssylvdckr(At.M,Bt.M,Ct.M,rev = true)
Xt = PeriodicMatrix(Yt,3)
@test norm(At*pmshift(Xt)+Xt*Bt-Ct) < 1.e-13

A00 = [2.7 0.9]; A01 = [4.2;;]; A02 = [6.1; -3.1];
C00 = [13.2 10.6;]; C01 = [26.4 21.2;]; C02 = [38.6 32.1; 1.6 24.2];
At0 = PeriodicMatrix([A00, A01, A02],3); 
Bt0 = PeriodicMatrix([B0, B1, B2],3); 
Ct0 = PeriodicMatrix([C00, C01, C02],3); 

Yt0 = pssylvdckr(At0.M,Bt0.M,Ct0.M)
Xt0 = PeriodicMatrix(Yt0,3)
@test norm(At0*Xt0+pmshift(Xt0)*Bt0-Ct0) < 1.e-13

At0 = PeriodicMatrix([A02, A01, A00],3); 
Bt0 = PeriodicMatrix([B2, B1, B0],3); 
Ct0 = PeriodicMatrix([C02, C01, C00],3); 
Yt0 = pssylvdckr(At0.M,Bt0.M,Ct0.M,rev=true)
Xt0 = PeriodicMatrix(Yt0,3)
@test norm(At0*pmshift(Xt0)+Xt0*Bt0-Ct0) < 1.e-13


A00 = [2.7 0.9]; A01 = [4.2;;]; A02 = [6.1; -3.1];
C00 = [13.2 10.6;]; C01 = [26.4 21.2;]; C02 = [38.6 32.1; 1.6 24.2];
Bt0 = PeriodicMatrix([A00', A01', A02'],3); 
At0 = PeriodicMatrix([B0', B1', B2'],3); 
Ct0 = PeriodicMatrix([C00', C01', C02'],3); 
Yt0 = pssylvdckr(At0.M,Bt0.M,Ct0.M,rev=true)
Xt0 = PeriodicMatrix(Yt0,3)
@test norm(At0*pmshift(Xt0)+Xt0*Bt0-Ct0) < 1.e-13

Bt0 = PeriodicMatrix([A00, A01, A02],3); 
At0 = PeriodicMatrix([B0, B1, B2],3); 
C00 = [13.2 10.6; 0.6 8.4]; C01 = [26.4; 21.2]; C02 = [38.6 ; 1.6];
Ct0 = PeriodicMatrix([C00, C01, C02],3); 

Yt0 = pssylvdckr(At0.M,Bt0.M,Ct0.M)
Xt0 = PeriodicMatrix(Yt0,3)
@test norm(At0*Xt0+pmshift(Xt0)*Bt0-Ct0) < 1.e-13

At0 = PeriodicMatrix([A00, A01, A02],3); 
Yt0 = pssylvdckr(At0.M,At0.M,At0.M)
Xt0 = PeriodicMatrix(Yt0,3)
@test norm(At0*Xt0 + pmshift(Xt0)*At0 - At0) < 1.e-13

At0 = PeriodicMatrix([A00', A01', A02'],3);
Yt0 = pssylvdckr(At0.M,At0.M,At0.M,rev=true);
Xt0 = PeriodicMatrix(Yt0,3)
@test norm(At0*pmshift(Xt0)+Xt0*At0-At0) < 1.e-13


end

# @testset "dpsylv2, dpsylv2!, dpsylv2krsol!" begin


# W = Matrix{Float64}(undef,2,14)
# WZ = Matrix{Float64}(undef,8,8)
# WY = Vector{Float64}(undef,8)
# WX = Matrix{Float64}(undef,4,5)

# p = 2; 
# al = [-0.0028238980383030643;;; 0.3319882632937995]
# ar = [-0.0028238980383030643;;; 0.3319882632937995]
# q = rand(2,2,p);
# REV = true
# KSCHUR = 1
# n1 = 1; n2 = 1; 
# X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# i1 = 1:n1; i2 = 1:n2
# rez1 = al[i1,i1,1]'*X[i1,i2,2]*ar[i2,i2,1]-X[i1,i2,1]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]'*X[i1,i2,1]*ar[i2,i2,2]-X[i1,i2,2]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7

# X1 = copy(q[i1,i2,1:p]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez1 = al[i1,i1,1]'*X1[i1,i2,2]*ar[i2,i2,1]-X1[i1,i2,1]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]'*X1[i1,i2,1]*ar[i2,i2,2]-X1[i1,i2,2]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  

# al = [-0.0028238980383030643;;; 0.3319882632937995]*100
# ar = [-0.0028238980383030643;;; 0.3319882632937995]*100
# X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# rez1 = al[i1,i1,1]'*X[i1,i2,2]*ar[i2,i2,1]-X[i1,i2,1]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]'*X[i1,i2,1]*ar[i2,i2,2]-X[i1,i2,2]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7 

# X1 = copy(q[i1,i2,1:p]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez1 = al[i1,i1,1]'*X1[i1,i2,2]*ar[i2,i2,1]-X1[i1,i2,1]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]'*X1[i1,i2,1]*ar[i2,i2,2]-X1[i1,i2,2]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  



# p = 1; 
# al = 100*rand(2,2,p); ar = rand(2,2,p); q = rand(2,2,p);
# REV = true

# KSCHUR = 1
# n1 = 1; n2 = 1; 
# X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# i1 = 1:n1; i2 = 1:n2
# rez = al[i1,i1,1]'*X[i1,i2,1]*ar[i2,i2,1]-X[i1,i2,1]+q[i1,i2,1]   
# @test norm(rez) < 1.e-7         
# X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# rez = al[i1,i1,1]*X[i1,i2,1]*ar[i2,i2,1]'-X[i1,i2,1]+q[i1,i2,1]   
# @test norm(rez) < 1.e-7  

# X1 = copy(q[i1,i2,1:p]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez = al[i1,i1,1]*X1[i1,i2,1]*ar[i2,i2,1]'-X1[i1,i2,1]+q[i1,i2,1]   
# @test norm(rez) < 1.e-7  
# X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez = al[i1,i1,1]*X1[i1,i2,1]*ar[i2,i2,1]'-X1[i1,i2,1]+q[i1,i2,1]   
# @test norm(rez) < 1.e-7  


# n1 = 1; n2 = 2; 
# X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# i1 = 1:n1; i2 = 1:n2
# rez = al[i1,i1,1]'*X[i1,i2,1]*ar[i2,i2,1]-X[i1,i2,1]+q[i1,i2,1]   
# @test norm(rez) < 1.e-7         
# X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# rez = al[i1,i1,1]*X[i1,i2,1]*ar[i2,i2,1]'-X[i1,i2,1]+q[i1,i2,1]   
# @test norm(rez) < 1.e-7  


# X1 = copy(q[i1,i2,1:p]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez = al[i1,i1,1]'*X1[i1,i2,1]*ar[i2,i2,1]-X1[i1,i2,1]+q[i1,i2,1]   
# @test norm(rez) < 1.e-7  
# X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez = al[i1,i1,1]*X1[i1,i2,1]*ar[i2,i2,1]'-X1[i1,i2,1]+q[i1,i2,1]   
# @test norm(rez) < 1.e-7  



# n1 = 2; n2 = 1; 
# X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# i1 = 1:n1; i2 = 1:n2
# rez = al[i1,i1,1]'*X[i1,i2,1]*ar[i2,i2,1]-X[i1,i2,1]+q[i1,i2,1]   
# @test norm(rez) < 1.e-7         
# X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# rez = al[i1,i1,1]*X[i1,i2,1]*ar[i2,i2,1]'-X[i1,i2,1]+q[i1,i2,1]   
# @test norm(rez) < 1.e-7  

# X1 = copy(q[i1,i2,1:p]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez = al[i1,i1,1]'*X1[i1,i2,1]*ar[i2,i2,1]-X1[i1,i2,1]+q[i1,i2,1]   
# @test norm(rez) < 1.e-7  
# X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez = al[i1,i1,1]*X1[i1,i2,1]*ar[i2,i2,1]'-X1[i1,i2,1]+q[i1,i2,1]   
# @test norm(rez) < 1.e-7  


# n1 = 2; n2 = 2; 
# X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# i1 = 1:n1; i2 = 1:n2
# rez = al[i1,i1,1]'*X[i1,i2,1]*ar[i2,i2,1]-X[i1,i2,1]+q[i1,i2,1]   
# @test norm(rez) < 1.e-7         
# X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# rez = al[i1,i1,1]*X[i1,i2,1]*ar[i2,i2,1]'-X[i1,i2,1]+q[i1,i2,1]   
# @test norm(rez) < 1.e-7  

# X1 = copy(q[i1,i2,1:p]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez = al[i1,i1,1]'*X1[i1,i2,1]*ar[i2,i2,1]-X1[i1,i2,1]+q[i1,i2,1]   
# @test norm(rez) < 1.e-7  
# X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez = al[i1,i1,1]*X1[i1,i2,1]*ar[i2,i2,1]'-X1[i1,i2,1]+q[i1,i2,1]   
# @test norm(rez) < 1.e-7  

# p = 1; 
# al = rand(2,2,p); al[:,:,1] = triu(al[:,:,1]); ar = rand(2,2,p); ar[:,:,1] = triu(ar[:,:,1]); 
# q = rand(2,2,p);
# REV = true
# KSCHUR = 1
# n1 = 1; n2 = 1; 
# X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# WUSD = Array{Float64,3}(undef,4,4,p)
# WUD = Array{Float64,3}(undef,4,4,p)
# WUL = Matrix{Float64}(undef,4*p,4)
# WY = Vector{Float64}(undef,4*p)
# W = Matrix{Float64}(undef,8,8)
# qr_ws = QRWs(zeros(8), zeros(4))
# ormqr_ws = QROrmWs(zeros(4), qr_ws.τ)  
# i1 = 1:n1; i2 = 1:n2
# X3 = copy(q[i1,i2,1:p]); dpsylv2krsol!(REV, n1, n2, KSCHUR, al, ar, X3, WUD, WUSD, WUL, WY, W, qr_ws, ormqr_ws)  
# rez1 = al[i1,i1,1]'*X3[i1,i2,1]*ar[i2,i2,1]-X3[i1,i2,1]+q[i1,i2,1]   
# #rez2 = al[i1,i1,2]'*X3[i1,i2,1]*ar[i2,i2,2]-X3[i1,i2,2]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  


# W = Matrix{Float64}(undef,2,14)
# WZ = Matrix{Float64}(undef,8,8)
# WY = Vector{Float64}(undef,8)
# WX = Matrix{Float64}(undef,4,5)

# p = 2; 
# al = rand(2,2,p); al[:,:,1] = triu(al[:,:,1]); ar = rand(2,2,p); ar[:,:,1] = triu(ar[:,:,2]); 
# q = rand(2,2,p);
# REV = true
# KSCHUR = 2
# n1 = 1; n2 = 1; 
# X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# i1 = 1:n1; i2 = 1:n2
# rez1 = al[i1,i1,1]'*X[i1,i2,2]*ar[i2,i2,1]-X[i1,i2,1]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]'*X[i1,i2,1]*ar[i2,i2,2]-X[i1,i2,2]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         
# X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# rez1 = al[i1,i1,1]*X[i1,i2,1]*ar[i2,i2,1]'-X[i1,i2,2]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]*X[i1,i2,2]*ar[i2,i2,2]'-X[i1,i2,1]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  

# X1 = copy(q[i1,i2,1:p]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez1 = al[i1,i1,1]'*X1[i1,i2,2]*ar[i2,i2,1]-X1[i1,i2,1]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]'*X1[i1,i2,1]*ar[i2,i2,2]-X1[i1,i2,2]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         
# X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez1 = al[i1,i1,1]*X1[i1,i2,1]*ar[i2,i2,1]'-X1[i1,i2,2]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]*X1[i1,i2,2]*ar[i2,i2,2]'-X1[i1,i2,1]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  


# WUSD3 = Array{Float64,3}(undef,4,4,p)
# WUD3 = Array{Float64,3}(undef,4,4,p)
# WUL3 = Matrix{Float64}(undef,4*p,4)
# WY1 = Vector{Float64}(undef,4*p)
# W1 = Matrix{Float64}(undef,8,8)
# qr_ws = QRWs(zeros(8), zeros(4))
# ormqr_ws = QROrmWs(zeros(4), qr_ws.τ)   



# X3 = copy(q[i1,i2,1:p]); dpsylv2krsol!(REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1, qr_ws, ormqr_ws)  
# rez1 = al[i1,i1,1]'*X3[i1,i2,2]*ar[i2,i2,1]-X3[i1,i2,1]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]'*X3[i1,i2,1]*ar[i2,i2,2]-X3[i1,i2,2]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  
# X3 = copy(q[i1,i2,1:p]); dpsylv2krsol!(!REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1, qr_ws, ormqr_ws)  
# rez1 = al[i1,i1,1]*X3[i1,i2,1]*ar[i2,i2,1]'-X3[i1,i2,2]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]*X3[i1,i2,2]*ar[i2,i2,2]'-X3[i1,i2,1]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  

# al1 = 100*al; ar1 = 100*ar; 
# X = dpsylv2(REV, n1, n2, KSCHUR, al1, ar1, q, W, WX) 
# i1 = 1:n1; i2 = 1:n2
# rez1 = al1[i1,i1,1]'*X[i1,i2,2]*ar1[i2,i2,1]-X[i1,i2,1]+q[i1,i2,1]   
# rez2 = al1[i1,i1,2]'*X[i1,i2,1]*ar1[i2,i2,2]-X[i1,i2,2]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         
# X = dpsylv2(!REV, n1, n2, KSCHUR, al1, ar1, q, W, WX) 
# rez1 = al1[i1,i1,1]*X[i1,i2,1]*ar1[i2,i2,1]'-X[i1,i2,2]+q[i1,i2,1]   
# rez2 = al1[i1,i1,2]*X[i1,i2,2]*ar1[i2,i2,2]'-X[i1,i2,1]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7 

# X1 = copy(q[i1,i2,1:p]); dpsylv2!(REV, n1, n2, KSCHUR, al1, ar1, X1, WZ, WY) 
# rez1 = al1[i1,i1,1]'*X1[i1,i2,2]*ar1[i2,i2,1]-X1[i1,i2,1]+q[i1,i2,1]   
# rez2 = al1[i1,i1,2]'*X1[i1,i2,1]*ar1[i2,i2,2]-X1[i1,i2,2]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         
# X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al1, ar1, X1, WZ, WY) 
# rez1 = al1[i1,i1,1]*X1[i1,i2,1]*ar1[i2,i2,1]'-X1[i1,i2,2]+q[i1,i2,1]   
# rez2 = al1[i1,i1,2]*X1[i1,i2,2]*ar1[i2,i2,2]'-X1[i1,i2,1]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7 


# al = rand(2,2,p); al[:,:,1] = triu(al[:,:,1]); ar = rand(2,2,p); ar[:,:,1] = triu(ar[:,:,2]); 
# n1 = 1; n2 = 2; 
# X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# i1 = 1:n1; i2 = 1:n2
# rez1 = al[i1,i1,1]'*X[i1,i2,2]*ar[i2,i2,1]-X[i1,i2,1]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]'*X[i1,i2,1]*ar[i2,i2,2]-X[i1,i2,2]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         
# X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# rez1 = al[i1,i1,1]*X[i1,i2,1]*ar[i2,i2,1]'-X[i1,i2,2]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]*X[i1,i2,2]*ar[i2,i2,2]'-X[i1,i2,1]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7   

# X1 = copy(q[i1,i2,1:p]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez1 = al[i1,i1,1]'*X1[i1,i2,2]*ar[i2,i2,1]-X1[i1,i2,1]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]'*X1[i1,i2,1]*ar[i2,i2,2]-X1[i1,i2,2]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         
# X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez1 = al[i1,i1,1]*X1[i1,i2,1]*ar[i2,i2,1]'-X1[i1,i2,2]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]*X1[i1,i2,2]*ar[i2,i2,2]'-X1[i1,i2,1]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  


# X3 = copy(q[i1,i2,1:p]); dpsylv2krsol!(REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1, qr_ws, ormqr_ws)  
# rez1 = al[i1,i1,1]'*X3[i1,i2,2]*ar[i2,i2,1]-X3[i1,i2,1]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]'*X3[i1,i2,1]*ar[i2,i2,2]-X3[i1,i2,2]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  
# X3 = copy(q[i1,i2,1:p]); dpsylv2krsol!(!REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1, qr_ws, ormqr_ws)  
# rez1 = al[i1,i1,1]*X3[i1,i2,1]*ar[i2,i2,1]'-X3[i1,i2,2]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]*X3[i1,i2,2]*ar[i2,i2,2]'-X3[i1,i2,1]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  

# al = 100*al; ar = 100*ar; 
# X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# i1 = 1:n1; i2 = 1:n2
# rez1 = al[i1,i1,1]'*X[i1,i2,2]*ar[i2,i2,1]-X[i1,i2,1]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]'*X[i1,i2,1]*ar[i2,i2,2]-X[i1,i2,2]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         
# X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# rez1 = al[i1,i1,1]*X[i1,i2,1]*ar[i2,i2,1]'-X[i1,i2,2]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]*X[i1,i2,2]*ar[i2,i2,2]'-X[i1,i2,1]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7   

# X1 = copy(q[i1,i2,1:p]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez1 = al[i1,i1,1]'*X1[i1,i2,2]*ar[i2,i2,1]-X1[i1,i2,1]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]'*X1[i1,i2,1]*ar[i2,i2,2]-X1[i1,i2,2]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         
# X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) # fails
# rez1 = al[i1,i1,1]*X1[i1,i2,1]*ar[i2,i2,1]'-X1[i1,i2,2]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]*X1[i1,i2,2]*ar[i2,i2,2]'-X1[i1,i2,1]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  

# al = rand(2,2,p); al[:,:,1] = triu(al[:,:,1]); ar = rand(2,2,p); ar[:,:,1] = triu(ar[:,:,2]); 
# n1 = 2; n2 = 1; 
# X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# i1 = 1:n1; i2 = 1:n2
# rez1 = al[i1,i1,1]'*X[i1,i2,2]*ar[i2,i2,1]-X[i1,i2,1]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]'*X[i1,i2,1]*ar[i2,i2,2]-X[i1,i2,2]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         
# X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# rez1 = al[i1,i1,1]*X[i1,i2,1]*ar[i2,i2,1]'-X[i1,i2,2]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]*X[i1,i2,2]*ar[i2,i2,2]'-X[i1,i2,1]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         

# X1 = copy(q[i1,i2,1:p]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez1 = al[i1,i1,1]'*X1[i1,i2,2]*ar[i2,i2,1]-X1[i1,i2,1]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]'*X1[i1,i2,1]*ar[i2,i2,2]-X1[i1,i2,2]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         
# X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez1 = al[i1,i1,1]*X1[i1,i2,1]*ar[i2,i2,1]'-X1[i1,i2,2]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]*X1[i1,i2,2]*ar[i2,i2,2]'-X1[i1,i2,1]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  


# X3 = copy(q[i1,i2,1:p]); dpsylv2krsol!(REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1, qr_ws, ormqr_ws)  
# rez1 = al[i1,i1,1]'*X3[i1,i2,2]*ar[i2,i2,1]-X3[i1,i2,1]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]'*X3[i1,i2,1]*ar[i2,i2,2]-X3[i1,i2,2]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  
# X3 = copy(q[i1,i2,1:p]); dpsylv2krsol!(!REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1, qr_ws, ormqr_ws)  
# rez1 = al[i1,i1,1]*X3[i1,i2,1]*ar[i2,i2,1]'-X3[i1,i2,2]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]*X3[i1,i2,2]*ar[i2,i2,2]'-X3[i1,i2,1]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  

# al1 = 100*al; ar1 = 100*ar; 
# X = dpsylv2(REV, n1, n2, KSCHUR, al1, ar1, q, W, WX) 
# i1 = 1:n1; i2 = 1:n2
# rez1 = al1[i1,i1,1]'*X[i1,i2,2]*ar1[i2,i2,1]-X[i1,i2,1]+q[i1,i2,1]   
# rez2 = al1[i1,i1,2]'*X[i1,i2,1]*ar1[i2,i2,2]-X[i1,i2,2]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         
# X = dpsylv2(!REV, n1, n2, KSCHUR, al1, ar1, q, W, WX) 
# rez1 = al1[i1,i1,1]*X[i1,i2,1]*ar1[i2,i2,1]'-X[i1,i2,2]+q[i1,i2,1]   
# rez2 = al1[i1,i1,2]*X[i1,i2,2]*ar1[i2,i2,2]'-X[i1,i2,1]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         

# X1 = copy(q[i1,i2,1:p]); dpsylv2!(REV, n1, n2, KSCHUR, al1, ar1, X1, WZ, WY) 
# rez1 = al1[i1,i1,1]'*X1[i1,i2,2]*ar1[i2,i2,1]-X1[i1,i2,1]+q[i1,i2,1]   
# rez2 = al1[i1,i1,2]'*X1[i1,i2,1]*ar1[i2,i2,2]-X1[i1,i2,2]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         
# X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al1, ar1, X1, WZ, WY) 
# rez1 = al1[i1,i1,1]*X1[i1,i2,1]*ar1[i2,i2,1]'-X1[i1,i2,2]+q[i1,i2,1]   
# rez2 = al1[i1,i1,2]*X1[i1,i2,2]*ar1[i2,i2,2]'-X1[i1,i2,1]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  


# al = rand(2,2,p); al[:,:,1] = triu(al[:,:,1]); ar = rand(2,2,p); ar[:,:,1] = triu(ar[:,:,2]); 
# n1 = 2; n2 = 2; 
# X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# i1 = 1:n1; i2 = 1:n2
# rez1 = al[i1,i1,1]'*X[i1,i2,2]*ar[i2,i2,1]-X[i1,i2,1]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]'*X[i1,i2,1]*ar[i2,i2,2]-X[i1,i2,2]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         
# X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# rez1 = al[i1,i1,1]*X[i1,i2,1]*ar[i2,i2,1]'-X[i1,i2,2]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]*X[i1,i2,2]*ar[i2,i2,2]'-X[i1,i2,1]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7   

# X1 = copy(q[i1,i2,1:p]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez1 = al[i1,i1,1]'*X1[i1,i2,2]*ar[i2,i2,1]-X1[i1,i2,1]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]'*X1[i1,i2,1]*ar[i2,i2,2]-X1[i1,i2,2]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         
# X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez1 = al[i1,i1,1]*X1[i1,i2,1]*ar[i2,i2,1]'-X1[i1,i2,2]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]*X1[i1,i2,2]*ar[i2,i2,2]'-X1[i1,i2,1]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  

# X3 = copy(q[i1,i2,1:p]); dpsylv2krsol!(REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1, qr_ws, ormqr_ws)  
# rez1 = al[i1,i1,1]'*X3[i1,i2,2]*ar[i2,i2,1]-X3[i1,i2,1]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]'*X3[i1,i2,1]*ar[i2,i2,2]-X3[i1,i2,2]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  
# X3 = copy(q[i1,i2,1:p]); dpsylv2krsol!(!REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1, qr_ws, ormqr_ws)  
# rez1 = al[i1,i1,1]*X3[i1,i2,1]*ar[i2,i2,1]'-X3[i1,i2,2]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]*X3[i1,i2,2]*ar[i2,i2,2]'-X3[i1,i2,1]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7  

# al = 10*al; ar = 10*ar; 
# X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# i1 = 1:n1; i2 = 1:n2
# rez1 = al[i1,i1,1]'*X[i1,i2,2]*ar[i2,i2,1]-X[i1,i2,1]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]'*X[i1,i2,1]*ar[i2,i2,2]-X[i1,i2,2]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         
# X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# rez1 = al[i1,i1,1]*X[i1,i2,1]*ar[i2,i2,1]'-X[i1,i2,2]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]*X[i1,i2,2]*ar[i2,i2,2]'-X[i1,i2,1]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7   

# X1 = copy(q[i1,i2,1:p]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez1 = al[i1,i1,1]'*X1[i1,i2,2]*ar[i2,i2,1]-X1[i1,i2,1]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]'*X1[i1,i2,1]*ar[i2,i2,2]-X1[i1,i2,2]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7         
# X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez1 = al[i1,i1,1]*X1[i1,i2,1]*ar[i2,i2,1]'-X1[i1,i2,2]+q[i1,i2,1]   
# rez2 = al[i1,i1,2]*X1[i1,i2,2]*ar[i2,i2,2]'-X1[i1,i2,1]+q[i1,i2,2]   
# @test norm(rez1) < 1.e-7  && norm(rez2) < 1.e-7 


# p = 100; 
# WZ = Matrix{Float64}(undef,p*4,p*4)
# WY = Vector{Float64}(undef,p*4)
# W = Matrix{Float64}(undef,2,14)
# WX = Matrix{Float64}(undef,4,5)


# WUSD3 = Array{Float64,3}(undef,4,4,p)
# WUD3 = Array{Float64,3}(undef,4,4,p)
# WUL3 = Matrix{Float64}(undef,4*p,4)
# WY1 = Vector{Float64}(undef,4*p)
# W1 = Matrix{Float64}(undef,8,8)



# al = rand(2,2,p); ar = rand(2,2,p); q = rand(2,2,p);
# [triu!(view(al,:,:,i)) for i in 2:p]
# [triu!(view(ar,:,:,i)) for i in 2:p]
# REV = true

# KSCHUR = 1

# n1 = 1; n2 = 1; 
# @time X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX);
# i1 = 1:n1; i2 = 1:n2
# ip = 1:p; ip1 = mod.(ip,p).+1;
# rez = [ al[i1,i1,ip[i]]'*X[i1,i2,ip1[i]]*ar[i2,i2,ip[i]]-X[i1,i2,ip[i]]+q[i1,i2,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7           
# X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# rez = [ al[i1,i1,ip[i]]*X[i1,i2,ip[i]]*ar[i2,i2,ip[i]]'-X[i1,i2,ip1[i]]+q[i1,i2,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7 

# X1 = copy(q[i1,i2,1:p]); @time dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez = [ al[i1,i1,ip[i]]'*X1[i1,i2,ip1[i]]*ar[i2,i2,ip[i]]-X1[i1,i2,ip[i]]+q[i1,i2,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7           
# X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez = [ al[i1,i1,ip[i]]*X1[i1,i2,ip[i]]*ar[i2,i2,ip[i]]'-X1[i1,i2,ip1[i]]+q[i1,i2,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7 


# X3 = copy(q[i1,i2,1:p]); @time dpsylv2krsol!(REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1, qr_ws, ormqr_ws) 
# rez = [ al[i1,i1,ip[i]]'*X3[i1,i2,ip1[i]]*ar[i2,i2,ip[i]]-X3[i1,i2,ip[i]]+q[i1,i2,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7  
# X3 = copy(q[i1,i2,1:p]); @time dpsylv2krsol!(!REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1, qr_ws, ormqr_ws)  
# rez = [ al[i1,i1,ip[i]]*X3[i1,i2,ip[i]]*ar[i2,i2,ip[i]]'-X3[i1,i2,ip1[i]]+q[i1,i2,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7 


# n1 = 1; n2 = 2; 
# X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# i1 = 1:n1; i2 = 1:n2
# ip = 1:p; ip1 = mod.(ip,p).+1;
# rez = [ al[i1,i1,ip[i]]'*X[i1,i2,ip1[i]]*ar[i2,i2,ip[i]]-X[i1,i2,ip[i]]+q[i1,i2,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7           
# X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# rez = [ al[i1,i1,ip[i]]*X[i1,i2,ip[i]]*ar[i2,i2,ip[i]]'-X[i1,i2,ip1[i]]+q[i1,i2,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7 

# X1 = copy(q[i1,i2,1:p]); @time dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez = [ al[i1,i1,ip[i]]'*X1[i1,i2,ip1[i]]*ar[i2,i2,ip[i]]-X1[i1,i2,ip[i]]+q[i1,i2,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7           
# X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez = [ al[i1,i1,ip[i]]*X1[i1,i2,ip[i]]*ar[i2,i2,ip[i]]'-X1[i1,i2,ip1[i]]+q[i1,i2,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7 


# X3 = copy(q[i1,i2,1:p]); @time dpsylv2krsol!(REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1, qr_ws, ormqr_ws)  
# rez = [ al[i1,i1,ip[i]]'*X3[i1,i2,ip1[i]]*ar[i2,i2,ip[i]]-X3[i1,i2,ip[i]]+q[i1,i2,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7  
# X3 = copy(q[i1,i2,1:p]); dpsylv2krsol!(!REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1, qr_ws, ormqr_ws)  
# rez = [ al[i1,i1,ip[i]]*X3[i1,i2,ip[i]]*ar[i2,i2,ip[i]]'-X3[i1,i2,ip1[i]]+q[i1,i2,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7 



# n1 = 2; n2 = 1; 
# X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# i1 = 1:n1; i2 = 1:n2
# ip = 1:p; ip1 = mod.(ip,p).+1;
# rez = [ al[i1,i1,ip[i]]'*X[i1,i2,ip1[i]]*ar[i2,i2,ip[i]]-X[i1,i2,ip[i]]+q[i1,i2,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7           
# X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# rez = [ al[i1,i1,ip[i]]*X[i1,i2,ip[i]]*ar[i2,i2,ip[i]]'-X[i1,i2,ip1[i]]+q[i1,i2,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7 

# X1 = copy(q[i1,i2,1:p]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez = [ al[i1,i1,ip[i]]'*X1[i1,i2,ip1[i]]*ar[i2,i2,ip[i]]-X1[i1,i2,ip[i]]+q[i1,i2,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7           
# X1 = copy(q[i1,i2,1:p]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez = [ al[i1,i1,ip[i]]*X1[i1,i2,ip[i]]*ar[i2,i2,ip[i]]'-X1[i1,i2,ip1[i]]+q[i1,i2,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7 


# X3 = copy(q[i1,i2,1:p]); dpsylv2krsol!(REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1, qr_ws, ormqr_ws)  
# rez = [ al[i1,i1,ip[i]]'*X3[i1,i2,ip1[i]]*ar[i2,i2,ip[i]]-X3[i1,i2,ip[i]]+q[i1,i2,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7  
# X3 = copy(q[i1,i2,1:p]); dpsylv2krsol!(!REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1, qr_ws, ormqr_ws)  
# rez = [ al[i1,i1,ip[i]]*X3[i1,i2,ip[i]]*ar[i2,i2,ip[i]]'-X3[i1,i2,ip1[i]]+q[i1,i2,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7 



# n1 = 2; n2 = 2; 
# @time X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX); 
# i1 = 1:n1; i2 = 1:n2
# ip = 1:p; ip1 = mod.(ip,p).+1;
# rez = [ al[i1,i1,ip[i]]'*X[i1,i2,ip1[i]]*ar[i2,i2,ip[i]]-X[i1,i2,ip[i]]+q[i1,i2,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7           
# X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# rez = [ al[i1,i1,ip[i]]*X[i1,i2,ip[i]]*ar[i2,i2,ip[i]]'-X[i1,i2,ip1[i]]+q[i1,i2,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7 

# X1 = copy(q[i1,i2,1:p]); @time dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez = [ al[i1,i1,ip[i]]'*X1[i1,i2,ip1[i]]*ar[i2,i2,ip[i]]-X1[i1,i2,ip[i]]+q[i1,i2,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7           
# X1 = copy(q[i1,i2,1:p]); @time dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez = [ al[i1,i1,ip[i]]*X1[i1,i2,ip[i]]*ar[i2,i2,ip[i]]'-X1[i1,i2,ip1[i]]+q[i1,i2,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7 

# X3 = copy(q[i1,i2,1:p]); @time dpsylv2krsol!(REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1, qr_ws, ormqr_ws)  
# rez = [ al[i1,i1,ip[i]]'*X3[i1,i2,ip1[i]]*ar[i2,i2,ip[i]]-X3[i1,i2,ip[i]]+q[i1,i2,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7  
# X3 = copy(q[i1,i2,1:p]); @time dpsylv2krsol!(!REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1, qr_ws, ormqr_ws)  
# rez = [ al[i1,i1,ip[i]]*X3[i1,i2,ip[i]]*ar[i2,i2,ip[i]]'-X3[i1,i2,ip1[i]]+q[i1,i2,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7 



# #pq = 2; p = 1
# pq = 10; p = 2
# WZ = Matrix{Float64}(undef,pq*4,pq*4)
# WY = Vector{Float64}(undef,pq*4)
# W = Matrix{Float64}(undef,2,14)
# WX = Matrix{Float64}(undef,4,5)

# WUSD3 = Array{Float64,3}(undef,4,4,pq)
# WUD3 = Array{Float64,3}(undef,4,4,pq)
# WUL3 = Matrix{Float64}(undef,4*pq,4)
# WY1 = Vector{Float64}(undef,4*pq)


# al = 0.1*rand(2,2,p); ar = rand(2,2,p); q = rand(2,2,pq);
# [triu!(view(al,:,:,i)) for i in 2:p]
# [triu!(view(ar,:,:,i)) for i in 2:p]
# REV = true
# KSCHUR = 1

# n1 = 1; n2 = 1; 
# X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# i1 = 1:n1; i2 = 1:n2
# ip = mod.(0:pq-1,p).+1; ip1 = mod.(ip,p).+1;  ipq = mod.(0:pq-1,pq).+1; ipq1 = mod.(1:pq,pq).+1; 
# rez = [ al[i1,i1,ip[i]]'*X[i1,i2,ipq1[i]]*ar[i2,i2,ip[i]]-X[i1,i2,ipq[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
# @test norm(rez) < 1.e-7           
# X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# rez = [ al[i1,i1,ip[i]]*X[i1,i2,ipq[i]]*ar[i2,i2,ip[i]]'-X[i1,i2,ipq1[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
# @test norm(rez) < 1.e-7 

# X1 = copy(q[i1,i2,1:pq]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez = [ al[i1,i1,ip[i]]'*X1[i1,i2,ipq1[i]]*ar[i2,i2,ip[i]]-X1[i1,i2,ipq[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
# @test norm(rez) < 1.e-7           
# X1 = copy(q[i1,i2,1:pq]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez = [ al[i1,i1,ip[i]]*X1[i1,i2,ipq[i]]*ar[i2,i2,ip[i]]'-X1[i1,i2,ipq1[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
# @test norm(rez) < 1.e-7 

# X3 = copy(q[i1,i2,1:pq]); @time dpsylv2krsol!(REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1, qr_ws, ormqr_ws)  
# rez = [ al[i1,i1,ip[i]]'*X3[i1,i2,ipq1[i]]*ar[i2,i2,ip[i]]-X3[i1,i2,ipq[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
# @test norm(rez) < 1.e-7           
# X3 = copy(q[i1,i2,1:pq]); @time dpsylv2krsol!(!REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1, qr_ws, ormqr_ws)  
# rez = [ al[i1,i1,ip[i]]*X3[i1,i2,ipq[i]]*ar[i2,i2,ip[i]]'-X3[i1,i2,ipq1[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
# @test norm(rez) < 1.e-7 



# n1 = 1; n2 = 2; 
# X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# i1 = 1:n1; i2 = 1:n2
# ip = mod.(0:pq-1,p).+1; ip1 = mod.(ip,p).+1;  ipq = mod.(0:pq-1,pq).+1; ipq1 = mod.(1:pq,pq).+1; 
# rez = [ al[i1,i1,ip[i]]'*X[i1,i2,ipq1[i]]*ar[i2,i2,ip[i]]-X[i1,i2,ipq[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
# @test norm(rez) < 1.e-7           
# X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# rez = [ al[i1,i1,ip[i]]*X[i1,i2,ipq[i]]*ar[i2,i2,ip[i]]'-X[i1,i2,ipq1[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
# @test norm(rez) < 1.e-7 

# X1 = copy(q[i1,i2,1:pq]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) # fails
# rez = [ al[i1,i1,ip[i]]'*X1[i1,i2,ipq1[i]]*ar[i2,i2,ip[i]]-X1[i1,i2,ipq[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
# @test norm(rez) < 1.e-7           
# X1 = copy(q[i1,i2,1:pq]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez = [ al[i1,i1,ip[i]]*X1[i1,i2,ipq[i]]*ar[i2,i2,ip[i]]'-X1[i1,i2,ipq1[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
# @test norm(rez) < 1.e-7 

# X3 = copy(q[i1,i2,1:pq]); @time dpsylv2krsol!(REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1, qr_ws, ormqr_ws)  
# rez = [ al[i1,i1,ip[i]]'*X3[i1,i2,ipq1[i]]*ar[i2,i2,ip[i]]-X3[i1,i2,ipq[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
# @test norm(rez) < 1.e-7           
# X3 = copy(q[i1,i2,1:pq]); @time dpsylv2krsol!(!REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1, qr_ws, ormqr_ws)  
# rez = [ al[i1,i1,ip[i]]*X3[i1,i2,ipq[i]]*ar[i2,i2,ip[i]]'-X3[i1,i2,ipq1[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
# @test norm(rez) < 1.e-7 




# n1 = 2; n2 = 1; 
# X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# i1 = 1:n1; i2 = 1:n2
# ip = mod.(0:pq-1,p).+1; ip1 = mod.(ip,p).+1;  ipq = mod.(0:pq-1,pq).+1; ipq1 = mod.(1:pq,pq).+1; 
# rez = [ al[i1,i1,ip[i]]'*X[i1,i2,ipq1[i]]*ar[i2,i2,ip[i]]-X[i1,i2,ipq[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
# @test norm(rez) < 1.e-7           
# X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# rez = [ al[i1,i1,ip[i]]*X[i1,i2,ipq[i]]*ar[i2,i2,ip[i]]'-X[i1,i2,ipq1[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
# @test norm(rez) < 1.e-7 

# X1 = copy(q[i1,i2,1:pq]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez = [ al[i1,i1,ip[i]]'*X1[i1,i2,ipq1[i]]*ar[i2,i2,ip[i]]-X1[i1,i2,ipq[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
# @test norm(rez) < 1.e-7           
# X1 = copy(q[i1,i2,1:pq]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez = [ al[i1,i1,ip[i]]*X1[i1,i2,ipq[i]]*ar[i2,i2,ip[i]]'-X1[i1,i2,ipq1[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
# @test norm(rez) < 1.e-7 

# X3 = copy(q[i1,i2,1:pq]); @time dpsylv2krsol!(REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1, qr_ws, ormqr_ws)  
# rez = [ al[i1,i1,ip[i]]'*X3[i1,i2,ipq1[i]]*ar[i2,i2,ip[i]]-X3[i1,i2,ipq[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
# @test norm(rez) < 1.e-7           
# X3 = copy(q[i1,i2,1:pq]); @time dpsylv2krsol!(!REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1, qr_ws, ormqr_ws)  
# rez = [ al[i1,i1,ip[i]]*X3[i1,i2,ipq[i]]*ar[i2,i2,ip[i]]'-X3[i1,i2,ipq1[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
# @test norm(rez) < 1.e-7 

# n1 = 2; n2 = 2; 
# X = dpsylv2(REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# i1 = 1:n1; i2 = 1:n2
# ip = mod.(0:pq-1,p).+1; ip1 = mod.(ip,p).+1;  ipq = mod.(0:pq-1,pq).+1; ipq1 = mod.(1:pq,pq).+1; 
# rez = [ al[i1,i1,ip[i]]'*X[i1,i2,ipq1[i]]*ar[i2,i2,ip[i]]-X[i1,i2,ipq[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
# @test norm(rez) < 1.e-7           
# X = dpsylv2(!REV, n1, n2, KSCHUR, al, ar, q, W, WX) 
# rez = [ al[i1,i1,ip[i]]*X[i1,i2,ipq[i]]*ar[i2,i2,ip[i]]'-X[i1,i2,ipq1[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
# @test norm(rez) < 1.e-7 

# X1 = copy(q[i1,i2,1:pq]); dpsylv2!(REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez = [ al[i1,i1,ip[i]]'*X1[i1,i2,ipq1[i]]*ar[i2,i2,ip[i]]-X1[i1,i2,ipq[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
# @test norm(rez) < 1.e-7           
# X1 = copy(q[i1,i2,1:pq]); dpsylv2!(!REV, n1, n2, KSCHUR, al, ar, X1, WZ, WY) 
# rez = [ al[i1,i1,ip[i]]*X1[i1,i2,ipq[i]]*ar[i2,i2,ip[i]]'-X1[i1,i2,ipq1[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
# @test norm(rez) < 1.e-7 

# X3 = copy(q[i1,i2,1:pq]); @time dpsylv2krsol!(REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1, qr_ws, ormqr_ws)  
# rez = [ al[i1,i1,ip[i]]'*X3[i1,i2,ipq1[i]]*ar[i2,i2,ip[i]]-X3[i1,i2,ipq[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
# @test norm(rez) < 1.e-7           
# X3 = copy(q[i1,i2,1:pq]); @time dpsylv2krsol!(!REV, n1, n2, KSCHUR, al, ar, X3, WUD3, WUSD3, WUL3, WY1, W1, qr_ws, ormqr_ws)  
# rez = [ al[i1,i1,ip[i]]*X3[i1,i2,ipq[i]]*ar[i2,i2,ip[i]]'-X3[i1,i2,ipq1[i]]+q[i1,i2,ipq[i]] for i in 1:pq]
# @test norm(rez) < 1.e-7 

# end # dpsylv2


# @testset "prdlyap && pfdlyap" begin

# # constant dimensions
# na = [5, 5]; ma = [5,5]; pa = 2; pc = 2;   
# Ad = PeriodicMatrix([rand(Float64,ma[i],na[i]) for i in 1:pa],pa);  
# x = [rand(na[i],na[i]) for i in 1:pc]
# Qd = PeriodicMatrix([ x[i]+x[i]' for i in 1:pc],pc);
# X2 = PeriodicMatrix(pslyapdkr(Ad.M, Qd.M; adj = true), lcm(pa,pc));
# @test norm(Ad'*pmshift(X2)*Ad-X2+Qd) < 1.e-7 

# # time-varying dimensions
# na = [5, 3, 3, 4, 1]; ma = [3, 3, 4, 1, 5]; pa = 5; pc = 5;   
# #na = 5*na; ma = 5*ma;
# Ad = PeriodicMatrix([rand(Float64,ma[i],na[i]) for i in 1:pa],pa);  
# x = [rand(na[i],na[i]) for i in 1:pc]
# Qd = PeriodicMatrix([ x[i]+x[i]' for i in 1:pc],pc);
# X = prdlyap(Ad, Qd);
# @test norm(Ad'*pmshift(X)*Ad-X+Qd) < 1.e-7 

# Ad1 = convert(PeriodicArray,Ad); Qd1 = convert(PeriodicArray,Qd);
# X1 = prdlyap(Ad1, Qd1); 
# @test norm(Ad1'*pmshift(X1)*Ad1-X1+Qd1) < 1.e-7 

# X2 = PeriodicMatrix(pslyapdkr(Ad.M, Qd.M; adj = true), lcm(pa,pc));
# @test norm(Ad'*pmshift(X2)*Ad-X2+Qd) < 1.e-7 && norm(X1-pm2pa(X2)) < 1.e-7

# x = [rand(ma[i],ma[i]) for i in 1:pc]
# Qd = PeriodicMatrix([ x[i]+x[i]' for i in 1:pc],pc);

# X = pfdlyap(Ad, Qd) 
# @test norm(Ad*X*Ad'- pmshift(X)+Qd) < 1.e-7 

# Ad1 = convert(PeriodicArray,Ad); Qd1 = convert(PeriodicArray,Qd);
# X1 = pfdlyap(Ad1, Qd1); 
# @test norm(Ad1*X1*Ad1'-pmshift(X1)+Qd1) < 1.e-7 

# X2 = PeriodicMatrix(pslyapdkr(Ad.M, Qd.M; adj = false), lcm(pa,pc));
# @test norm(Ad*X2*Ad'-pmshift(X2)+Qd) < 1.e-7 && norm(X1-pm2pa(X2)) < 1.e-7


# # constant dimensions
# n = 5; pa = 10; pc = 2;     
# Ad = 0.5*PeriodicArray(rand(Float32,n,n,pa),pa);
# # q = rand(n,n,pc); [q[:,:,i] = q[:,:,i]'+q[:,:,i] for i in 1:pc];
# # Qd = PeriodicArray(q,pc);
# # pmsymadd!(Qd)
# Qd=pmsymadd!(pmrand(PeriodicArray,n,n,pc,ns=2))

# X = pdlyap(Ad, Qd, adj = true); 
# @test norm(Ad'*pmshift(X)*Ad-X+Qd) < 1.e-7 

# Y = pdlyap(Ad, Qd, adj = false); 
# @test norm(Ad*Y*Ad'-pmshift(Y)+Qd) < 1.e-7 

# Y1, X1 = pdlyap2(Ad,Qd,Qd)
# @test X ≈ X1 && Y ≈ Y1

# Adsw = convert(SwitchingPeriodicArray,Ad); Qdsw = convert(SwitchingPeriodicArray,Qd)
# Xsw = pdlyap(Adsw, Qdsw, adj = true); 
# @test norm(Adsw'*pmshift(Xsw)*Adsw-Xsw+Qdsw) < 1.e-7 

# Ysw = pdlyap(Adsw, Qdsw, adj = false); 
# @test norm(Adsw*Ysw*Adsw'-pmshift(Ysw)+Qdsw) < 1.e-7 

# Ysw1, Xsw1 = pdlyap2(Adsw,Qdsw,Qdsw)
# @test Xsw ≈ Xsw1 && Ysw ≈ Ysw1

# Adsw = convert(SwitchingPeriodicMatrix,Ad); Qdsw = convert(SwitchingPeriodicMatrix,Qd)
# Xsw = pdlyap(Adsw, Qdsw, adj = true); 
# @test norm(Adsw'*pmshift(Xsw)*Adsw-Xsw+Qdsw) < 1.e-7 

# Ysw = pdlyap(Adsw, Qdsw, adj = false); 
# @test norm(Adsw*Ysw*Adsw'-pmshift(Ysw)+Qdsw) < 1.e-7 

# Ysw1, Xsw1 = pdlyap2(Adsw,Qdsw,Qdsw)
# @test Xsw ≈ Xsw1 && Ysw ≈ Ysw1


# X = prdlyap(Ad, Qd); 
# @test norm(Ad'*pmshift(X)*Ad-X+Qd) < 1.e-7 

# p = lcm(pa,pc)
# ia = mod.(0:p-1,pa).+1; ipx = mod.(0:p-1,p).+1;  
# ipc = mod.(0:p-1,pc).+1; ipx1 = mod.(ipx,p).+1;  
# rez = [ Ad.M[:,:,ia[i]]'*X.M[:,:,ipx1[i]]*Ad.M[:,:,ia[i]]-X.M[:,:,ipx[i]]+Qd.M[:,:,ipc[i]] for i in ipx]
# @test norm(rez) < 1.e-7 

# X1 = prdlyap(convert(PeriodicMatrix,Ad), convert(PeriodicMatrix,Qd)); 
# X = convert(PeriodicArray,X1)
# @test norm(Ad'*pmshift(X)*Ad-X+Qd) < 1.e-7 

# X = pfdlyap(Ad, Qd) 
# @test norm(Ad*X*Ad'-pmshift(X)+Qd) < 1.e-7 

# X1 = pfdlyap(convert(PeriodicMatrix,Ad), convert(PeriodicMatrix,Qd)); 
# X = convert(PeriodicArray,X1)
# @test norm(Ad*X*Ad'-pmshift(X)+Qd) < 1.e-7 


# rez = [ Ad.M[:,:,ia[i]]*X.M[:,:,ipx[i]]*Ad.M[:,:,ia[i]]'-X.M[:,:,ipx1[i]]+Qd.M[:,:,ipc[i]] for i in ipx]
# @test norm(rez) < 1.e-7 

# # constant dimensions
# n = 5; pa = 1; pc = 1;     
# Ad = 0.5*PeriodicArray(rand(n,n,pa),pa);
# # q = rand(n,n,pc); [q[:,:,i] = q[:,:,i]'+q[:,:,i] for i in 1:pc];
# # Qd = PeriodicArray(q,pc);
# # pmsymadd!(Qd)
# Qd=pmsymadd!(pmrand(PeriodicArray,n,n,pc,ns=1))

# X = pdlyap(Ad, Qd, adj = true); 
# @test norm(Ad'*pmshift(X)*Ad-X+Qd) < 1.e-7 

# Y = pdlyap(Ad, Qd, adj = false); 
# @test norm(Ad*Y*Ad'-pmshift(Y)+Qd) < 1.e-7 

# Y1, X1 = pdlyap2(Ad,Qd,Qd)
# @test X ≈ X1 && Y ≈ Y1


# #pseig(Ad)

# n = 5; pa = 3; pc = 2;     
# Ad = 0.5*PeriodicArray(rand(Float64,n,n,pa),pa);
# q = rand(n,n,pc); [q[:,:,i] = q[:,:,i]'+q[:,:,i] for i in 1:pc];
# Qd = PeriodicArray(q,pc);

# X = prdlyap(Ad, Qd); 
# @test norm(Ad'*pmshift(X)*Ad-X+Qd) < 1.e-7 

# Y = pfdlyap(Ad, Qd) 
# @test norm(Ad*Y*Ad'-pmshift(Y)+Qd) < 1.e-7 

# p = lcm(pa,pc)
# X1 = zeros(n,n,p); Y1 = zeros(n,n,p); Xt = zeros(n,n); QW = zeros(n,n,p);
# PeriodicMatrixEquations.pslyapd!(X1, copy(Ad.M), copy(Qd.M), Xt, QW; adj = true) 
# PeriodicMatrixEquations.pslyapd!(Y1, copy(Ad.M), copy(Qd.M), Xt, QW; adj = false) 
# @test X1 ≈ X.M && Y1 ≈ Y.M

# T = Float64; N = p; 
# qr_ws = QRWs(zeros(8), zeros(4))
# #WORK1 = (Array{T,3}(undef, n, n, N), Array{T,3}(undef, n, n, N), Array{T,3}(undef, n, n, N), Matrix{T}(undef, n, n), Array{T,3}(undef,n,n,N), ws_pschur(Gt) )
# WORK2 = (Array{Float64,3}(undef,2,2,N), Array{Float64,3}(undef,4,4,N), Array{Float64,3}(undef,4,4,N),
# Matrix{Float64}(undef,4*N,4), Vector{Float64}(undef,4*N), Matrix{Float64}(undef,8,8),
# qr_ws, QROrmWs(zeros(4), qr_ws.τ))

# PeriodicMatrixEquations.pslyapd2!(Y1, X1, copy(Ad.M), copy(Qd.M), copy(Qd.M), Xt, QW, WORK2, ws_pschur(Ad.M))
# @test X1 ≈ X.M && Y1 ≈ Y.M


# n = 5; pa = 1; pc = 1;     
# Ad = 0.5*PeriodicArray(rand(Float64,n,n,pa),pa);
# q = rand(n,n,pc); [q[:,:,i] = q[:,:,i]'+q[:,:,i] for i in 1:pc];
# Qd = PeriodicArray(q,pc);

# X = prdlyap(Ad, Qd); 
# @test norm(Ad'*pmshift(X)*Ad-X+Qd) < 1.e-7 

# Y = pfdlyap(Ad, Qd) 
# @test norm(Ad*Y*Ad'-pmshift(Y)+Qd) < 1.e-7 

# p = lcm(pa,pc)
# X1 = zeros(n,n,p); Y1 = zeros(n,n,p); Xt = zeros(n,n); QW = zeros(n,n,p);
# PeriodicMatrixEquations.pslyapd!(X1, copy(Ad.M), copy(Qd.M), Xt, QW; adj = true) 
# PeriodicMatrixEquations.pslyapd!(Y1, copy(Ad.M), copy(Qd.M), Xt, QW; adj = false) 
# @test X1 ≈ X.M && Y1 ≈ Y.M

# T = Float64; N = p; 
# qr_ws = QRWs(zeros(8), zeros(4))
# #WORK1 = (Array{T,3}(undef, n, n, N), Array{T,3}(undef, n, n, N), Array{T,3}(undef, n, n, N), Matrix{T}(undef, n, n), Array{T,3}(undef,n,n,N), ws_pschur(Gt) )
# WORK2 = (Array{Float64,3}(undef,2,2,N), Array{Float64,3}(undef,4,4,N), Array{Float64,3}(undef,4,4,N),
# Matrix{Float64}(undef,4*N,4), Vector{Float64}(undef,4*N), Matrix{Float64}(undef,8,8),
# qr_ws, QROrmWs(zeros(4), qr_ws.τ))

# PeriodicMatrixEquations.pslyapd2!(Y1, X1, copy(Ad.M), copy(Qd.M), copy(Qd.M), Xt, QW, WORK2, ws_pschur(Ad.M))
# @test X1 ≈ X.M && Y1 ≈ Y.M


# #pseig(Ad)

# n = 5; pa = 3; pc = 1;     
# Ad = 0.5*PeriodicArray(rand(Float32,n,n,pa),pa);
# q = rand(n,n); q = q'+q;

# X = prdlyap(Ad, q); 
# @test norm(Ad'*pmshift(X)*Ad-X+q) < 1.e-7 

# X = pfdlyap(Ad, q) 
# @test norm(Ad*X*Ad'-pmshift(X)+q) < 1.e-7 



# end 

# @testset "pslyapd" begin


# n = 5; pa = 1; pc = 1; p = lcm(pa,pc)
# a = (1/(n*n*pa))*rand(n,n,pa); q = rand(n,n,pc); 
# [q[:,:,i] = q[:,:,i]'+q[:,:,i] for i in 1:pc];

# X = pslyapd(a, q; adj = true) 
# ia = mod.(0:p-1,pa).+1; ipx = mod.(0:p-1,p).+1;  
# ipc = mod.(0:p-1,pc).+1; ipx1 = mod.(ipx,p).+1;  

# rez = [ a[:,:,ia[i]]'*X[:,:,ipx1[i]]*a[:,:,ia[i]]-X[:,:,ipx[i]]+q[:,:,ipc[i]] for i in ipx]
# @test norm(rez) < 1.e-7 

# X = pslyapd(a, q; adj = false) 
# ia = mod.(0:p-1,pa).+1; ipx = mod.(0:p-1,p).+1;  
# ipc = mod.(0:p-1,pc).+1; ipx1 = mod.(ipx,p).+1;  

# rez = [ a[:,:,ia[i]]*X[:,:,ipx[i]]*a[:,:,ia[i]]'-X[:,:,ipx1[i]]+q[:,:,ipc[i]] for i in ipx]
# @test norm(rez) < 1.e-7 


# n = 5; pa = 5; pc = 10; p = lcm(pa,pc)
# a = (1/(n*n*pa))*rand(n,n,pa); q = rand(n,n,pc); 
# [q[:,:,i] = q[:,:,i]'+q[:,:,i] for i in 1:pc];

# X = pslyapd(a, q; adj = true) 
# ia = mod.(0:p-1,pa).+1; ipx = mod.(0:p-1,p).+1;  
# ipc = mod.(0:p-1,pc).+1; ipx1 = mod.(ipx,p).+1;  

# rez = [ a[:,:,ia[i]]'*X[:,:,ipx1[i]]*a[:,:,ia[i]]-X[:,:,ipx[i]]+q[:,:,ipc[i]] for i in ipx]
# @test norm(rez) < 1.e-7 

# X = pslyapd(a, q; adj = false) 
# ia = mod.(0:p-1,pa).+1; ipx = mod.(0:p-1,p).+1;  
# ipc = mod.(0:p-1,pc).+1; ipx1 = mod.(ipx,p).+1;  

# rez = [ a[:,:,ia[i]]*X[:,:,ipx[i]]*a[:,:,ia[i]]'-X[:,:,ipx1[i]]+q[:,:,ipc[i]] for i in ipx]
# @test norm(rez) < 1.e-7 

# X = pslyapdkr(a, q; adj = true) 
# ia = mod.(0:p-1,pa).+1; ipx = mod.(0:p-1,p).+1;  
# ipc = mod.(0:p-1,pc).+1; ipx1 = mod.(ipx,p).+1;  

# rez = [ a[:,:,ia[i]]'*X[:,:,ipx1[i]]*a[:,:,ia[i]]-X[:,:,ipx[i]]+q[:,:,ipc[i]] for i in ipx]
# @test norm(rez) < 1.e-7 

# X = pslyapdkr(a, q; adj = false) 
# ia = mod.(0:p-1,pa).+1; ipx = mod.(0:p-1,p).+1;  
# ipc = mod.(0:p-1,pc).+1; ipx1 = mod.(ipx,p).+1;  

# rez = [ a[:,:,ia[i]]*X[:,:,ipx[i]]*a[:,:,ia[i]]'-X[:,:,ipx1[i]]+q[:,:,ipc[i]] for i in ipx]
# @test norm(rez) < 1.e-7 



# # q = copy(qs);
# # X1 = pslyapdkr(a,q)
# # ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
# # rez = [ a[:,:,ip[i]]'*X1[:,:,ip1[i]]*a[:,:,ip[i]]-X1[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]
# # @test norm(rez) < 1.e-7  

# # KSCHUR = 1
# # q = copy(qs);
# # X = pdlyaps1!(KSCHUR, a, q; adj = true) 
# # ip = mod.(0:pc-1,p).+1; ip1 = mod.(ip,p).+1;  
# # ipc = mod.(0:pc-1,pc).+1; ipc1 = mod.(ipc,pc).+1;  

# # rez = [ a[:,:,ip[i]]'*X[:,:,ipc1[i]]*a[:,:,ip[i]]-X[:,:,ipc[i]]+qs[:,:,ipc[i]] for i in ipc]
# # @test norm(rez) < 1.e-7 

# # # q = copy(qs);
# # # X1 = pslyapdkr(a,q; adj = false)
# # # ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
# # # rez = [ a[:,:,ip[i]]*X1[:,:,ip[i]]*a[:,:,ip[i]]'-X1[:,:,ip1[i]]+qs[:,:,ip[i]] for i in ip]
# # # @test norm(rez) < 1.e-7  


# # q = copy(qs);
# # X = pdlyaps1!(KSCHUR, a, q; adj = false) 
# # rez = [ a[:,:,ip[i]]*X[:,:,ipc[i]]*a[:,:,ip[i]]'-X[:,:,ipc1[i]]+qs[:,:,ipc[i]] for i in ipc]
# # @test norm(rez) < 1.e-7  


# end # pdlyap

# @testset "pdlyaps1!" begin


# p = 1; n = 1; 
# a = rand(n,n,p); q = rand(n,n,p);
# KSCHUR = 1
# qs = copy(q)
# X = pdlyaps1!(KSCHUR, a, q; adj = true) 
# rez = a[:,:,1]'*X[:,:,1]*a[:,:,1]-X[:,:,1]+qs[:,:,1]   
# @test norm(rez) < 1.e-7   
# q = copy(qs)      
# X = pdlyaps1!(KSCHUR, a, q; adj = false) 
# rez = a[:,:,1]*X[:,:,1]*a[:,:,1]'-X[:,:,1]+qs[:,:,1]   
# @test norm(rez) < 1.e-7  

# X = copy(qs)
# pdlyaps!(KSCHUR, a, X; adj = true) 
# rez = a[:,:,1]'*X[:,:,1]*a[:,:,1]-X[:,:,1]+qs[:,:,1]   
# @test norm(rez) < 1.e-7   
# X = copy(qs)
# pdlyaps!(KSCHUR, a, X; adj = false) 
# rez = a[:,:,1]*X[:,:,1]*a[:,:,1]'-X[:,:,1]+qs[:,:,1]   
# @test norm(rez) < 1.e-7  


# p = 10; n = 1; 
# a = rand(n,n,p); q = rand(n,n,p);
# KSCHUR = 1
# qs = copy(q)
# X = pdlyaps1!(KSCHUR, a, q; adj = true) 
# ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
# rez = [ a[:,:,ip[i]]'*X[:,:,ip1[i]]*a[:,:,ip[i]]-X[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7   
# q = copy(qs)      
# X = pdlyaps1!(KSCHUR, a, q; adj = false) 
# rez = [ a[:,:,ip[i]]*X[:,:,ip[i]]*a[:,:,ip[i]]'-X[:,:,ip1[i]]+qs[:,:,ip[i]] for i in ip] 
# @test norm(rez) < 1.e-7  

# X1 = copy(qs)
# pdlyaps!(KSCHUR, a, X1; adj = true) 
# ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
# rez = [ a[:,:,ip[i]]'*X1[:,:,ip1[i]]*a[:,:,ip[i]]-X1[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7   
# X1 = copy(qs)     
# pdlyaps!(KSCHUR, a, X1; adj = false) 
# rez = [ a[:,:,ip[i]]*X1[:,:,ip[i]]*a[:,:,ip[i]]'-X1[:,:,ip1[i]]+qs[:,:,ip[i]] for i in ip] 
# @test norm(rez) < 1.e-7  


# p = 10; n = 2; 
# a = rand(n,n,p); q = rand(n,n,p);
# a[:,:,1] = 0.01*[1 -2;2 1]; [a[:,:,i] = triu(a[:,:,i]) for i in 2:p]; 
# [q[:,:,i] = q[:,:,i]'*q[:,:,i] for i in 1:p];
# KSCHUR = 1
# qs = copy(q)
# X = pdlyaps1!(KSCHUR, a, q; adj = true) 
# ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
# rez = [ a[:,:,ip[i]]'*X[:,:,ip1[i]]*a[:,:,ip[i]]-X[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7   
# q = copy(qs)      
# X = pdlyaps1!(KSCHUR, a, q; adj = false) 
# rez = [ a[:,:,ip[i]]*X[:,:,ip[i]]*a[:,:,ip[i]]'-X[:,:,ip1[i]]+qs[:,:,ip[i]] for i in ip] 
# @test norm(rez) < 1.e-7  

# X1 = copy(qs)
# pdlyaps!(KSCHUR, a, X1; adj = true) 
# ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
# rez = [ a[:,:,ip[i]]'*X1[:,:,ip1[i]]*a[:,:,ip[i]]-X1[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7   
# X1 = copy(qs)     
# pdlyaps!(KSCHUR, a, X1; adj = false) 
# rez = [ a[:,:,ip[i]]*X1[:,:,ip[i]]*a[:,:,ip[i]]'-X1[:,:,ip1[i]]+qs[:,:,ip[i]] for i in ip] 
# @test norm(rez) < 1.e-7  


# p = 2; n = 2; 
# a = rand(n,n,p); q = rand(n,n,p); x = rand(n,n,p);
# a[:,:,1] = 0.01*schur(rand(n,n)).T; [a[:,:,i] = triu(a[:,:,i]) for i in 2:p]; 
# [q[:,:,i] = q[:,:,i]'*q[:,:,i] for i in 1:p];

# # a[:,:,1] = [0.5 2;0 -0.5]; a[:,:,2] = [1 0;0 1.];
# # x[:,:,1] =  [1 3;3 1.]; x[:,:,2] =  [2 1;1 2.]; 
# # q[:,:,1] = -a[:,:,1]'*x[:,:,2]*a[:,:,1]+x[:,:,1];
# # q[:,:,2] = -a[:,:,2]'*x[:,:,1]*a[:,:,2]+x[:,:,2];
# # qs = copy(q);
# # ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
# # rez = [ a[:,:,ip[i]]'*x[:,:,ip1[i]]*a[:,:,ip[i]]-x[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]

# qs = copy(q);
# X1 = pslyapdkr(a,q)
# ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
# rez1 = [ a[:,:,ip[i]]'*X1[:,:,ip1[i]]*a[:,:,ip[i]]-X1[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]
# @test norm(rez1) < 1.e-7 

# KSCHUR = 1
# q = copy(qs);
# X = pdlyaps1!(KSCHUR, a, q; adj = true) 
# ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
# rez = [ a[:,:,ip[i]]'*X[:,:,ip1[i]]*a[:,:,ip[i]]-X[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7   
# q = copy(qs)      
# X = pdlyaps1!(KSCHUR, a, q; adj = false) 
# rez = [ a[:,:,ip[i]]*X[:,:,ip[i]]*a[:,:,ip[i]]'-X[:,:,ip1[i]]+qs[:,:,ip[i]] for i in ip] 
# @test norm(rez) < 1.e-7  

# X1 = copy(qs)
# pdlyaps!(KSCHUR, a, X1; adj = true) 
# ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
# rez = [ a[:,:,ip[i]]'*X1[:,:,ip1[i]]*a[:,:,ip[i]]-X1[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7   
# X1 = copy(qs)     
# pdlyaps!(KSCHUR, a, X1; adj = false) 
# rez = [ a[:,:,ip[i]]*X1[:,:,ip[i]]*a[:,:,ip[i]]'-X1[:,:,ip1[i]]+qs[:,:,ip[i]] for i in ip] 
# @test norm(rez) < 1.e-7  


# p = 5; n = 5; 
# a = rand(n,n,p); q = rand(n,n,p); x = rand(n,n,p);
# a[:,:,1] = 0.01*schur(rand(n,n)).T; [a[:,:,i] = triu(a[:,:,i]) for i in 2:p]; 
# [q[:,:,i] = q[:,:,i]'*q[:,:,i] for i in 1:p];

# qs = copy(q);
# X1 = pslyapdkr(a,q)
# ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
# rez = [ a[:,:,ip[i]]'*X1[:,:,ip1[i]]*a[:,:,ip[i]]-X1[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7  

# KSCHUR = 1
# q = copy(qs);
# X = pdlyaps1!(KSCHUR, a, q; adj = true) 
# rez = [ a[:,:,ip[i]]'*X[:,:,ip1[i]]*a[:,:,ip[i]]-X[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7 

# X1 = copy(qs)
# pdlyaps!(KSCHUR, a, X1; adj = true) 
# ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
# rez = [ a[:,:,ip[i]]'*X1[:,:,ip1[i]]*a[:,:,ip[i]]-X1[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7  

# q = copy(qs);
# X1 = pslyapdkr(a,q; adj = false)
# ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
# rez = [ a[:,:,ip[i]]*X1[:,:,ip[i]]*a[:,:,ip[i]]'-X1[:,:,ip1[i]]+qs[:,:,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7  


# q = copy(qs);
# X = pdlyaps1!(KSCHUR, a, q; adj = false) 
# rez = [ a[:,:,ip[i]]*X[:,:,ip[i]]*a[:,:,ip[i]]'-X[:,:,ip1[i]]+qs[:,:,ip[i]] for i in ip]
# @test norm(rez) < 1.e-7  

# X1 = copy(qs)     
# pdlyaps!(KSCHUR, a, X1; adj = false) 
# rez = [ a[:,:,ip[i]]*X1[:,:,ip[i]]*a[:,:,ip[i]]'-X1[:,:,ip1[i]]+qs[:,:,ip[i]] for i in ip] 
# @test norm(rez) < 1.e-7  


# p = 5; n = 5; pc = 10
# a = rand(n,n,p); q = rand(n,n,pc); x = rand(n,n,pc);
# a[:,:,1] = 0.01*schur(rand(n,n)).T; [a[:,:,i] = triu(a[:,:,i]) for i in 2:p]; 
# [q[:,:,i] = q[:,:,i]'*q[:,:,i] for i in 1:pc];
# qs = copy(q);

# # q = copy(qs);
# # X1 = pslyapdkr(a,q)
# # ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
# # rez = [ a[:,:,ip[i]]'*X1[:,:,ip1[i]]*a[:,:,ip[i]]-X1[:,:,ip[i]]+qs[:,:,ip[i]] for i in ip]
# # @test norm(rez) < 1.e-7  

# KSCHUR = 1
# q = copy(qs);
# X = pdlyaps1!(KSCHUR, a, q; adj = true) 
# ip = mod.(0:pc-1,p).+1; ip1 = mod.(ip,p).+1;  
# ipc = mod.(0:pc-1,pc).+1; ipc1 = mod.(ipc,pc).+1;  

# rez = [ a[:,:,ip[i]]'*X[:,:,ipc1[i]]*a[:,:,ip[i]]-X[:,:,ipc[i]]+qs[:,:,ipc[i]] for i in ipc]
# @test norm(rez) < 1.e-7 

# X1 = copy(qs)
# @time pdlyaps!(KSCHUR, a, X1; adj = true); 
# rez = [ a[:,:,ip[i]]'*X1[:,:,ipc1[i]]*a[:,:,ip[i]]-X1[:,:,ipc[i]]+qs[:,:,ipc[i]] for i in ipc]
# @test norm(rez) < 1.e-7  

# # q = copy(qs);
# # X1 = pslyapdkr(a,q; adj = false)
# # ip = mod.(0:p-1,p).+1; ip1 = mod.(ip,p).+1;  
# # rez = [ a[:,:,ip[i]]*X1[:,:,ip[i]]*a[:,:,ip[i]]'-X1[:,:,ip1[i]]+qs[:,:,ip[i]] for i in ip]
# # @test norm(rez) < 1.e-7  


# q = copy(qs);
# X = pdlyaps1!(KSCHUR, a, q; adj = false) 
# rez = [ a[:,:,ip[i]]*X[:,:,ipc[i]]*a[:,:,ip[i]]'-X[:,:,ipc1[i]]+qs[:,:,ipc[i]] for i in ipc]
# @test norm(rez) < 1.e-7  

# X1 = copy(qs)     
# pdlyaps!(KSCHUR, a, X1; adj = false) 
# rez = [ a[:,:,ip[i]]*X1[:,:,ipc[i]]*a[:,:,ip[i]]'-X1[:,:,ipc1[i]]+qs[:,:,ipc[i]] for i in ip] 
# @test norm(rez) < 1.e-7  

# for adj in (false,true)
#    @time pdlyaps!(KSCHUR, a, copy(qs); adj); 
#    @time pdlyaps1!(KSCHUR, a, copy(qs); adj); 
#    @time pdlyaps2!(KSCHUR, a, copy(qs); adj); 
#    @time pdlyaps3!(KSCHUR, a, copy(qs); adj); 
# end

# KSCHUR = 1
# p = 1; n = 5; pc = 1
# a = rand(n,n,p); q = rand(n,n,pc); x = rand(n,n,pc);
# a[:,:,1] = 0.01*schur(rand(n,n)).T; [a[:,:,i] = triu(a[:,:,i]) for i in 2:p]; 
# [q[:,:,i] = q[:,:,i]'*q[:,:,i] for i in 1:pc];
# qs = copy(q);
# @time pdlyaps3!(KSCHUR, a, copy(qs); adj = true)
# @time pdlyaps2!(KSCHUR, a, copy(qs); adj = true); 
# @time pdlyaps2!(KSCHUR, a, copy(qs); adj = true); 


# # benchmark
# # using BenchmarkTools
# # using MatrixEquations
# # p = 5; n = 400; pc = 5
# # p = 200; n = 8; pc = 200
# # a = 0.1*rand(n,n,p); q = rand(n,n,pc); x = rand(n,n,pc); [x[:,:,i] = x[:,:,i]'+x[:,:,i] for i in 1:pc];
# # a[:,:,1] = 0.01*schur(rand(n,n)).T; [a[:,:,i] = triu(a[:,:,i]) for i in 2:p]; 
# # ip = mod.(0:pc-1,p).+1; ip1 = mod.(ip,p).+1;  
# # ipc = mod.(0:pc-1,pc).+1; ipc1 = mod.(ipc,pc).+1;  
# # [q[:,:,i] = -a[:,:,ip[i]]'*x[:,:,ipc1[i]]*a[:,:,ip[i]]-x[:,:,ipc[i]] for i in ipc];
# # [q[:,:,i] = 0.5*(q[:,:,i]'+q[:,:,i]) for i in 1:pc];
# # qs = copy(q);

# # KSCHUR = 1
# # X = copy(qs);
# # pdlyaps1!(KSCHUR, a, X; adj = true) 
# # ip = mod.(0:pc-1,p).+1; ip1 = mod.(ip,p).+1;  
# # ipc = mod.(0:pc-1,pc).+1; ipc1 = mod.(ipc,pc).+1;  
# # rez = [ a[:,:,ip[i]]'*X[:,:,ipc1[i]]*a[:,:,ip[i]]-X[:,:,ipc[i]]+qs[:,:,ipc[i]] for i in ipc]
# # @test norm(rez) < 1.e-5 

# # X1 = copy(qs);
# # pdlyaps!(KSCHUR, a, X1; adj = true) 
# # rez = [ a[:,:,ip[i]]'*X1[:,:,ipc1[i]]*a[:,:,ip[i]]-X1[:,:,ipc[i]]+qs[:,:,ipc[i]] for i in ipc]
# # @test norm(rez) < 1.e-5  

# # X2 = copy(qs)
# # pdlyaps2!(KSCHUR, a, X2; adj = true) 
# # rez = [ a[:,:,ip[i]]'*X2[:,:,ipc1[i]]*a[:,:,ip[i]]-X2[:,:,ipc[i]]+qs[:,:,ipc[i]] for i in ipc]
# # @test norm(rez) < 1.e-5  


# # KSCHUR = 1
# # @btime pdlyaps!(KSCHUR, a, copy(qs); adj = true); 
# # @btime pdlyaps1!(KSCHUR, a, copy(qs); adj = true); 
# # @btime pdlyaps2!(KSCHUR, a, copy(qs); adj = true); 
# # @btime pdlyaps3!(KSCHUR, a, copy(qs); adj = true); 

# # @btime pdlyaps!(KSCHUR, a, copy(qs); adj = false); 
# # @btime pdlyaps1!(KSCHUR, a, copy(qs); adj = false); 
# # @btime pdlyaps2!(KSCHUR, a, copy(qs); adj = false); 
# # @btime pdlyaps3!(KSCHUR, a, copy(qs); adj = false); 

# p = 100; n = 4; pc = 100
# a = 0.1*rand(n,n,p); q = rand(n,n,pc); x = rand(n,n,pc); [x[:,:,i] = x[:,:,i]'+x[:,:,i] for i in 1:pc];
# a[:,:,1] = 0.01*schur(rand(n,n)).T; [a[:,:,i] = triu(a[:,:,i]) for i in 2:p]; 
# ip = mod.(0:pc-1,p).+1; ip1 = mod.(ip,p).+1;  
# ipc = mod.(0:pc-1,pc).+1; ipc1 = mod.(ipc,pc).+1;  
# [q[:,:,i] = -a[:,:,ip[i]]'*x[:,:,ipc1[i]]*a[:,:,ip[i]]-x[:,:,ipc[i]] for i in ipc];
# [q[:,:,i] = 0.5*(q[:,:,i]'+q[:,:,i]) for i in 1:pc];
# qs = copy(q);



# # q = copy(qs);
# # A1 = copy(a[:,:,1]); X1 = copy(q[:,:,1]);
# # lyapds!(A1, X1; adj = true)
# # rez1 = a[:,:,1]'*X1*a[:,:,1]- X1 +qs[:,:,1] 
# # @test norm(rez1) < 1.e-6 

# # @btime lyapds!(A1, copy(X1); adj = true); 


# end # pdlyaps1!


end # module