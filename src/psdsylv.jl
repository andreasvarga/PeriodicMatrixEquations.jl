for PM in (:PeriodicArray, :PeriodicMatrix)
    @eval begin
       function pdsylvc(A::$PM, B::$PM, C::$PM; rev::Bool = false, isgn = 1, fast::Bool = true) 
          A.Ts ≈ B.Ts ≈ C.Ts || error("A, B and C must have the same sampling time")
          abs(isgn) == 1 || throw(ArgumentError("only isgn = 1 or isgn = -1 allowed"))
          period = promote_period(A, B, C)
          na = rationalize(period/A.period).num
          K = na*A.nperiod*A.dperiod
          X = pssylvdc(A.M, B.M, C.M; rev, isgn, fast)
          p = lcm(length(A),length(B),length(C))
          return $PM(X, period; nperiod = div(K,p))
       end
    end
end
function pdsylvc(A::PM, B::PM, C::PM; kwargs...) where {PM <: SwitchingPeriodicMatrix}
   X = pdsylvc(convert(PeriodicMatrix,A),convert(PeriodicMatrix,B),convert(PeriodicMatrix,C); kwargs...)
   return convert(SwitchingPeriodicMatrix,X)
end
function pdsylvc(A::PM, B::PM, C::PM; kwargs...) where {PM <: SwitchingPeriodicArray}
   X = pdsylvc(convert(PeriodicArray,A),convert(PeriodicArray,B),convert(PeriodicArray,C); kwargs...)
   return convert(SwitchingPeriodicArray,X)
end

"""
    pdsylvc(A, B, C; rev = false, isgn = 1, fast = true) -> X

Solve the periodic discrete-time Sylvester equation of continuous-time flavour

    A*X + isgn*σX*B = C  for rev = false 

or 

    isgn*A*σX + X*B = C  for rev = true,
    
where `σ` is the forward shift operator `σX(i) = X(i+1)` and `abs(isgn) = 1`. 

The periodic matrices `A`, `B` and `C` must have the same type and commensurate periods. 
The resulting periodic solution `X` has the period 
set to the least common commensurate period of `A`, `B` and `C` and the number of subperiods
is adjusted accordingly. 

The periodic discrete analog of the Bartels-Stewart method based on the periodic Schur form
of the periodic matrices `A` and `B` is employed (see Appendix II of [1]). 
If `fast = true`, the QR factorization of bordered-almost-block-diagonal (_BABD_) matrix
algorithm of [2] is employed to solve periodic Sylvester equations up to order 2. 
This option is more appropriate for large periods. If `fast = false`, the QR factorization of the cyclic Kronecker form 
for the periodic Sylvester operator is used to to solve periodic Sylvester equations up to order 2.

For the existence of a solution `A` and `B` must not have characteristic multipliers `α` and `β` such that
`α +isgn*β = 0`.

_Reference:_

[1] A. Varga. Robust and minimum norm pole assignment with periodic state feedback. 
              IEEE Trans. on Automatic Control, vol. 45, pp. 1017-1022, 2000.

[2] R. Granat, B. Kågström, and D. Kressner,  Computing periodic deflating subspaces associated with a specified 
    set of eigenvalues. BIT Numerical Mathematics vol. 47, pp. 763–791,  2007.             
"""
pdsylvc(A::PeriodicArray, B::PeriodicArray, C::PeriodicArray) 

for PM in (:PeriodicArray, :PeriodicMatrix, :SwitchingPeriodicMatrix, :SwitchingPeriodicArray)
   @eval begin
      function prdsylvc(A::$PM, B::$PM, C::$PM; isgn = 1, fast = true) 
         pdsylvc(A, B, C; rev = true, isgn, fast)
      end
      function pfdsylvc(A::$PM, B::$PM, C::$PM; isgn = 1, fast = true) 
         pdsylvc(A, B, C; rev = false, isgn, fast)
      end
   end
end
"""
    prdsylvc(A, B, C; isgn = 1, fast = true) -> X

Solve the reverse-time periodic discrete-time Sylvester equation of continuous-time flavour

    isgn*A*σX + X*B = C ,

where `σ` is the forward shift operator `σX(i) = X(i+1)` and `abs(isgn) = 1`.                 

The periodic matrices `A`, `B` and `C` must have the same type and commensurate periods. 
The resulting periodic solution `X` has the period 
set to the least common commensurate period of `A`, `B` and `C` and the number of subperiods
is adjusted accordingly. 
"""
prdsylvc(A::PeriodicArray, B::PeriodicArray, C::PeriodicArray) 
"""
    pfdsylvc(A, B, C; isgn = 1, fast = true) -> X

Solve the forward-time periodic discrete-time Sylvester equation of continuous-time flavour

    A*X + isgn*σX*B = C , 
    
where `σ` is the forward shift operator `σX(i) = X(i+1)` and `abs(isgn) = 1`. 

The periodic matrices `A`, `B` and `C` must have the same type and commensurate periods. 
The resulting periodic solution `X` has the period 
set to the least common commensurate period of `A`, `B` and `C` and the number of subperiods
is adjusted accordingly. 
"""
pfdsylvc(A::PeriodicArray, B::PeriodicArray, C::PeriodicArray) 
"""
    pssylvdc(A, B, C; rev = false, isgn = 1, fast = true) -> X

Solve the periodic discrete-time Sylvester equation of continuous-time flavour.

For the square `n`-th order periodic matrices `A(i)`, `i = 1, ..., pa`, `B(i)`, `i = 1, ..., pb`  and 
`C(i)`, `i = 1, ..., pc`  of periods `pa`, `pb` and `pc`, respectively, 
the periodic solution `X(i)`, `i = 1, ..., p` of period `p = lcm(pa,pb,pc)` of the 
periodic Sylvester equation is computed:  

    A(i)*X(i) + isgn*X(i+1)*B(i) = C(i), i = 1, ..., p     for `rev = false`; 

    isgn*A(i)*X(i+1) + X(i)*B(i) = C(i), i = 1, ..., p     for `rev = true`.   

The periodic matrices `A`, `B` and `C` are stored in the `m×m×pa`, `n×n×pb` and `m×n×pc` 3-dimensional 
arrays `A`, `B` and `C`, respectively, and `X` results as a `m×n×p` 3-dimensional array.  

Alternatively, the periodic matrices `A`, `B` and `C` can be stored in the  `pa`-, `pb`-  and `pc`-dimensional
vectors of matrices `A`, `B` and `C`, respectively, and `X` results as a `p`-dimensional vector of matrices.

The periodic discrete analog of the Bartels-Stewart method based on the periodic Schur form
of the periodic matrices `A` and `B` is employed (see Appendix II of [1]). 
If `fast = true`, the QR factorization of bordered-almost-block-diagonal (_BABD_) matrix
algorithm of [2] is employed to solve periodic Sylvester equations up to order 2. 
This option is more appropriate for large periods. If `fast = false`, the QR factorization of the cyclic Kronecker form 
for the periodic Sylvester operator is used to to solve periodic Sylvester equations up to order 2.

For the existence of a solution `A` and `B` must not have characteristic multipliers `α` and `β` such that
`α +isgn*β = 0`.

_Reference:_

[1] A. Varga. Robust and minimum norm pole assignment with periodic state feedback. 
              IEEE Trans. on Automatic Control, vol. 45, pp. 1017-1022, 2000.

[2] R. Granat, B. Kågström, and D. Kressner,  Computing periodic deflating subspaces associated with a specified 
    set of eigenvalues. BIT Numerical Mathematics vol. 47, pp. 763–791,  2007.             
"""
function pssylvdc(A::AbstractArray{T1, 3}, B::AbstractArray{T2, 3}, C::AbstractArray{T3, 3}; rev::Bool = false, isgn = 1, fast = true) where {T1, T2, T3}
   # if rev
   #    Y = pssylvdc(PermutedDimsArray(B,[2,1,3]),PermutedDimsArray(A,[2,1,3]),PermutedDimsArray(C,[2,1,3]); rev = false, isgn)
   #    return permutedims(Y,[2,1,3])
   # end
   m = LinearAlgebra.checksquare(A[:,:,1])
   n = LinearAlgebra.checksquare(B[:,:,1])
   pa = size(A,3)
   pb = size(B,3)
   pc = size(C,3)
   (size(C[:,:,1],1) == m && size(C[:,:,1],2) == n) ||
      throw(DimensionMismatch("C must be an $m x $n x $pc array"))
   p = lcm(pa,pb,pc)

   T = promote_type(T1, T2, T3)
   T <: BlasFloat  || (T = promote_type(Float64,T))
   A1 = T1 == T ? A : A1 = convert(Array{T,3},A)
   B1 = T2 == T ? B : B1 = convert(Array{T,3},B)
   C1 = T3 == T ? C : C1 = convert(Array{T,3},C)

   # Reduce A and B to periodic Schur forms AS and BS, such that:
   # AS = Q'*A*σQ and BS = Z'*B*σZ if rev = true
   # AS = σQ'*A*Q and BS = σZ'*B*Z if rev = false
   AS, Q, _, KSCHURA = PeriodicMatrices.pschur(A1, rev = !rev)
   BS, Z, _, KSCHURB = PeriodicMatrices.pschur(B1, rev = !rev)
     
   X = Array{T,3}(undef, m, n, p)
   Xw = Array{T,2}(undef, m, n)   
   if rev
      #X = Q'*C*σZ
      for i = 1:p
         ia = mod(i-1,pa)+1
         ic = mod(i-1,pc)+1
         ib1 = mod(i,pb)+1
         mul!(Xw,view(C1,:,:,ic),view(Z,:,:,ib1))  
         mul!(view(X,:,:,i),view(Q,:,:,ia)',Xw)
     end
   else
      #X = σQ'*C*Z
      for i = 1:p
         ib = mod(i-1,pb)+1
         ic = mod(i-1,pc)+1
         ia1 = mod(i,pa)+1
         mul!(Xw,view(C1,:,:,ic),view(Z,:,:,ib))  
         mul!(view(X,:,:,i),view(Q,:,:,ia1)',Xw)
     end
   end

   # solve AσX + isgn*XB = C if rev = true or AX + isgn*σXB = C if rev = false
   pdsylvcs!(KSCHURA, AS, KSCHURB, BS, X; rev, isgn, fast)

   #X <- Q*X*Z'
   for i = 1:p
       ia = mod(i-1,pa)+1
       ib = mod(i-1,pb)+1
       mul!(Xw,view(X,:,:,i),view(Z,:,:,ib)')  
       mul!(view(X,:,:,i),view(Q,:,:,ia),Xw)
   end
   return X
end
function pssylvdc(A::AbstractVector{Matrix{T1}}, B::AbstractVector{Matrix{T2}}, C::AbstractVector{Matrix{T3}}; rev::Bool = false, isgn = 1, fast = true) where {T1, T2, T3}
   pa = length(A) 
   pb = length(B) 
   pc = length(C)
   ma, na = size.(A,1), size.(A,2) 
   mb, nb = size.(B,1), size.(B,2) 
   mc, nc = size.(C,1), size.(C,2) 
   p = lcm(pa,pb,pc)
   if rev
      all(ma .== view(na, mod.(pa-1:pa+pa-2,pa).+1)) || 
        error("the number of columns of A[i] must be equal to the number of rows of A[i+1]")         
      all(mb .== view(nb, mod.(pb-1:pb+pb-2,pb).+1)) || 
        error("the number of columns of B[i] must be equal to the number of rows of B[i+1]")
   else
      all(ma .== view(na,mod.(1:pa,pa).+1)) || 
        error("the number of columns of A[i+1] must be equal to the number of rows of A[i]")
      all(mb .== view(nb,mod.(1:pb,pb).+1)) || 
        error("the number of columns of B[i+1] must be equal to the number of rows of B[i]")
   end
   all([ma[mod(i-1,pa)+1] == mc[mod(i-1,pc)+1]  for i in 1:p]) ||
           throw(DimensionMismatch("incompatible dimensions between A and C"))
   all([nb[mod(i-1,pb)+1] == nc[mod(i-1,pc)+1] for i in 1:p]) ||
           throw(DimensionMismatch("incompatible dimensions between B and C"))

   m = maximum(na)
   n = maximum(nb)

   T = promote_type(T1, T2, T3)
   T <: BlasFloat  || (T = promote_type(Float64,T))
   A1 = zeros(T, m, m, pa)
   B1 = zeros(T, n, n, pb)
   C1 = zeros(T, m, n, pc)
   [copyto!(view(A1,1:ma[i],1:na[i],i), T.(A[i])) for i in 1:pa]
   [copyto!(view(B1,1:mb[i],1:nb[i],i), T.(B[i])) for i in 1:pb]
   [copyto!(view(C1,1:mc[i],1:nc[i],i), T.(C[i])) for i in 1:pc] 

   # if rev
   #    Y = pssylvdc(PermutedDimsArray(B1,[2,1,3]),PermutedDimsArray(A1,[2,1,3]),PermutedDimsArray(C1,[2,1,3]); rev = false, isgn)
   #    X = permutedims(Y,[2,1,3])
   #    return [X[1:mc[mod(i-1,pc)+1],1:nb[mod(i-1,pb)+1],i] for i in 1:p] 
   # end

   # Reduce A and B to periodic Schur forms AS and BS, such that:
   # AS = Q'*A*σQ and BS = Z'*B*σZ if rev = true
   # AS = σQ'*A*Q and BS = σZ'*B*Z if rev = false
   AS, Q, _, KSCHURA = PeriodicMatrices.pschur(A1, rev = !rev)
   BS, Z, _, KSCHURB = PeriodicMatrices.pschur(B1, rev = !rev)
     
   #X = σQ'*C*Z
   X = Array{T,3}(undef, m, n, p)
   Xw = Array{T,2}(undef, m, n)   
   if rev
      #X = Q'*C*σZ
      for i = 1:p
         ia = mod(i-1,pa)+1
         ic = mod(i-1,pc)+1
         ib1 = mod(i,pb)+1
         mul!(Xw,view(C1,:,:,ic),view(Z,:,:,ib1))  
         mul!(view(X,:,:,i),view(Q,:,:,ia)',Xw)
     end
   else
      #X = σQ'*C*Z
      for i = 1:p
         ib = mod(i-1,pb)+1
         ic = mod(i-1,pc)+1
         ia1 = mod(i,pa)+1
         mul!(Xw,view(C1,:,:,ic),view(Z,:,:,ib))  
         mul!(view(X,:,:,i),view(Q,:,:,ia1)',Xw)
     end
   end

   # solve isgn*A*σX + X*B = C if rev = true or A*X + isgn*σX*B = C if rev = false
   pdsylvcs!(KSCHURA, AS, KSCHURB, BS, X; rev, isgn, fast)

   #X <- Q*X*Z'
   for i = 1:p
       ia = mod(i-1,pa)+1
       ib = mod(i-1,pb)+1
       mul!(Xw,view(X,:,:,i),view(Z,:,:,ib)')  
       mul!(view(X,:,:,i),view(Q,:,:,ia),Xw)
   end
   return rev ? [X[1:ma[mod(i-1,pa)+1],1:mb[mod(i-1,pb)+1],i] for i in 1:p] : [X[1:na[mod(i-1,pa)+1],1:nc[mod(i-1,pc)+1],i] for i in 1:p]  
end
"""
     pssylvdckr(A, B, C; adj = true, isgn = 1) -> X

For the periodic matrices `A`, `B` and `C` compute the periodic matrix `X`, which satisfies  
the periodic discrete-time Sylvester matrix equation of continuous-time flavour

      isgn*A(i)*X(i+1) + X(i)*B(i) = C(i), if rev = true,

or 

      A(i)*X(i) + isgn*X(i+1)*B(i) =  C(i), if rev = false, 

where `abs(isgn) = 1`. The periodic matrices `A`, `B` and `C` have periods `pa`, `pb` and `pc`, respectively, 
and are stored as either 3-dimensional arrays or as vectors of matrices.  
The resulting periodic matrix `X` has period `p = LCM(pa,pb,pc)` and is stored correspondingly. 

For the solvability of the above equations the following conditions must hold: if `rev = false`, 
the matrix products `A(p)*A(p-1)...A(1)` and `B(p)*B(p-1)*...B(1)` must not have eigenvalues `α` and `β` such that
`α +(-isgn)^p*β = 0`, while if `rev = true`, the matrix products `A(1)*A(2)...A(p)` and `B(1)*B(2)*...B(p)` 
must not have eigenvalues `α` and `β` such that `α +(-isgn)^p*β = 0`.              

The Kronecker product expansion of equations is employed and therefore 
this function is not recommended for large order matrices or large periods.
"""
function pssylvdckr(A::AbstractArray{T1, 3}, B::AbstractArray{T2, 3}, C::AbstractArray{T3, 3}; rev = false, isgn = 1) where {T1, T2, T3}
   m, n, pc = size(C)
   m == LinearAlgebra.checksquare(A[:,:,1]) 
   n == LinearAlgebra.checksquare(B[:,:,1]) 
   pa = size(A,3)
   pb = size(B,3)
   p = lcm(pa,pb,pc)
   if p == 1 
      R = rev ? kron(isgn*I(n),A[:,:,1])+kron(transpose(B[:,:,1]),I(m)) : kron(I(n),A[:,:,1])+kron(transpose(B[:,:,1]),isgn*I(m))
      return reshape(R \ vec(C[:,:,1]),m,n,1)
   end
   mn = m*n
   N = p*mn
   T = promote_type(T1,T2,T3)
   R = zeros(T, N, N)
   if rev
      Im = T.(I(m))
      In = T.(isgn*I(n))
      copyto!(view(R,1:mn,N-mn+1:N),kron(B[:,:,pb]',Im))
      copyto!(view(R,1:mn,1:mn),kron(In,A[:,:,pa])) 
      i1 = mn+1; j1 = 1 
      for i = 1:p-1        
          ia = mod(i-1,pa)+1
          ib = mod(i-1,pb)+1
          i2 = i1+mn-1
          j2 = j1+mn-1
          copyto!(view(R,i1:i2,j1:j2),kron(B[:,:,ib]',Im)) 
          copyto!(view(R,i1:i2,j1+mn:j2+mn),kron(In,A[:,:,ia])) 
          i1 = i2+1
          j1 = j2+1
      end
   else
      Im = T.(isgn*I(m))
      In = T.(I(n))
      copyto!(view(R,1:mn,N-mn+1:N),kron(In,A[:,:,pa])) 
      copyto!(view(R,1:mn,1:mn),kron(B[:,:,pb]',Im)) 
      i1 = mn+1; j1 = 1 
      for i = 1:p-1        
          ia = mod(i-1,pa)+1
          ib = mod(i-1,pb)+1
          i2 = i1+mn-1
          j2 = j1+mn-1
          copyto!(view(R,i1:i2,j1:j2),kron(In,A[:,:,ia])) 
          copyto!(view(R,i1:i2,j1+mn:j2+mn),kron(B[:,:,ib]',Im)) 
          i1 = i2+1
          j1 = j2+1
      end
   end
   indc = [pc;mod.(0:p-2,pc).+1]
   return reshape(R \ (C[:,:,indc][:]), m, n, p)
end
function pssylvdckr2(Rw::Vector{T}, A::AbstractArray{T,3}, B::AbstractArray{T,3}, C::AbstractArray{T,3}; rev = false, isgn = 1) where {T}
   # version intended for solving periodic Sylvester equations of orders up to 2. 
   p = size(C,3)
   m = LinearAlgebra.checksquare(A[:,:,1]) 
   n = LinearAlgebra.checksquare(B[:,:,1]) 
   if p == 1 
      R = rev ? kron(isgn*I(n),A[:,:,1])+kron(transpose(B[:,:,1]),I(m)) : kron(I(n),A[:,:,1])+kron(transpose(B[:,:,1]),isgn*I(m))
      return reshape(R \ vec(C[:,:,1]),m,n,1)
   end
   pa = size(A,3)
   pb = size(B,3)
   mn = m*n
   N = p*mn
   R = reshape(view(Rw,1:N*N),N,N)
   R = zeros(T, N, N)
   if rev
      Im = T.(I(m))
      In = T.(isgn*I(n))
      copyto!(view(R,1:mn,N-mn+1:N),kron(transpose(B[:,:,pb]),Im))
      copyto!(view(R,1:mn,1:mn),kron(In,A[:,:,pa])) 
      i1 = mn+1; j1 = 1 
      for i = 1:p-1        
          ia = mod(i-1,pa)+1
          ib = mod(i-1,pb)+1
          i2 = i1+mn-1
          j2 = j1+mn-1
          copyto!(view(R,i1:i2,j1:j2),kron(transpose(B[:,:,ib]),Im)) 
          copyto!(view(R,i1:i2,j1+mn:j2+mn),kron(In,A[:,:,ia])) 
          i1 = i2+1
          j1 = j2+1
      end
   else
      Im = T.(isgn*I(m))
      In = T.(I(n))
      copyto!(view(R,1:mn,N-mn+1:N),kron(In,A[:,:,pa])) 
      copyto!(view(R,1:mn,1:mn),kron(transpose(B[:,:,pb]),Im)) 
      i1 = mn+1; j1 = 1 
      for i = 1:p-1        
          ia = mod(i-1,pa)+1
          ib = mod(i-1,pb)+1
          i2 = i1+mn-1
          j2 = j1+mn-1
          copyto!(view(R,i1:i2,j1:j2),kron(In,A[:,:,ia])) 
          copyto!(view(R,i1:i2,j1+mn:j2+mn),kron(transpose(B[:,:,ib]),Im)) 
          i1 = i2+1
          j1 = j2+1
      end
   end
   indc = [p;1:p-1]
   return reshape(R \ (C[:,:,indc][:]), m, n, p)
end

function pssylvdckr(A::AbstractVector{Matrix{T1}}, B::AbstractVector{Matrix{T2}}, C::AbstractVector{Matrix{T3}}; rev = false, isgn = 1) where {T1, T2, T3}
   pa = length(A) 
   pb = length(B) 
   pc = length(C)
   ma, na = size.(A,1), size.(A,2) 
   mb, nb = size.(B,1), size.(B,2) 
   mc, nc = size.(C,1), size.(C,2)  
   p = lcm(pa,pb,pc)
   if rev
      all(ma .== view(na, mod.(pa-1:pa+pa-2,pa).+1)) || 
        error("the number of columns of A[i] must be equal to the number of rows of A[i+1]")         
      all(mb .== view(nb, mod.(pb-1:pb+pb-2,pb).+1)) || 
        error("the number of columns of B[i] must be equal to the number of rows of B[i+1]")
   else
      all(ma .== view(na,mod.(1:pa,pa).+1)) || 
        error("the number of columns of A[i+1] must be equal to the number of rows of A[i]")
      all(mb .== view(nb,mod.(1:pb,pb).+1)) || 
        error("the number of columns of B[i+1] must be equal to the number of rows of B[i]")
   end
   all([ma[mod(i-1,pa)+1] == mc[mod(i-1,pc)+1]  for i in 1:p]) ||
           throw(DimensionMismatch("incompatible dimensions between A and C"))
   all([nb[mod(i-1,pb)+1] == nc[mod(i-1,pc)+1] for i in 1:p]) ||
           throw(DimensionMismatch("incompatible dimensions between B and C"))
   
   T = promote_type(T1, T2, T3)
   mnc =  similar([1],p)
   mnx =  similar([1],p)
   if rev
      for i = 1:p
          ib = mod(i-1,pb)+1
          ic = mod(i-1,pc)+1
          mnc[i] = mc[ic]*nc[ic]
          mnx[i] = mc[ic]*mb[ib]
      end
   else
      for i = 1:p
          ia = mod(i-1,pa)+1
          ic = mod(i-1,pc)+1
          mnc[i] = mc[ic]*nc[ic]
          mnx[i] = na[ia]*nc[ic]
      end
   end
   N1 = sum(mnc)
   N2 = sum(mnx)
   R = zeros(T, N1, N2)
   Y = zeros(T, N1)
   
   if rev 
      Im = T.(I(mc[pc]))
      In = T.(isgn*I(nc[pc]))
      copyto!(view(R,1:mnc[p],1:mnx[1]),kron(In,A[pa])) 
      copyto!(view(R,1:mnc[p],N2-mnx[p]+1:N2),kron(transpose(B[pb]),Im)) 
      copyto!(view(Y,1:mnc[p]),C[pc][:]) 
      i1 = mnc[p]+1; j1 = 1 
      for i = 1:p-1      
          ia = mod(i-1,pa)+1
          ib = mod(i-1,pb)+1
          ic = mod(i-1,pc)+1
          i2 = i1+mnc[i]-1
          j2 = j1+mnx[i]-1
          Im = T.(I(mc[ic]))
          In = T.(isgn*I(nc[ic]))
          copyto!(view(R,i1:i2,j1:j2),kron(transpose(B[ib]),Im)) 
          copyto!(view(R,i1:i2,j1+mnx[i]:j2+mnx[i+1]),kron(In,A[ia])) 
          copyto!(view(Y,i1:i2),C[ic][:]) 
          i1 = i2+1
          j1 = j2+1
      end
      if N1 == N2
         ldiv!(qr!(R),Y)
      else
         Y = R\Y
      end
      z = Vector{Matrix{T}}(undef,0)
      i1 = 1
      for i = 1:p 
          ib = mod(i-1,pb)+1
          ic = mod(i-1,pc)+1
          i2 = i1+mc[ic]*mb[ib]-1
          push!(z,reshape(view(Y,i1:i2),mc[ic],mb[ib]))
          i1 = i2+1
      end
      return z
   else
      Im = T.(isgn*I(mc[pc]))
      In = T.(I(nc[pc]))
      copyto!(view(R,1:mnc[pc],N2-mnx[p]+1:N2),kron(In,A[pa])) 
      copyto!(view(R,1:mnc[pc],1:mnx[1]),kron(transpose(B[pb]),Im)) 
      copyto!(view(Y,1:mnc[pc]),C[pc][:]) 
      i1 = mnc[pc]+1; j1 = 1 
      for i = 1:p-1     
          ia = mod(i-1,pa)+1
          ib = mod(i-1,pb)+1
          ic = mod(i-1,pc)+1
          i2 = i1+mnc[ic]-1
          j2 = j1+mnx[i]-1
          Im = T.(isgn*I(mc[ic]))
          In = T.(I(nc[ic]))
          copyto!(view(R,i1:i2,j1:j2),kron(In,A[ia])) 
          copyto!(view(R,i1:i2,j1+mnx[i]:j2+mnx[i+1]),kron(transpose(B[ib]),Im)) 
          copyto!(view(Y,i1:i2),C[ic][:]) 
          i1 = i2+1
          j1 = j2+1
      end
      if N1 == N2
         ldiv!(qr!(R),Y)
      else
         Y = R\Y
      end
      z = Vector{Matrix{T}}(undef,0)
      i1 = 1
      for i = 1:p 
          ia = mod(i-1,pa)+1
          ic = mod(i-1,pc)+1
          i2 = i1+na[ia]*nc[ic]-1
          push!(z,reshape(view(Y,i1:i2),na[ia],nc[ic]))
          i1 = i2+1
      end
      return z
   end
end  

function pdsylvcs!(KSCHURA::Int, A::AbstractArray{T1,3}, KSCHURB::Int, B::AbstractArray{T1,3}, C::AbstractArray{T1,3}; isgn = 1, rev = false, fast = true) where {T1<:BlasReal}
   # Standard solver for A and B in a periodic Schur forms, with structure exploiting solution of
   # the underlying 2x2 periodic Sylvester equations. 
   m, n, p = size(C)
   ma, na, pa = size(A)
   mb, nb, pb = size(B)
   ma == na || throw(ArgumentError("A must have equal first dimensions"))
   mb == nb || throw(ArgumentError("B must have equal first dimensions"))
   (m == ma && n == nb) ||
      throw(DimensionMismatch("C must be an $ma x $nb x $pc array"))
   rem(p,pa) == 0 || error("the period of C must be an integer multiple of A")
   rem(p,pb) == 0 || error("the period of C must be an integer multiple of B")
   (KSCHURA <= 0 || KSCHURA > pa ) && 
         error("KSCHURA has a value $KSCHURA, which is inconsistent with A ")
   (KSCHURB <= 0 || KSCHURB > pb ) && 
         error("KSCHURB has a value $KSCHURB, which is inconsistent with B ")

   if p == 1   
      if rev
         sylvcs!(isgn*view(A,:,:,1), view(B,:,:,1), view(C,:,:,1))
      else
         sylvcs!(view(A,:,:,1), view(B,:,:,1), view(C,:,:,1), isgn)
      end
      return C[:,:,:]
   end
   ONE = one(T1)

   # allocate cache for 2x2 periodic Sylvester solver
   # G = Array{T1,3}(undef,2,2,pc)
   # WUSD = Array{Float64,3}(undef,4,4,pc)
   # WUD = Array{Float64,3}(undef,4,4,pc)
   # WUL = Matrix{Float64}(undef,4*pc,4)
   # WY = Vector{Float64}(undef,4*pc)
   # W = Matrix{Float64}(undef,8,8)
   # qr_ws = QRWs(zeros(8), zeros(4))
   # ormqr_ws = QROrmWs(zeros(4), qr_ws.τ)   

   # determine the dimensions of the diagonal blocks of real Schur form
   ba, pa1 = MatrixEquations.sfstruct(A[:,:,KSCHURA])
   bb, pb1 = MatrixEquations.sfstruct(B[:,:,KSCHURB])
   # """
   # The (K,L)th block of X(i) is determined starting from
   # bottom-left corner column by column by

   #       A(i)[K,K]*X(i)[K,L] + isgn*X(i+1)[K,L]*B(i)[L,L] = C(i)[K,L] - R(i)[K,L]

   # where
   #                                    M                               L-1
   #     if rev = false: R(i)[K,L] =   SUM (A(i)[K,J]*X(i)[J,L]) + isgn*SUM (X(i+1)[K,II]*B(i)[II,L])
   #                                  J=K+1                             II=1
   #
   #                                       M                             L-1
   #     if rev = true:  R(i)[K,L] = isgn*SUM (A(i)[K,J]*X(i+1)[J,L]) + SUM (X(i)[K,II]*B(i)[II,L])
   #                                     J=K+1                           II=1
   #
   # """
   fast || (Rw = Vector{T1}(undef,16*p*p))
   j = 1
   for ll = 1:pb1
       dl = bb[ll]
       il1 = 1:j-1
       l = j:j+dl-1
       i = m
       for kk = pa1:-1:1
           dk = ba[kk]
           k = i-dk+1:i
           y = view(C,k,l,1:p)
           if kk < pa1
              ir = i+1:m
              if rev
                 for ii = 1:p
                     ia = mod(ii-1,pa)+1
                     ii1 = ii == p ? 1 : ii+1
                     mul!(view(y,:,:,ii),view(A,k,ir,ia),view(C,ir,l,ii1),-isgn*ONE,ONE)
                 end
              else
                 for ii = 1:p
                     ia = mod(ii-1,pa)+1
                     mul!(view(y,:,:,ii),view(A,k,ir,ia),view(C,ir,l,ii),-ONE,ONE)
                 end
               end
           end
           if ll > 1
              if rev
                 for ii = 1:p
                     ib = mod(ii-1,pb)+1
                     mul!(view(y,:,:,ii),view(C,k,il1,ii),view(B,il1,l,ib),-ONE,ONE)
                 end
              else
                 for ii = 1:p
                     ib = mod(ii-1,pb)+1
                     ii1 = mod(ii,p)+1
                     mul!(view(y,:,:,ii),view(C,k,il1,ii1),view(B,il1,l,ib),-isgn*ONE,ONE)
                 end
              end
           end
           if fast
              C[k,l,1:p] = psylsolve2(view(A,k,k,:), view(B,l,l,:), y; rev, isgn)
           else
              C[k,l,1:p] = pssylvdckr2(Rw, view(A,k,k,:), view(B,l,l,:), y; rev, isgn)
           end
           i -= dk
       end
       j += dl
   end
   return #C[:,:,:]
end

function psylsolve2(A::AbstractArray{T,3}, B::AbstractArray{T,3}, C::AbstractArray{T,3}; rev = false, isgn = 1) where {T}
   # This function is adapted from the function _psylsolve in the PeriodicSchurDecompositions.jl package                 
   p = size(C,3)
   p1 = LinearAlgebra.checksquare(A[:,:,1])
   p2 = LinearAlgebra.checksquare(B[:,:,1])
   if p == 1 
      R = rev ? kron(isgn*I(p2),A[:,:,1])+kron(transpose(B[:,:,1]),I(p1)) : kron(I(p2),A[:,:,1])+kron(B[:,:,1]',isgn*I(p1))
      return reshape(R \ vec(C[:,:,1]),p1,p2,1)
   end
   pa = size(A,3)
   pb = size(B,3)
   if rev
      eye1 = T.(I(p1))
      eye2 = T.(isgn*I(p2))
      Zl = [kron(transpose(B[:,:,1]), eye1)]
      Zd = [kron(eye2, A[:,:,pa])]
      for k in 1:(p - 1)
          ib1 = mod(k,pb)+1
          ia = mod(k-1,pa)+1
          push!(Zl, kron(transpose(B[:,:,ib1]), eye1))
          push!(Zd, kron(eye2, A[:,:,ia]))
      end
   else
      eye1 = T.(isgn*I(p1))
      eye2 = T.(I(p2))
      Zd = [kron(transpose(B[:,:,pb]), eye1)]
      Zl = [kron(eye2, A[:,:,1])]
      for k in 1:(p - 1)
          ib = mod(k-1,pb)+1
          ia1 = mod(k,pa)+1
          push!(Zd, kron(transpose(B[:,:,ib]), eye1))
          push!(Zl, kron(eye2, A[:,:,ia1]))
      end
   end
   indc = [p; 1:p-1]
   y = C[:,:,indc][:]
   R, Zu, Zr, _ =  PeriodicSchurDecompositions._babd_qr!(Zd, Zl, y)
   for r in R
         PeriodicSchurDecompositions._checkqr(r)
   end
   Xv =  PeriodicSchurDecompositions._babd_solve!(R, Zu, Zr, y)
   return reshape(Xv,p1,p2,p)
end
# function _psyl2spkron(A::AbstractArray{T,3}, B::AbstractArray{T,3}; isgn = 1) where {T}
#     K = size(A,3)
#     p1 = LinearAlgebra.checksquare(A[:,:,1])
#     p2 = LinearAlgebra.checksquare(B[:,:,1])
#     eye1 = isgn*I(p1)
#     eye2 = I(p2)
#     Zd = [kron(B[:,:,K]', -eye1)]
#     Zl = [kron(eye2, A[:,:,1])]
#     for k in 1:(K - 1)
#         push!(Zd, kron(B[:,:,k]', -eye1))
#         push!(Zl, kron(eye2, A[:,:,k + 1]))
#     end
#     return Zd, Zl
# end

