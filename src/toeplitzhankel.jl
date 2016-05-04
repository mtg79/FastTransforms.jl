function partialchol(H::Hankel)
    # Assumes positive definite
    σ=eltype(H)[]
    n=size(H,1)
    C=Vector{eltype(H)}[]
    v=[H[:,1];vec(H[end,2:end])]
    d=diag(H)
    @assert length(v) ≥ 2n-1
    tol=eps(eltype(H))*log(n)
    for k=1:n
        mx,idx=findmax(d)
        if mx ≤ tol break end
        push!(σ,inv(mx))
        push!(C,v[idx:n+idx-1])
        for j=1:k-1
            nCjidxσj = -C[j][idx]*σ[j]
            Base.axpy!(nCjidxσj, C[j], C[k])
        end
        @simd for p=1:n
            @inbounds d[p]-=C[k][p]^2/mx
        end
    end
    for k=1:length(σ) scale!(C[k],sqrt(σ[k])) end
    C
end

function toeplitzcholmult(T,C,v)
    n,K = length(v),length(C)
    ret,temp1,temp2 = zero(v),zero(v),zero(v)
    un,ze = one(eltype(v)),zero(eltype(v))
    broadcast!(*, temp1, C[K], v)
    A_mul_B!(un, T, temp1, ze, temp2)
    broadcast!(*, ret, C[K], temp2)
    for k=K-1:-1:1
        broadcast!(*, temp1, C[k], v)
        A_mul_B!(un, T, temp1, ze, temp2)
        broadcast!(*, temp1, C[k], temp2)
        broadcast!(+, ret, ret, temp1)
    end
    ret
end


# Plan a multiply by DL*(T.*H)*DR
immutable ToeplitzHankelPlan{TT}
    T::TriangularToeplitz{TT}
    C::Vector{Vector{TT}}   # A cholesky factorization of H: H=CC'
    DL::Vector{TT}
    DR::Vector{TT}

    ToeplitzHankelPlan(T,C,DL,DR)=new(T,C,DL,DR)
end


function ToeplitzHankelPlan(T::TriangularToeplitz,C::Vector,DL::AbstractVector,DR::AbstractVector)
    TT=promote_type(eltype(T),eltype(C[1]),eltype(DL),eltype(DR))
    ToeplitzHankelPlan{TT}(T,C,collect(TT,DL),collect(TT,DR))
end
ToeplitzHankelPlan(T::TriangularToeplitz,C::Matrix) =
    ToeplitzHankelPlan(T,C,ones(size(T,1)),ones(size(T,2)))
ToeplitzHankelPlan(T::TriangularToeplitz,H::Hankel,D...) =
    ToeplitzHankelPlan(T,partialchol(H),D...)

*(P::ToeplitzHankelPlan,v::Vector)=P.DL.*toeplitzcholmult(P.T,P.C,P.DR.*v)



# Legendre transforms

#=
Λ{T}(::Type{T},z)=z<5?gamma(z+one(T)/2)/gamma(z+one(T)):exp(lgamma(z+one(T)/2)-lgamma(z+one(T)))

# use recurrence to construct Λ fast on a range of values
function Λ{T}(::Type{T},r::UnitRange)
    n=length(r)
    ret=Array(T,n)
    ret[1]=Λ(T,r[1])
    for k=2:n
        ret[k]=ret[k-1]*(r[k-1]+one(T)/2)/r[k]
    end
    ret
end

function Λ{T}(::Type{T},r::FloatRange)
    @assert step(r)==0.5
    n=length(r)
    ret=Array(T,n)
    ret[1]=Λ(T,r[1])
    ret[2]=Λ(T,r[2])
    for k=3:n
        ret[k]=ret[k-2]*(r[k-2]+one(T)/2)/r[k]
    end
    ret
end

Λ(z::Number)=Λ(typeof(z),z)
Λ(z::AbstractArray)=Λ(eltype(z),z)
=#

function leg2chebuTH{TT}(::Type{TT},n)
    λ=Λ(0:half(TT):n-1)
    t=zeros(TT,n)
    t[1:2:end]=λ[1:2:n]./(((1:2:n)-2))
    T=TriangularToeplitz(-2/π*t,:U)
    H=Hankel(λ[1:n]./((1:n)+1),λ[n:end]./((n:2n-1)+1))
    T,H
end

th_leg2chebuplan{TT}(::Type{TT},n)=ToeplitzHankelPlan(leg2chebuTH(TT,n)...,1:n,ones(TT,n))

function leg2chebTH{TT}(::Type{TT},n)
    λ=Λ(0:half(TT):n-1)
    t=zeros(TT,n)
    t[1:2:end]=λ[1:2:n]
    T=TriangularToeplitz(2/π*t,:U)
    H=Hankel(λ[1:n],λ[n:end])
    DL=ones(TT,n)
    DL[1]/=2
    T,H,DL
end

th_leg2chebplan{TT}(::Type{TT},n)=ToeplitzHankelPlan(leg2chebTH(TT,n)...,ones(TT,n))

# function leg2chebTHslow(n)
#     λ=map(Λ,0:0.5:n-1)
#     t=zeros(n)
#     t[1:2:end]=λ[1:2:n]
#     T=TriangularToeplitz(2/π*t,:U)
#     H=Hankel(λ[1:n],λ[n:end])
#     DL=ones(n)
#     DL[1]*=0.5
#     T,H,DL
# end


th_leg2cheb(v)=th_leg2chebplan(eltype(v),length(v))*v
th_leg2chebu(v)=th_leg2chebuplan(eltype(v),length(v))*v
