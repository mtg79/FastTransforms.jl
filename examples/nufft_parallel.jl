using Distributed
addprocs(2);
using Revise

@everywhere  using Pkg; Pkg.activate(".")
@everywhere  push!(LOAD_PATH, "/Users/mtg79/Documents/FastTransforms.jl/src/")
@everywhere  using Revise
@everywhere  using FastTransforms
@everywhere  using Distributed, DistributedArrays

println("a")
###
x = rand(10);
xc = complex(x);

A = plan_nufft1(x);
@time b = A*xc;

#Apar = par_plan_nufft1(x);
#bpar = complex(zeros(size(b)))
#@time mulpar!(bpar,A,xc);
#t = @spawnat 2 plan_nufft1(x)*complex(xc);


Ap = parallel_plan_nufft1(x)

bpar = complex(zeros(size(x)))
f2 = mulpar2( Ap, xc);

partial_plan_nufft1(Darray[1.0], [1.0], [1.0], [1.0], 1e-12)

@everywhere using FFTW;
At = @spawnat 2 plan_nufft1(x);
Av = @spawnat 2 fetch(At)*xc;
fetch(Av)

c = xc;
sumt = zeros(eltype(c), length(c))
f2 = @spawnat 1 fetch(partial_nufft_plans[1])*c
f = fetch(f2)
sumt = sumt + f

f2 = @spawnat 2 fetch(partial_nufft_plans[2])*c
f = fetch(f2)
# sumt = sumt + f


@everywhere function mulpar2(partial_nufft_plans::Vector{Future}, c::AbstractVector{T}) where {T}

    sumt = zeros(eltype(c), length(c))
    for p in procs()
        f2 = @spawnat p fetch(partial_nufft_plans[p])*c
        f = fetch(f2)
        sumt = sumt + f
    end
    sumt
end
