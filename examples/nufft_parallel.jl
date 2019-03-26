using Distributed
addprocs(1);
using Revise
using LinearAlgebra
@everywhere  using Pkg; Pkg.activate(".")
@everywhere  push!(LOAD_PATH, "/Users/mtg79/Documents/FastTransforms.jl/src/")
@everywhere  using Revise, FastTransforms, Distributed, DistributedArrays

##
x = rand(10^2)
c = complex(x)

## serial
@time A = plan_nufft1(x)
@time b = A*c

## parallel
@time plan =parallel_plan_nufft1(x)
@time f = mulpar3( plan, c)
