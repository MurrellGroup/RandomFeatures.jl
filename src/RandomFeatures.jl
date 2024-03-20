module RandomFeatures

using Flux: Flux, unsqueeze, batched_mul
using Optimisers

# Random fourier features
# -----------------------
struct RandomFourierFeatures{T <: Real, A <: AbstractMatrix{T}}
    W::A
end

Flux.@functor RandomFourierFeatures
Optimisers.trainable(::RandomFourierFeatures) = (;)  # no trainable parameters

RandomFourierFeatures(dims::Pair{<:Integer, <:Integer}, σ::Real) =
    RandomFourierFeatures(dims, float(σ))

# d1: input dimension, d2: output dimension (d1 => d2)
function RandomFourierFeatures((d1, d2)::Pair{<:Integer, <:Integer}, σ::AbstractFloat)
    iseven(d2) || throw(ArgumentError("dimension must be even"))
    isfinite(σ) && σ > 0 || throw(ArgumentError("scale must be finite and positive"))
    return RandomFourierFeatures(randn(typeof(σ), d1, d2 ÷ 2) * σ * oftype(σ, 2π))
end

function (rff::RandomFourierFeatures{T})(X::AbstractMatrix{T}) where {T <: Real}
    WtX = rff.W'X
    return [cos.(WtX); sin.(WtX)]
end

#Handling batched inputs
function (rff::RandomFourierFeatures{T})(X::AbstractArray{T}) where {T <: Real}
    WtX = batched_mul(rff.W',X)
    return [cos.(WtX); sin.(WtX)]
end

export RandomFourierFeatures

#rff = RandomFourierFeatures(10=>20,0.1f0)
#x = randn(Float32,10,11,1)
#isapprox(rff(x[:,:]) , rff(x)[:,:,1])

###RANDOM ORIENTATION FEATURES###

#We want to build a map that takes a pair of residue locations+orientations (each residue is a vector and a rot matrix)
#and projects this to M features. These will be the pairwise distances between points.

###SHARED
struct RandomOrientationFeatures{A}
    FA::A
    FB::A
end

Flux.@functor RandomOrientationFeatures
Optimisers.trainable(::RandomOrientationFeatures) = (;)  # no trainable parameters


# d1: input dimension, d2: output dimension (d1 => d2)
function RandomOrientationFeatures(dim::Integer, σ::AbstractFloat)
    isfinite(σ) && σ > 0 || throw(ArgumentError("scale must be finite and positive"))
    return RandomOrientationFeatures(randn(typeof(σ), 3, dim, 1) .* σ, randn(typeof(σ), 3, dim, 1) .* σ)
end

###NON-GRAPH


#For non-graph version, with batch dim
function T_R3(mat::AbstractArray, rot::AbstractArray, trans::AbstractArray)
    rotc = reshape(rot, 3,3,:)  
    trans = reshape(trans, 3,1,:)
    matc = reshape(mat,3,size(mat,2),:) 
    rotated_mat = batched_mul(rotc,matc) .+ trans
    return rotated_mat
end 

function (rof::RandomOrientationFeatures)(Ti::Tuple{AbstractArray, AbstractArray})
    dim = size(rof.FA, 2)
    Nr,batch = size(Ti[1])[3:4]

    p1 = reshape(T_R3(rof.FA, Ti[1], Ti[2]),3,dim,Nr,batch)
    p2 = reshape(T_R3(rof.FB, Ti[1], Ti[2]),3,dim,Nr,batch)
    return sqrt.(sum(abs2.(unsqueeze(p1,dims=4) .- unsqueeze(p2, dims = 3)),dims=1))[1,:,:,:,:]
end




###GRAPH - will be enabled when InvariantPointAttention is registered

#=
#Note: This might be bad at sensing position in frame A for a frame B that is far from A. Unclear.
####Random Orientation Features####
function rot_features(T1,T2,FA,FB)
    l = size(T1.translations, 2)
    pointsA = transform(T1, FA)
    pointsB = transform(T2, FB)
    return reshape(sqrt.(sum(abs2.(pointsA .- pointsB), dims = 1)), (:, l))
end

subt(xi, xj, e) = xj .- xi
function graph_rot_features(g, gT, FA, FB)
    K = size(FA, 2)
    pointsA = transform(gT, FA)
    pointsB = transform(gT, FB)
    diffs = apply_edges(subt, g, xi=pointsB, xj=pointsA)
    return reshape(sqrt.(sum(abs2.(diffs), dims = 1)), (K, :))
end


function (rof::RandomOrientationFeatures)(g::GNNGraph, gT::RigidTransformation{T}) where {T <: Real}
    return graph_rot_features(g, gT,rof.FA,rof.FB)
end

function (rof::RandomOrientationFeatures)(T1::RigidTransformation{T}, T2::RigidTransformation{T}) where {T <: Real}
    return rot_features(T1,T2,rof.FA,rof.FB)
end

=#

export RandomOrientationFeatures



end
