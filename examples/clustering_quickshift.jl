import ClusteringAPI
using NearestNeighbors, ProgressMeter, Distances
using StatsBase: sample

#####################################################################################
# Clustering API
#####################################################################################
# TODO: make it work with non-matrix, as it uses KDTrees anyways
@kwdef struct QuickShift <: ClusteringAlgorithm
    sigma = nothing
    maxlength = nothing
    maxknn = 100
end

function ClusteringAPI.cluster(config::QuickShift, data::Union{AbstractVector, AbstractMatrix})
    data = to_matrix_32(data)
    return cluster(config, data)
end

to_matrix_32(data::AbstractMatrix) = convert(Matrix{Float32}, data)
to_matrix_32(data) = Float32.(reduce(hcat, vec(data)))

function ClusteringAPI.cluster(config::QuickShift, data::Matrix{Float32})
    sigma, maxlength = _quickshift_params(config, data)
    config = QuickShift(sigma, maxlength, config.maxknn)
    cr = _quickshift(Float32.(data), config)
    return cr
end

using Statistics: median
function _quickshift_params(config::QuickShift, data)
    if isnothing(config.sigma)
        sdata = sample(data, min(1000, input_data_size(data)[2]); replace = false)
        sigma = Float32(median(pairwise(Euclidean(), sdata))/config.maxknn)
    else
        sigma = Float32(config.sigma)
    end
    maxlength = isnothing(config.maxlength) ? 10sigma : config.maxlength
    return sigma, maxlength
end

struct QuickShiftResults
    algorithm
    labels::Vector{Int}
    rootind
    links
end

#####################################################################################
# Source code
#####################################################################################
function _quickshift(data::Matrix{Float32}, config::QuickShift)
    (; sigma, maxlength, maxknn) = config
    tree = KDTree(data)
    # @show sigma
    d, N = input_data_size(data)
    factor1 = 1f0 / (2*sigma^2)
    factor2 = 1/(2*pi*sigma^2*N)
    G = zeros(Float32, 1, N)
    nninds = Vector{Vector{Int}}(undef, N)
    @showprogress "Computing kernel distances ... " for n = 1:N
        knnind = knn(tree, view(data, :, n), min(d, maxknn), true)[1]
        # what was this supposed to achieve? How can knnind ever be less than `10`???
        # ind = length(knnind) > 10 ? knnind : 1:N
        nninds[n] = knnind
        G[n] = gauss(data, n, knnind, factor1, factor2)
    end
    # println("median lenghts:",(@p map nninds length | flatten | median))
    links = [Any[] for i in 1:N]
    rootind = -1
    inflength = typemax(eltype(G))
    Nrange = 1:N
    minind = 0
    mindist = inflength
    @showprogress 1 "Linking ... " for i in 1:N
        for inds = [nninds[i]; Nrange]
            minind, mindist = link(i, inds, G, data)
            if minind != 0
                break
            end
        end
        if mindist == inflength
            if rootind < 0
                rootind = i
            else
                push!(links[rootind], (sqrt(mindist), i))
            end
        else
            push!(links[minind], (sqrt(mindist), i))
        end
    end
    labels = quickshiftlabels(links, rootind, maxlength)
    return QuickShiftResults(config, labels, rootind, links)
end

function quickshiftlabels(links, rootind, maxlength)
    labels = zeros(Int, length(links))
    cut_internal(rootind, links, labels, maxlength, 1, 2)
    labels
end

#####################################################################################
# Source code
#####################################################################################
function _dist(a::Array{Float32,2}, i::Int, b::Array{Float32,2}, j::Int)
    sum = 0f0
    for d = 1:size(a,1)
        sum += (a[d,i]-b[d,j])^2
    end
    sum
end

function link(i, inds, G, data)
    mindist = typemax(eltype(data))
    minind = 0
    for n = inds
        if G[n]>G[i]
            d = _dist(data,i,data,n)
            if d < mindist
                mindist = d
                minind = n::Int
            end
        end
    end
    minind, mindist
end

function gauss(data, n, ind, factor1, factor2)
    s = 0f0
    for m = ind
        s += exp(-_dist(data,n,data,m)*factor1)
    end
    factor2 * s
end

function cut_internal(ind, links, labels, maxlength, label, maxlabel)
    labels[ind] = label
    for x in links[ind]
        if x[1] > maxlength
            maxlabel += 1
        end
        maxlabel = max(label, cut_internal(x[2], links, labels, maxlength, x[1] > maxlength ? maxlabel : label, maxlabel))
    end
    maxlabel
end

# test:

# data = @p map unstack(1:10) (x->10*randn(2,1).+randn(2,100)) | flatten

# 10 clusters
# clusters = map(x -> 10*randn(2, 1) .+ randn(2, 100), 1:10)
# data = hcat(clusters...)
# qs = QuickShift()
# res = ClusteringAPI.cluster(qs, data)
# doesn't give 10 clusters.