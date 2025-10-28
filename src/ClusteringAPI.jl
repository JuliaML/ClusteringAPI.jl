module ClusteringAPI

export ClusteringAlgorithm, ClusteringResults
export cluster, cluster_number, cluster_labels, cluster_probs

abstract type ClusteringAlgorithm end
abstract type ClusteringResults end

"""
```julia
cluster(ca::ClusteringAlgortihm, data) → cr::ClusteringResults
```

Cluster input `data` according to the algorithm specified by `ca`.
All options related to the algorithm are given as _keyword_ arguments when
constructing `ca`.

The input `data` is a length-`m` iterable of "data points".
A data point is something generic: any objects that
a distance can be defined on them so that they can be clustered.
In the majority of cases data points are just vectors of real numbers.
If you have a matrix with each row a data point, simply pass in `eachrow(matrix)` as `data`.

The output is always a subtype of `ClusteringResults` that can be further queried.
The cluster labels are always the
positive integers `1:n` with `n::Int` the number of created clusters,
Data points that couldn't get clustered (e.g., outliers or noise)
get assigned negative integers, typically just `-1`.

`ClusteringResults` subtypes always implement the following functions:

- `cluster_labels(cr)` returns a length-`m` vector `labels::Vector{Int}` containing
  the clustering labels , so that `data[i]` has label `labels[i]`.
- `cluster_probs(cr)` returns a length-`m` vector `probs`, whose elements are length-`n` vectors
  containing the "probabilities" or "scores" for each point belonging to one of
  the created clusters (useful for fuzzy clustering algorithms).
- `cluster_number(cr)` returns `n`.

Other algorithm-related output can be obtained as a field of the result type,
or by using other specific functions of the result type.
This is described in the individual algorithm implementations docstrings.
"""
function cluster(ca::ClusteringAlgorithm, data)
    throw(ArgumentError("No implementation for `cluster` for $(typeof(ca))."))
end

@doc let # make README the `cluster` function docstring.
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end cluster

"""
    cluster_number(cr::ClusteringResults) → n::Int

Return the number of created clusters in the output of [`cluster`](@ref).
"""
function cluster_number(cr::ClusteringResults)
    return count(>(0), Set(cluster_labels(cr))) # fastest way to count positive labels
end

"""
    cluster_labels(cr::ClusteringResults) → labels::Vector{Int}

Return the cluster labels of the data points used in [`cluster`](@ref).
"""
function cluster_labels(cr::ClusteringResults)
    return cr.labels # typically there
end

"""
    cluster_probs(cr::ClusteringResults) → probs::Vector{Vector{Real}}

Return the cluster probabilities of the data points used in [`cluster`](@ref).
They are length-`n` vectors containing the "probabilities" or "score" of each point
belonging to one of the created clusters (used with fuzzy clustering algorithms).
"""
function cluster_probs(cr::ClusteringResults)
    labels = cluster_labels(cr)
    n = cluster_number(cr)
    probs = [zeros(Real, n) for _ in 1:length(labels)]
    for (i, label) in enumerate(labels)
        probs[i][label] = 1
    end
    return probs
end

end