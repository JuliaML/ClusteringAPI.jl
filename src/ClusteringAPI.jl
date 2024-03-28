module ClusteringAPI

# Use the README as the module docs
@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end ClusteringAPI

export ClusteringAlgorithm, ClusteringResults
export cluster, cluster_number, cluster_labels

abstract type ClusteringAlgorithm end
abstract type ClusteringResults end

"""
    cluster(ca::ClusteringAlgortihm, data) → cr::ClusteringResults

Cluster input `data` according to the algorithm specified by `ca`.
All options related to the algorithm are given as keyword arguments when
constructing `ca`. The input data can be specified two ways:

- as a (d, m) matrix, with d the dimension of the data points and m the amount of
  data points (i.e., each column is a data point).
- as a length-m vector of length-d vectors (i.e., each inner vector is a data point).

The cluster labels are always the
positive integers `1:n` with `n::Int` the number of created clusters.

The output is always a subtype of `ClusteringResults`,
which always extends the following two methods:

- `cluster_number(cr)` returns `n`.
- `cluster_labels(cr)` returns `labels::Vector{Int}` a length-m vector of labels
  mapping each data point to each cluster (`1:n`).

and always includes `ca` in the field `algorithm`.

Other algorithm-related output can be obtained as a field of the result type,
or other specific functions of the result type.
This is described in the individual algorithm implementations.
"""
function cluster(ca::ClusteringAlgorithm, data::AbstractMatrix)
    throw(ArgumentError("No implementation for `cluster` for $(typeof(ca))."))
end

"""
    cluster_number(cr::ClusteringResults) → n::Int

Return the number of created clusters in the output of [`cluster`](@ref).
"""
function cluster_number(cr::ClusteringResults)
    return length(Set(cluster_labels(cr))) # fastest way to count unique elements
end

"""
    cluster_labels(cr::ClusteringResults) → labels::Vector{Int}

Return the cluster labels of the data points used in [`cluster`](@ref).
"""
function cluster_labels(cr::ClusteringResults)
    return cr.labels # typically there
end

# two helper functions for agnostic input data type
"""
    input_data_size(data) → (d, m)

Return the data point dimension and number of data points.
"""
input_data_size(A::AbstractMatrix) = size(A)
input_data_size(A::AbstractVector{<:AbstractVector}) = (length(first(A)), length(A))

"""
    each_data_point(data)

Return an indexable iterator over each data point in `data`, that can be
indexed with indices `1:m`.
"""
each_data_point(A::AbstractMatrix) = eachcol(A)
each_data_point(A::AbstractVector{<:AbstractVector}) = A

end