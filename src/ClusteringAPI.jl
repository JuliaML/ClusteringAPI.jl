module ClusteringAPI

export ClusteringAlgorithm, ClusteringResults
export cluster, cluster_number, cluster_labels, cluster_probs

abstract type ClusteringAlgorithm end
abstract type ClusteringResults end


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