using Test
using ClusteringAPI

struct TestClustering <: ClusteringAlgorithm
end
struct TestResults <: ClusteringResults
    labels::Vector{Int}
    n::Int
end

function ClusteringAPI.cluster(::TestClustering, data)
    return TestResults(fill(1, length(data)), 2)
end

cr = cluster(TestClustering(), randn(100))
@test cluster_number(cr) == 1
@test cluster_labels(cr) == fill(1, 100)
@test cluster_probs(cr) == fill([1.0], 100)
