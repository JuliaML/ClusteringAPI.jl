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

@test ClusteringAPI.input_data_size([rand(3) for _ in 1:30]) == (3, 30)
@test ClusteringAPI.input_data_size(rand(3,30)) == (3, 30)

v = [ones(3) for _ in 1:30]
@test ClusteringAPI.each_data_point(v) == v
@test ClusteringAPI.each_data_point(ones(3,30)) == v