import ClusteringAPI
import Clustering


# The automatic H-clustering which decides the number of clusters
# by splitting at the largest gap in the dendrogram
struct HClustAuto <: ClusteringAlgorithm end

struct HClustAutoResults
    labels::Vector{Int}
    hclust
    splitidx
end

function ClusteringAPI.cluster(::HClustAuto, X)
    input_data_size(X)[2] < 3000 || throw(ArgumentError("size more than 3000"))
    hc = Clustering.hclust(pairwise(Euclidean(), X))
    # Split at the largest gap in the dendrogram
    idx = argmax(diff(hc.heights))
    labels = Clustering.cutree(hc; h=mean(hc.heights[idx:idx+1]))
    return HClustAutoResults(labels, hc, idx)
end