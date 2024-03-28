import ClusteringAPI
import Clustering
import Optim
using Distances

#####################################################################################
# Clustering API
#####################################################################################
# improved dbscan algorithm from Attractors.jl
@kwdef struct ADBSCAN{R<:Union{Real, String}, M, F<:Function} <: ClusteringAlgorithm
    clust_distance_metric::M = Euclidean()
    min_neighbors::Int = 10
    rescale_features::Bool = true
    optimal_radius_method::R = "silhouettes_optim"
    num_attempts_radius::Int = 100
    silhouette_statistic::F = mean
    max_used_features::Int = 0
    use_mmap::Bool = false
end

function ClusteringAPI.cluster(config::ADBSCAN, data)
    d, m = input_data_size(data) # dimension = rows, amount = columns
    if d ≥ m
        throw(ArgumentError("""
        Not enough data points. The algorithm needs more data points than data dimensions.
        """))
    end
    if config.rescale_features
        data = _rescale_to_01(data)
    end
    ϵ_optimal, v_optimal = _extract_ϵ_optimal(data, config)
    distances = _distance_matrix(data, config)
    labels = _cluster_distances_into_labels(distances, ϵ_optimal, config.min_neighbors)
    return ADBSCANResult(ca, labels, ϵ_optimal, v_optimal, data)
end

struct ADBSCANResult
    algorithm::ADBSCAN
    labels::Vector{Int}
    ε_optimal
    v_optimal
    data
end

ClusteringAPI.cluster_labels(r::ADBSCANResult) = r.labels

#####################################################################################
# Source code
#####################################################################################
function _rescale_to_01(features)
    mini, maxi = _minmaxima(features)
    rescaled = map(f -> (f .- mini) ./ (maxi .- mini), each_data_point(features))
    # TODO: This doesn't respect matrix input
    return rescaled # ensure it stays the same type
end

function _minmaxima(features)
    points = each_data_point(features)
    D = length(first(points))
    mi = fill(Inf, D)
    ma = fill(-Inf, D)
    for point in points
        for i in 1:D
            if point[i] > ma[i]
                ma[i] = point[i]
            elseif point[i] < mi[i]
                mi[i] = point[i]
            end
        end
    end
    return mi, ma
end

function _distance_matrix(data, config::ADBSCAN)
    metric = config.clust_distance_metric
    L = input_data_size(data)[2]
    if config.use_mmap
        pth, s = mktemp()
        dists = Mmap.mmap(s, Matrix{Float32}, (L, L))
    else
        dists = zeros(L, L)
    end
    if metric isa Metric # then the `pairwise` function is valid
        # ensure that we give the vector of static vectors to pairwise!
        pairwise!(metric, dists, data; symmetric = true)
    else # it is any arbitrary distance function, e.g., used in aggregating attractors
        points = each_data_point(data)
        @inbounds for i in eachindex(points)
            Threads.@threads for j in i:length(points)
                v = metric(points[i], points[j])
                dists[i, j] = dists[j, i] = v # utilize symmetry
            end
        end
    end
    return dists
end

function _extract_ϵ_optimal(features, config::ADBSCAN)
    (; min_neighbors, clust_distance_metric, optimal_radius_method,
    num_attempts_radius, silhouette_statistic, max_used_features) = config
    m = input_data_size(features)[2]

    if optimal_radius_method isa String
        if max_used_features == 0 || max_used_features > m
            features_for_optimal = each_data_point(features)
        else
            # subsample features to accelerate optimal radius search
            features_for_optimal = sample(each_data_point(features), max_used_features; replace = false)
        end
        # get optimal radius (function dispatches on the radius method)
        ϵ_optimal, v_optimal = optimal_radius_dbscan(
            features_for_optimal, min_neighbors, clust_distance_metric,
            optimal_radius_method, num_attempts_radius, silhouette_statistic
        )
    elseif optimal_radius_method isa Real
        ϵ_optimal = optimal_radius_method
        v_optimal = missing
    else
        error("Specified `optimal_radius_method` is incorrect. Please specify the radius
        directly as a `Real` number or the method to compute it as a `String`")
    end
    return ϵ_optimal, v_optimal
end

# Already expecting the distance matrix, the output of `pairwise`
function _cluster_distances_into_labels(distances, ϵ_optimal, min_neighbors)
    dbscanresult = Clustering.dbscan(distances, ϵ_optimal; min_neighbors, metric=nothing)
    cluster_labels = cluster_assignment(dbscanresult)
    return cluster_labels
end

"""
Util function for `classify_features`. Returns the assignment vector, in which the i-th
component is the cluster index of the i-th feature.
"""
function cluster_assignment(clusters, data; include_boundary=true)
    assign = zeros(Int, size(data)[2])
    for (idx, cluster) in enumerate(clusters)
        assign[cluster.core_indices] .= idx
        if cluster.boundary_indices != []
            if include_boundary
                assign[cluster.boundary_indices] .= idx
            else
                assign[cluster.boundary_indices] .= -1
            end
        end
    end
    return assign
end
function cluster_assignment(dbscanresult::Clustering.DbscanResult)
    labels = dbscanresult.assignments
    # Attractors.jl follows the convention of `-1` to unclustered points.
    return replace!(labels, 0=>-1)
end

#####################################################################################
# Source code - optimal radius
#####################################################################################
function optimal_radius_dbscan(features, min_neighbors, metric, optimal_radius_method,
    num_attempts_radius, silhouette_statistic)
    if optimal_radius_method == "silhouettes"
        ϵ_optimal, v_optimal = optimal_radius_dbscan_silhouette(
            features, min_neighbors, metric, num_attempts_radius, silhouette_statistic
        )
    elseif optimal_radius_method == "silhouettes_optim"
        ϵ_optimal, v_optimal = optimal_radius_dbscan_silhouette_optim(
            features, min_neighbors, metric, num_attempts_radius, silhouette_statistic
        )
    elseif optimal_radius_method == "knee"
        ϵ_optimal, v_optimal = optimal_radius_dbscan_knee(features, min_neighbors, metric)
    elseif optimal_radius_method isa Real
      ϵ_optimal = optimal_radius_method
      v_optimal = NaN
    else
        error("Unknown `optimal_radius_method`.")
    end
    return ϵ_optimal, v_optimal
end

"""
Find the optimal radius ε of a point neighborhood to use in DBSCAN, the unsupervised
clustering method for `AttractorsViaFeaturizing`. Iteratively search
for the radius that leads to the best clustering, as characterized by quantifiers known as
silhouettes. Does a linear (sequential) search.
"""
function optimal_radius_dbscan_silhouette(features, min_neighbors, metric,
       num_attempts_radius, silhouette_statistic
    )
    feat_ranges = features_ranges(features)
    ϵ_grid = range(
        minimum(feat_ranges)/num_attempts_radius, minimum(feat_ranges);
        length=num_attempts_radius
    )
    s_grid = zeros(size(ϵ_grid)) # silhouette statistic values (which we want to maximize)

    # vary ϵ to find the best one (which will maximize the mean silhouette)
    dists = pairwise(metric, features)
    for i in eachindex(ϵ_grid)
        clusters = dbscan(dists, ϵ_grid[i]; min_neighbors, metric = nothing)
        sils = silhouettes_new(clusters, dists)
        s_grid[i] = silhouette_statistic(sils)
    end

    optimal_val, idx = findmax(s_grid)
    ϵ_optimal = ϵ_grid[idx]
    return ϵ_optimal, optimal_val
end

function features_ranges(features)
    d = StateSpaceSet(features) # zero cost if `features` is a `Vector{<:SVector}`
    mini, maxi = minmaxima(d)
    return maxi .- mini
end

"""
Same as `optimal_radius_dbscan_silhouette`,
but find minimum via optimization with Optim.jl.
"""
function optimal_radius_dbscan_silhouette_optim(
        features, min_neighbors, metric, num_attempts_radius, silhouette_statistic
    )
    feat_ranges = features_ranges(features)
    # vary ϵ to find the best radius (which will maximize the mean silhouette)
    dists = pairwise(metric, features)
    f = (ϵ) -> silhouettes_from_distances(
        ϵ, dists, min_neighbors, silhouette_statistic
    )
    opt = Optim.optimize(
        f, minimum(feat_ranges)/100, minimum(feat_ranges); iterations=num_attempts_radius
    )
    ϵ_optimal = Optim.minimizer(opt)
    optimal_val = -Optim.minimum(opt) # we minimize using `-`
    return ϵ_optimal, optimal_val
end

function silhouettes_from_distances(ϵ, dists, min_neighbors, silhouette_statistic=mean)
    clusters = Clustering.dbscan(dists, ϵ; min_neighbors, metric = nothing)
    sils = silhouettes_new(clusters, dists)
    # We return minus here because Optim finds minimum; we want maximum
    return -silhouette_statistic(sils)
end

"""
Find the optimal radius ϵ of a point neighborhood for use in DBSCAN through the elbow method
(knee method, highest derivative method).
"""
function optimal_radius_dbscan_knee(_features::Vector, min_neighbors, metric)
    features = StateSpaceSet(_features)
    tree = searchstructure(KDTree, features, metric)
    # Get distances, excluding distance to self (hence the Theiler window)
    d, n = size(features)
    features_vec = [features[:,j] for j=1:n]
    _, distances = bulksearch(tree, features_vec, NeighborNumber(min_neighbors), Theiler(0))
    meandistances = map(mean, distances)
    sort!(meandistances)
    maxdiff, idx = findmax(diff(meandistances))
    ϵ_optimal = meandistances[idx]
    return ϵ_optimal, maxdiff
end


# The following function is left here for reference. It is not used anywhere in the code.
# It is the original implementation we have written based on the bSTAB paper.
"""
Find the optimal radius ε of a point neighborhood to use in DBSCAN, the unsupervised clustering
method for `AttractorsViaFeaturizing`. The basic idea is to iteratively search for the radius that
leads to the best clustering, as characterized by quantifiers known as silhouettes.
"""
function optimal_radius_dbscan_silhouette_original(features, min_neighbors, metric; num_attempts_radius=200)
    d,n = size(features)
    feat_ranges = maximum(features, dims = d)[:,1] .- minimum(features, dims = d)[:,1];
    ϵ_grid = range(minimum(feat_ranges)/num_attempts_radius, minimum(feat_ranges), length=num_attempts_radius)
    s_grid = zeros(size(ϵ_grid)) # average silhouette values (which we want to maximize)

    # vary ϵ to find the best one (which will maximize the minimum silhouette)
    for i in eachindex(ϵ_grid)
        clusters = Clustering.dbscan(features, ϵ_grid[i]; min_neighbors)
        dists = pairwise(metric, features)
        class_labels = cluster_assignment(clusters, features)
        if length(clusters) ≠ 1 # silhouette undefined if only one cluster
            sils = Clustering.silhouettes(class_labels, dists)
            s_grid[i] = minimum(sils)
        else
            s_grid[i] = 0; # considers single-cluster solution on the midpoint (following Wikipedia)
        end
    end

    max, idx = findmax(s_grid)
    ϵ_optimal = ϵ_grid[idx]
end


#####################################################################################
# Calculate Silhouettes
#####################################################################################
"""
Calculate silhouettes. A bit slower than the implementation in `Clustering.jl` but seems
to be more robust. The latter seems to be incorrect in some cases.
"""
function silhouettes_new(dbscanresult::Clustering.DbscanResult, dists::AbstractMatrix)
    labels = dbscanresult.assignments
    clusters = [findall(x->x==i, labels) for i=1:maximum(labels)] #all clusters
    if length(clusters) == 1 return zeros(length(clusters[1])) end #all points in the same cluster -> sil = 0
    sils = zeros(length(labels))
    outsideclusters = findall(x->x==0, labels)
    for (idx_c, cluster) in enumerate(clusters)
        @inbounds for i in cluster
            a = sum(@view dists[i, cluster])/(length(cluster)-1) #dists should be organized s.t. dist[i, cluster] i= dist from i to idxs in cluster
            b = _calcb!(i, idx_c, dists, clusters, outsideclusters)
            sils[i] = (b-a)/(max(a,b))
        end
    end
    return sils
end

function _calcb!(i, idx_c_i, dists, clusters, outsideclusters)
    min_dist_to_clstr = typemax(eltype(dists))
    for (idx_c, cluster) in enumerate(clusters)
        idx_c == idx_c_i && continue
        dist_to_clstr = mean(@view dists[cluster,i]) #mean distance to other clusters
        if dist_to_clstr < min_dist_to_clstr min_dist_to_clstr = dist_to_clstr end
    end
    min_dist_to_pts = typemax(eltype(dists))
    for point in outsideclusters
        dist_to_pts = dists[point, i] # distance to points outside clusters
        if dist_to_pts < min_dist_to_pts
            min_dist_to_pts = dist_to_pts
        end
    end
    return min(min_dist_to_clstr, min_dist_to_pts)
end
