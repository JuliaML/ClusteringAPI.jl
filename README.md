# ClusteringAPI

A common interface for clustering data.
The interface is composed of the abstract types:

- `ClusteringAlgorithm`
- `ClusteringResult`

which interplay with the functions:

- `cluster`
- `cluster_number`
- `cluster_labels`
- `cluster_probs`

## `cluster` documentation

```julia
cluster(ca::ClusteringAlgortihm, data) → cr::ClusteringResults
```

Cluster input `data` according to the algorithm specified by `ca`.
All options related to the algorithm are given as keyword arguments when
constructing `ca`.

The input `data` is a length-m iterable of "vectors" (data points).
"Vector" here is considered in the generalized sense, i.e., any objects that
a distance can be defined on them so that they can be clustered.
In the majority of cases these are vectors of real numbers.
If you have a matrix with each row a data point, simply pass in `eachrow(matrix)`.

The output is always a subtype of `ClusteringResults` that can be further queried.
The cluster labels are always the
positive integers `1:n` with `n::Int` the number of created clusters,
Data points that couldn't get clustered (e.g., outliers or noise)
get assigned negative integers, typically just `-1`.

`ClusteringResults` subtypes always implement the following functions:

- `cluster_labels(cr)` returns a length-m vector `labels::Vector{Int}` containing
  the clustering labels , so that `data[i]` has label `labels[i]`.
- `cluster_probs(cr)` returns `probs` a length-m vector of length-`n` vectors
  containing the "probabilities" or "score" of each point belonging to one of
  the created clusters (useful for fuzzy clustering algorithms).
- `cluster_number(cr)` returns `n`.

Other algorithm-related output can be obtained as a field of the result type,
or by using other specific functions of the result type.
This is described in the individual algorithm implementations docstrings.

## For developers

To create new clustering algorithms simply create a new
subtype of `ClusteringAlgorithm` that extends `cluster`
so that it returns a new subtype of `ClusteringResult`.
This result must extend `cluster_number, cluster_labels`
and optionally `cluster_probs`.

See also the two helper functions `each_data_point, input_data_size`
which help you can support matrix input while abiding the declared api
of iterable of vectors as input.

For more, see the docstring of `cluster`.