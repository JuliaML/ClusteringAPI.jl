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

To create new clustering algorithms simply create a new
subtype of `ClusteringAlgorithm` that extends `cluster`
so that it returns a new subtype of `ClusteringResult`.
The result must extend `cluster_number, cluster_labels`
and optionally `cluster_probs`.

Note that data input type must always be `AbstractVector` of vectors
(anything that can have distance defined).
Two helper functions `each_data_point, input_data_size` can help
making this harmonious with matrix inputs.

For more, see the docstring of `cluster`.