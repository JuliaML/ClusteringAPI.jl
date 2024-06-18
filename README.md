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
This result must extend `cluster_number, cluster_labels`
and optionally `cluster_probs`.

For developers: see two helper functions `each_data_point, input_data_size`
so that you can support matrix input while abiding the declared api
of iterable of vectors as input.

For more, see the docstring of `cluster`.