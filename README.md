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

The specification of the API is based on `cluster`: given
a `ClusteringAlgorithm` and some data in the form of iterable of "vectors", `cluster` returns a `ClusteringResult`.
The result can be queried with the functions `cluster_number, cluster_labels, cluster_probs`.

## For developers

To create new clustering algorithms simply create a new
subtype of `ClusteringAlgorithm` that extends `cluster`
so that it returns a new subtype of `ClusteringResult`.
This result must extend `cluster_number, cluster_labels`
and optionally `cluster_probs`.

For more, see the docstring of `cluster`.