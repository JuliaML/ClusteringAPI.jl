# ClusteringAPI

A common interface for clustering data.
The interface is composed of the abstract types:

- `ClusteringAlgorithm`
- `ClusteringResult`

which interplay with the functions:

- `cluster`
- `cluster_number`
- `cluster_labels`

To create new clustering algorithms simply create a new
subtype of `ClusteringAlgorithm` that extends `cluster`
so that it returns a new subtype of `ClusteringResult`
which itself extends `cluster_labels`.

For more, see the docstring of `cluster`.