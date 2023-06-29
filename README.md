# D-LiNN
## Decentralized Linear Spiking Neural Network
<br>

## Idea:
given a set of distributed layers of an LSM, a production of meaningful data representation, generalization, as well as patten generalization of through time-dependent spike-timings may produce high-dimensional data points.

Suppose you have a 1024 * 1024 * 1024 reservoir filled with detector neurons
When you supply a data cluster, be it images, binary streams, 3D objects or scatter plots, it will cause a "ripple" effect from the center of the 1024 * 1024 * 1024 reservoir
Measuring the ripples from all 6 faces of the reservoir may produce a high density 3-dimensional data.

To increase the output data's dimensionality, simply reorganize the reservoir of a ico-sphere, where the faces of each subdivision are directly linked to each dimension of a data output

k-means clustering of the detector neurons will be used for data reorganization.
"face-neurons" will be denoised through a DBSCAN algorithm.