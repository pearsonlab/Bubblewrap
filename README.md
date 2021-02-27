# NeuralFlow

(Private) repository for project to learn and iteratively refine a mesh points that approximates probability flow on low-dimensional neural manifolds.


## Current Tasks
- [ ] Add noise to data generation
- [ ] Compute bounding points in N-D
- [ ] Efficient neighboring points computation


## Conventions
* The mesh has a set of points, and each point has a fixed set of neighboring points.
* Observed flow is the vector difference between sequentially observed points.
* Observations of new data are mapped to use the midpoint between sequentially observed points; ie, an 'observation' that includes a flow vector has coordinates between the current and last observed points.
* Mesh points surrounding an observation are the bounding points.
