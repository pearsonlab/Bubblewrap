# NeuralFlow

(Private) repository for project to learn and iteratively refine a mesh points that approximates probability flow on low-dimensional neural manifolds.

## Structure
* datagen: Classes for generating observation data points; mostly from deterministic systems atm.
* models: Our implementations of others' methods for comparative purposes.
* field: root folder for our formulation of some kind of mesh/graph/field for modeling low-D trajectories or probabilistic transitions.
* scripts: Code to put all the pieces together and run a simulation.


## Current Tasks
- [ ] 
- [ ] 
- [ ] 


## Conventions
* The mesh has a set of points, and each point has a fixed set of neighboring points.
* Observed flow is the vector difference between sequentially observed points.
* Observations of new data are mapped to use the midpoint between sequentially observed points; ie, an 'observation' that includes a flow vector has coordinates between the current and last observed points.
* Mesh points surrounding an observation are the bounding points.
