Repository for code and figures associated with the paper 'Bubblewrap: Online tiling and real-time flow prediction on neural manifolds'.

## Requirements
Our algorithm is implemented in python with some extra packages including: numpy, jax (for GPU), and matplotlib (for plotting). 

The streaming dimension-reduction method is hosted in our repository here: [proSVD](https://github.com/pearsonlab/proSVD). Instructions for cloning and installing are included in its Readme. proSVD is needed to run the streaming dimension reduction example. 

We used python version 3.9 and conda to install the libraries listed in the environment file. 
We provide an environment file for use with conda to create a new environment with these requirements, though we note that jax requires additional setup for GPU integration (https://github.com/google/jax). 

Additionally, if running comparison models, VJF can be installed from its source repository (pip install git+https://github.com/catniplab/vjf.git).

## Evaluation
We provide a number of python scripts and jupyter notebooks for:
* data generation (for simulated datasets), in `datagen.py`
* streaming dimension reduction, for example `scripts/dimension_reduction_neuropixels.py`
* running our model bubblewrap, for example `scripts/run_bubblewrap.py`
* running comparison models (VJF and ZP(2016)), for example `models/VJF.ipynb`.


### Data generation: datagen.py
This script generates the simulated Van der pol and Lorenz attractor datasets.

* usage: `python datagen.py (vdp | lorenz)`
* output: 4 npz files for each run. 
* output file names: `(vdp | lorenz)_(1 | 100)trajectories_(num_dim)_500to2500_noise(0.05 | 0.2).npz`
* num_dim = 2 for vdp and 3 for lorenz
###
    In each output files: 
        output[‘x’] is the latent 
        output[‘y’] is the observations
        output[‘u’] is the controls that are all zeros


### Dimension reduction: scripts/dimension_reduction_neuropixels.py
This script loads the neuropixels data from https://doi.org/10.25378/janelia.7739750.v4 and uses Random Projections then ssSVD to produce a reduced dimension dataset, which is fed into Bubblewrap or ZP2016/VJF. The value 'X' can be replaced with any MxN matrix, where columns are neurons and rows are timepoints. 
Currently loads about 1/5th of the data, which runs in about 10 minutes. 

### Running Bubblewap: scripts/run_bubblewrap.py
This script loads pre-generated data from the 2D Van der Pol oscillator case, as generated with `python datagen.py vdp`. It creates the Bubblewrap model and runs online, with a plot of the log predictive probability generated upon completion. If desired, the relevant objects (such as the tile final locations) can also be saved for later plotting in, for example `scripts/plot_2d_3d.py`.


### Model comparison: models/ZP2016.ipynb
This Jupyter notebook runs the ZP2016 model using your desired dataset. 
Before running this, make a dataset either by the simulation using `dataset.py` or make a reduced dataset using `ssSVD` first. 

* Run section #1-2, then run only one of the 6 cells in section #3 depending on the dataset you’d like to run with.
* Section #4-8 are for training the model and computing the prediction log probability per time step. 
* Section #9 and 12 are for plotting and saving the MSEs to generate the plots in `Figure S4`.
* Section #10 and 11 are for making the log probability plots and computing the mean and std values in `Table 1`. 


### Model comparison: models/VJF.ipynb
This Jupyter notebook runs the VJF model using your desired dataset. 
Before running this, make a dataset either by the simulation using `dataset.py` or make a reduced dataset using `ssSVD` first. 

* Run section #1-2, then run only one of the 6 cells in section #3 depending on the dataset you’d like to run with.
* Section #4-6 are for training the model and computing the prediction log probability per time step. 
* Section #7 and 8 are for making the log probability plots and computing the mean and std values in `Table 1`. 


