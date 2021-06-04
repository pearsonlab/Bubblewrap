## datagen.py
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


## ZP2016.ipynb
This Jupyter notebook runs the ZP2016 model using your desired dataset. 
Before running this, make a dataset either by the simulation using `dataset.py` or make a reduced dataset using `ssSVD` first. 

* Run section #1-2, then run only one of the 6 cells in section #3 depending on the dataset you’d like to run with.
* Section #4-8 are for training the model and computing the prediction log probability per time step. 
* Section #9 and 12 are for plotting and saving the MSEs to generate the plots in `Figure S4`.
* Section #10 and 11 are for making the log probability plots and computing the mean and std values in `Table 1`. 


## VJF.ipynb
This Jupyter notebook runs the VJF model using your desired dataset. 
Before running this, make a dataset either by the simulation using `dataset.py` or make a reduced dataset using `ssSVD` first. 

* Run section #1-2, then run only one of the 6 cells in section #3 depending on the dataset you’d like to run with.
* Section #4-6 are for training the model and computing the prediction log probability per time step. 
* Section #7 and 8 are for making the log probability plots and computing the mean and std values in `Table 1`. 


