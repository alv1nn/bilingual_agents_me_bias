# pragmatic_agents_me_bias


This code belongs to the CogSci 2020 submission 'Reinforcement of Semantic Representations in Pragmatic Agents Leads to the
Emergence of a Mutual Exclusivity Bias.


## Notation

In the code 'labeling' usually refers to the single agent setting of the Lewis game and 'communication' refers to the two agent setting of the Lewis game (see paper).


## Prerequisites

The project was implemented with Python 3.7.3. We only list packages that are not part of the Python standard library. 

We used the following packages:
* for running the simulations: tensorflow 2.0 and numpy 1.16.4
* for plotting the data: matplotlib 3.1.0

While the code will only work with tensorflow 2.0 it might run with other python, numpy and matplotlib version. 


## File structure 

There project contains the python file, where the agents are implemented and one folder for each experiment in the 

### RSA_communication_agents.py

This file contains the agents. 

### ME_bias

This folder contains all code belonging specifically to experiment 1 that investigates the emergence of a mutual exclusivity bias in literal and pragmatic agents.

* ME_bias_labeling.ipynb: runs the simulations and saves the data for the single agent setting
* ME_bias_communication.ipynb: runs the simulations and saves the data for the two agents setting
* plot_ME_bias_labeling.ipynb: visualizes the data from the single agent setting
* plot_ME_bias_communication.ipynb: visualizes the data from the two agent setting
* ME_data_analysis_and_visualization_functions.py: the functions for plotting the data and caluclating the ME indices are outsourced to this file. 

The plots are apart from minor changes the figures used for the paper. 

### convergence_time_scales 

This folder contains all code belonging specifically to experiment 2 that investigates the convergence time scales of literal and pragmatic agents for different types of input distributions. 

* convergence_time_labeling.ipynb: runs the simulations and saves the data for the single agent setting
* convergence_time_communication.ipynb: runs the simulations and saves the data for the two agent setting
* plot_learning_time_scales.ipynb: plots the data from both settings 

Also here, the plots are apart from minor changes the figures used for the paper. 

## Practical information  

It might be that you encounter numerical instabilities when applying the model to larger systems or using larger alpha-values for the pragmatic agents. You can amend this by small changes to the *RSA_communication_agents.py* file by simply adding small epsilon values to avoid division by zero or taking the logarithm of zero. Just make sure that epsilon is small enough so it does not interfere with the calculation (*small enough* is use case dependent). As a side note on that: if you add epsilon to avoid taking the logarithm of zero in the sampling function, do so already on the return value of the call function to allow for correct computation of the gradient. 

We did not implement such a fix as it is not required for the simulations done for the paper. 