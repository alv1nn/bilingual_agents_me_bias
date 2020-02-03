## pragmatic_agents_me_bias
This code belongs to the CogSci 2020 submission **Reinforcement of Semantic Representations in Pragmatic Agents Leads to the
Emergence of a Mutual Exclusivity Bias**.

### 1. Notation
In any file and folder names 'labeling' usually refers to the single agent setting of the Lewis game and 'communication' refers to the two agent setting of the Lewis game (see paper).

### 2. Prerequisites
The project was implemented with Python 3.7.3. Apart from standard library packages we use: 
* tensorflow 2.0 and numpy 1.16.4 (for implementing the agents and running the simulations)
* matplotlib 3.1.0 (for plotting the data)

While tensorflow (>=)2.0 is prerequisites other verions for python and numpy may work.  

### 3. File structure 
There project contains
* file *RSA_communication_agents.py*, where the agents are implemented
* folder *ME_bias* for Experiment 1
* folder *convergence_time_scales* for Experiment 2

The two folders are described here in more detail. 

#### ME_bias
This folder contains all code belonging to Experiment 1, which investigates the emergence of a mutual exclusivity bias in literal and pragmatic agents.
* ME_bias_labeling.ipynb: runs the simulations and saves the data for the single agent setting
* ME_bias_communication.ipynb: runs the simulations and saves the data for the two agents setting
* plot_ME_bias_labeling.ipynb: visualizes the data from the single agent setting
* plot_ME_bias_communication.ipynb: visualizes the data from the two agent setting
* ME_data_analysis_and_visualization_functions.py: the functions for plotting the data and caluclating the ME indices are outsourced to this file. 

The plots are apart from minor changes the figures used for the paper. 

#### convergence_time_scales 
This folder contains all code belonging to Experiment 2, which investigates the convergence time scales of literal and pragmatic agents for different types of input distributions. 
* convergence_time_labeling.ipynb: runs the simulations and saves the data for the single agent setting
* convergence_time_communication.ipynb: runs the simulations and saves the data for the two agent setting
* plot_learning_time_scales.ipynb: plots the data from both settings 

Also here, the plots are apart from minor changes the figures used for the paper. 

### 4. Practical information  
When applying the model to larger systems or when using extreme values for the alpha parameter numerical instabilities can arise. This can be fixed by minor changes to the agents' implementations. Simply add a very small epsilon value (e.g. 10^-10) to prevent dividing by zero or taking the logarithm of zero. Make sure that epsilon is small enough so it does not interfere with the calculation. As a side note on that: if you add epsilon to avoid taking the logarithm of zero in the sampling function, do so already on the return value of the call function to allow for correct computation of the gradient. We did not implement such a fix as it is not required for the simulations in the paper. 
