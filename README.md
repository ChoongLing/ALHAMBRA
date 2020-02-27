This is my second PhD project.
I'm using synthesised spectroscopic data to predict ages and metallicites for ALHAMBRA survey data.

At the moment, I am using synthesised CALIFA data. Code using said data is in CAL_training/
Two Hyperas implementation files, for Age and Z. These allow for Set A and Set B-like splitting
(see Liew-Cain et al 2020 for details) by changing True <--> False.
The resulting parameters are given in Hyperas_Parameters.txt
Age_training.py and Z_training.py train the CNN on all of the CALIFA data (no validation set)
hyperparameters for CNN need to be changed manually.


