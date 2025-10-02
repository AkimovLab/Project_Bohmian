# Project_Bohmian
Quantum trajectory calculations made stable with Lorentzian trajectory basis functions


#================== Revision 2 ==============================

soft_3.py - run exact calculations

run_cauchy3.py - run Bohmian or classical simulations with the option parameterized by
                 the command-line input

run_all.py - run a series of Bohmian or classical calculations (by calling the `run_cauchy3` script)
             in a more automated way

Once you have run all the calculations (all the above scripts one by one), you can
start plotting the results using this Jupyter notebook:

plot_convergence.ipynb

The new set of exact calculations is too large but are fast to produce
by running soft_3.py, so they are not included in this revision - the .pt 
files already present are from the previous revision 


#================== Revision 1 ==============================
soft.py - run exact calculations

run.py - classical MD

run_cauchy.py - run Bohmian simulations with the option parameterized by 
                the command-line input

run_all.py - run a series of Bohmian calculations (by calling the `run_cauchy` script)
             in a more automated way

Once you have run all the calculations (all the above scripts one by one), you can
start plotting the results using this Jupyter notebook:

plot-all.ipynb

The `.pt` files included here contain the results ofr classical, Bohmian, and exact quantum 
calculations are are ready to be used for plotting


