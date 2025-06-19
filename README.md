# Project_Bohmian
Quantum trajectory calculations made stable with Lorentzian trajectory basis functions


soft.py - run exact calculations

run.py - classical MD

run_lorentz.py -  sigma = 0.1

run_lorentz2.py - sigma = 0.5 - leads to smaller transmission probabilities

run_lorentz3.py - sigma = 0.05 - nearly the same as 0.1, but starts showing energy conservation problem

run_lorentz3b.py - sigma = 0.05, but the integration timestep is half of that in `run_lorentz3.py` while 
             the number of steps is increased to make the total time the same

Once you have run all the calculations (all the above scripts one by one), you can
start plotting the results using this Jupyter notebook:

plot-all.ipynb
