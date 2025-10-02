import os


# The new one:
for model in [1,2,3]:
    for icase in [1,2]:
        for ntraj in [50, 100, 250, 500, 1000]:
            for dt in [1.0]:
                for do_bohmian in [0, 1]:
                    nsteps = int(1.0 * 4000/dt)
                    os.system(F"python run_cauchy3.py --model={model} --icase={icase} --dt={dt} --nsteps={nsteps} --ntraj={ntraj} --do_bohmian={do_bohmian}")


