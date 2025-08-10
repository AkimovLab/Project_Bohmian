import torch
import libra_py.dynamics.bohmian.compute as compute
import libra_py.dynamics.bohmian.plot as plot
#import matplotlib.pyplot as plt

def sech(x):
  return 1 / torch.cosh(x)
    
def Martens_model1(q, params):
    """
    q - Tensor(ntraj, ndof)
    """
    #params = {"Va": 0.00625, "Vb": 0.0106}
    return torch.sum( params["Va"] * sech(2.0*q[:, 0])**2 + 0.5 * params["Vb"] * q[:, 1]**2 , 0)

def Martens_model2(q, params):
    """
    q - Tensor(ntraj, ndof)

    Martens_model1 is just this one but with Vc = 0.0
    """
    #params = {"Va": 0.00625, "Vb": 0.0106}
    Va = params["Va"]
    Vb = params["Vb"]
    Vc = params["Vc"]
    return torch.sum( Va * sech(2.0*q[:, 0])**2 + 0.5 * Vb * (q[:, 1] + Vc * (q[:, 0]**2 - 1.0 ) )**2 , 0)


import argparse
parser = argparse.ArgumentParser(description='Bohmian...')
parser.add_argument('--opt', type=int)
args = parser.parse_args()


opt = args.opt
ntraj = 500
#  opt   |  1     2     3     4    
#  sigma | 0.1   0.5   0.05  0.05
#  suff  |  1     2     3     3b
#   dt   | 5.0   5.0    5.0   2.5
#  nsteps| 400   400    400   800

sigma, suff, dt, nsteps, which = None, None, None, None, None

if opt==1:
    sigma, suff, dt, nsteps, which = 0.1, "1", 5.0, 400, [0, 100, 200, 300, 399]
elif opt==2:
    sigma, suff, dt, nsteps, which = 0.5, "2", 5.0, 400, [0, 100, 200, 300, 399]
elif opt==3:
    sigma, suff, dt, nsteps, which = 0.05, "3", 5.0, 400, [0, 100, 200, 300, 399]
elif opt==4:
    sigma, suff, dt, nsteps, which = 0.05, "3b", 2.5, 800, [0, 200, 400, 600, 799]


#========== Case 1 ==========
params = {"nsteps": nsteps, "dt": dt, "print_period":8,
          "ham": Martens_model2,  "ham_params": {"Va": 0.00625, "Vb": 0.0106, "Vc":0.0 },
          "do_bohmian": 1,  "tbf_type":compute.rho_mv_cauchy, "qpot_sigmas": sigma
         }

prefix = F"bohmian-mv_cauchy-MartensI-case1-{suff}"
prms = dict(params); prms.update({"prefix":prefix})
q, p, masses = compute.init_variables(ntraj, 1)
compute.md(q, p, masses, prms)
plot.plot({ "filename":F"{prefix}.pt", "prefix":prefix, "do_show":False, "which_timesteps":which })


#========== Case 2 ==========
params = {"nsteps": nsteps, "dt": dt, "print_period":8,
          "ham": Martens_model2,  "ham_params": {"Va": 0.00625, "Vb": 0.0106, "Vc":0.0 },
          "do_bohmian": 1,  "tbf_type":compute.rho_mv_cauchy, "qpot_sigmas": sigma
         }

prefix = F"bohmian-mv_cauchy-MartensI-case2-{suff}"
prms = dict(params); prms.update({"prefix":prefix})
q, p, masses = compute.init_variables(ntraj, 2)
compute.md(q, p, masses, prms)
plot.plot({ "filename":F"{prefix}.pt", "prefix":prefix, "do_show":False, "which_timesteps":which })

#========== Case 3 ==========
params = {"nsteps": nsteps, "dt": dt, "print_period":8,
          "ham": Martens_model2,  "ham_params": {"Va": 0.00625, "Vb": 0.0106, "Vc":0.4 },
          "do_bohmian": 1,  "tbf_type":compute.rho_mv_cauchy, "qpot_sigmas": sigma
         }

prefix = F"bohmian-mv_cauchy-MartensII-case1-{suff}"
prms = dict(params); prms.update({"prefix":prefix})
q, p, masses = compute.init_variables(ntraj, 1)
compute.md(q, p, masses, prms)
plot.plot({ "filename":F"{prefix}.pt", "prefix":prefix, "do_show":False, "which_timesteps":which })

#========== Case 4 ==========
params = {"nsteps": nsteps, "dt": dt, "print_period":8,
          "ham": Martens_model2,  "ham_params": {"Va": 0.00625, "Vb": 0.0106, "Vc":0.4 },
          "do_bohmian": 1,  "tbf_type":compute.rho_mv_cauchy, "qpot_sigmas": sigma
         }

prefix = F"bohmian-mv_cauchy-MartensII-case2-{suff}"
prms = dict(params); prms.update({"prefix":prefix})
q, p, masses = compute.init_variables(ntraj, 2)
compute.md(q, p, masses, prms)
plot.plot({ "filename":F"{prefix}.pt", "prefix":prefix, "do_show":False, "which_timesteps":which })


