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


#========== Case 1 ==========
ntraj = 500
params = {"nsteps": 400, "dt": 5.0, "print_period":8,
          "ham": Martens_model2,  "ham_params": {"Va": 0.00625, "Vb": 0.0106, "Vc":0.0 },
          "do_bohmian": 1,  "tbf_type":compute.rho_lorentzian, "qpot_sigmas": torch.tensor([ [0.5, 0.5]] * ntraj)
         }


prefix = "bohmian-lorentzian-MartensI-case1-2"
prms = dict(params); prms.update({"prefix":prefix})
q, p, masses = compute.init_variables(ntraj, 1)
compute.md(q, p, masses, prms)
plot.plot({ "filename":F"{prefix}.pt", "prefix":prefix, "do_show":False, "which_timesteps":[0, 100, 200, 300, 399] })


#========== Case 2 ==========
ntraj = 500
params = {"nsteps": 400, "dt": 5.0, "print_period":8,
          "ham": Martens_model2,  "ham_params": {"Va": 0.00625, "Vb": 0.0106, "Vc":0.0 },
          "do_bohmian": 1,  "tbf_type":compute.rho_lorentzian, "qpot_sigmas": torch.tensor([ [0.5, 0.5]] * ntraj)
         }


prefix = "bohmian-lorentzian-MartensI-case2-2"
prms = dict(params); prms.update({"prefix":prefix})
q, p, masses = compute.init_variables(ntraj, 2)
compute.md(q, p, masses, prms)
plot.plot({ "filename":F"{prefix}.pt", "prefix":prefix, "do_show":False, "which_timesteps":[0, 100, 200, 300, 399] })

#========== Case 3 ==========
ntraj = 500
params = {"nsteps": 400, "dt": 5.0, "print_period":8,
          "ham": Martens_model2,  "ham_params": {"Va": 0.00625, "Vb": 0.0106, "Vc":0.4 },
          "do_bohmian": 1,  "tbf_type":compute.rho_lorentzian, "qpot_sigmas": torch.tensor([ [0.5, 0.5]] * ntraj)
         }


prefix = "bohmian-lorentzian-MartensII-case1-2"
prms = dict(params); prms.update({"prefix":prefix})
q, p, masses = compute.init_variables(ntraj, 1)
compute.md(q, p, masses, prms)
plot.plot({ "filename":F"{prefix}.pt", "prefix":prefix, "do_show":False, "which_timesteps":[0, 100, 200, 300, 399] })


#========== Case 4 ==========
ntraj = 500
params = {"nsteps": 400, "dt": 5.0, "print_period":8,
          "ham": Martens_model2,  "ham_params": {"Va": 0.00625, "Vb": 0.0106, "Vc":0.4 },
          "do_bohmian": 1,  "tbf_type":compute.rho_lorentzian, "qpot_sigmas": torch.tensor([ [0.5, 0.5]] * ntraj)
         }


prefix = "bohmian-lorentzian-MartensII-case2-2"
prms = dict(params); prms.update({"prefix":prefix})
q, p, masses = compute.init_variables(ntraj, 2)
compute.md(q, p, masses, prms)
plot.plot({ "filename":F"{prefix}.pt", "prefix":prefix, "do_show":False, "which_timesteps":[0, 100, 200, 300, 399] })


