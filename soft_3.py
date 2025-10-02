import torch
import torch.fft
import libra_py.dynamics.exact_torch.compute as compute



def sech(x):
  return 1 / torch.cosh(x)

def Martens_model(q, params):
    """
    q - Tensor(ndof)

    Martens_model1 is just this one but with Vc = 0.0
    """
    #params = {"Va": 0.00625, "Vb": 0.0106}
    Va = params.get("Va", 0.00625)
    Vb = params.get("Vb", 0.0106)
    Vc = params.get("Vc", 0.0)
    return Va * (sech(2.0*q[0]))**2 + 0.5 * Vb * (q[1] + Vc * (q[0]**2 - 1.0 ) )**2


def double_well_2D(q, params):
    """
    2D double well with 4 minima and coupling term.

    q : Tensor (ntraj, 2)
    params : dict, e.g. {"a": 0.005, "b": 0.015, "c": 0.005}
    """
    a = params.get("a", 0.005)
    b = params.get("b", 0.015)
    c = params.get("c", 0.005)

    x = q[0]
    y = q[1]

    return a * (x**2 - 1.0)**2 + b * (y**2 - 1.0)**2 + c * x**2 * y**2


m = 2000.0
w = 0.02

params_I_1 = { "grid_size":[1024, 256], "prefix":"MartensI-case1-exact",
              "q_min":[-30.0, -5.0], "q_max":[30.0, 5.0],
              "dt": 1.0,  "nsteps":4000, "save_every_n_steps":10,
              "mass": [m, m],
              "potential_fn":Martens_model, "potential_fn_params":{"Va":0.00625, "Vb":0.0106, "Vc":0.0},
              "psi0_fn": compute.gaussian_wavepacket, 
              "psi0_fn_params":{ "mass": [m, m], "omega":[w, w], "q0":[-5.0, 0.0], "p0":[1.0, 0.0] }
             }

params_I_2 = { "grid_size":[1024, 256], "prefix":"MartensI-case2-exact",
              "q_min":[-30.0, -5.0], "q_max":[30.0, 5.0],
              "dt": 1.0,  "nsteps":4000, "save_every_n_steps":10,
              "mass": [m, m],
              "potential_fn":Martens_model, "potential_fn_params":{"Va":0.00625, "Vb":0.0106, "Vc":0.0},
              "psi0_fn": compute.gaussian_wavepacket,
              "psi0_fn_params":{ "mass": [m, m], "omega":[w, w], "q0":[-5.0, 0.0], "p0":[5.0, 0.0] }
             }


params_II_1 = { "grid_size":[256, 256], "prefix":"MartensII-case1-exact",
              "q_min":[-5.0, -5.0], "q_max":[5.0, 5.0],
              "dt": 1.0,  "nsteps":4000, "save_every_n_steps":10,
              "mass": [m, m],
              "potential_fn":Martens_model, "potential_fn_params":{"Va":0.00625, "Vb":0.0106, "Vc":0.4},
              "psi0_fn": compute.gaussian_wavepacket,
              "psi0_fn_params":{ "mass": [m, m], "omega":[w, w], "q0":[-2.0, -1.0], "p0":[1.0, 1.0] }
             }

params_II_2 = { "grid_size":[256, 256], "prefix":"MartensII-case2-exact",
              "q_min":[-5.0, -5.0], "q_max":[5.0, 5.0],
              "dt": 1.0,  "nsteps":4000, "save_every_n_steps":10,
              "mass": [m, m],
              "potential_fn":Martens_model, "potential_fn_params":{"Va":0.00625, "Vb":0.0106, "Vc":0.4},
              "psi0_fn": compute.gaussian_wavepacket,
              "psi0_fn_params":{ "mass": [m, m], "omega":[w, w], "q0":[-2.0, -1.0], "p0":[2.0, 2.0] }
             }


params_III_1 = { "grid_size":[256, 256], "prefix":"Double_well_2D-case1-exact",
              "q_min":[-5.0, -5.0], "q_max":[5.0, 5.0],
              "dt": 1.0,  "nsteps":4000, "save_every_n_steps":10,
              "mass": [m, m],
              "potential_fn":double_well_2D, "potential_fn_params":{"a":0.005, "b":0.015, "c":0.005 },
              "psi0_fn": compute.gaussian_wavepacket,
              "psi0_fn_params":{ "mass": [m, m], "omega":[w, w], "q0":[-1.0, -1.0], "p0":[1.0, 2.0] }
             }

params_III_2 = { "grid_size":[256, 256], "prefix":"Double_well_2D-case2-exact",
              "q_min":[-5.0, -5.0], "q_max":[5.0, 5.0],
              "dt": 1.0,  "nsteps":4000, "save_every_n_steps":10,
              "mass": [m, m],
              "potential_fn":double_well_2D, "potential_fn_params":{"a":0.005, "b":0.015, "c":0.005 },
              "psi0_fn": compute.gaussian_wavepacket,
              "psi0_fn_params":{ "mass": [m, m], "omega":[w, w], "q0":[-1.0, -1.0], "p0":[2.0, 1.0] }
             }


solver1 = compute.exact_tdse_solver(params_I_1)
solver1.solve()

solver2 = compute.exact_tdse_solver(params_I_2)
solver2.solve()

solver3 = compute.exact_tdse_solver(params_II_1)
solver3.solve()

solver4 = compute.exact_tdse_solver(params_II_2)
solver4.solve()

solver5 = compute.exact_tdse_solver(params_III_1)
solver5.solve()

solver6 = compute.exact_tdse_solver(params_III_2)
solver6.solve()

