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



params_I_1 = { "grid_size":[512, 256], "prefix":"MartensI-case1-exact",
              "q_min":[-10.0, -5.0], "q_max":[20.0, 5.0],
              "dt": 5.0,  "nsteps":300, "save_every_n_steps":10,
              "mass": [2000.0, 2000.0],
              "potential_fn":Martens_model, "potential_fn_params":{"Va":0.00625, "Vb":0.0106, "Vc":0.0},
              "psi0_fn": compute.gaussian_wavepacket, 
              "psi0_fn_params":{ "mass": [2000.0, 2000.0], "omega":[0.004, 0.004], "q0":[-1.0, 0.0], "p0":[3.0, 0.0] }
             }

params_I_2 = { "grid_size":[512, 256], "prefix":"MartensI-case2-exact",
              "q_min":[-10.0, -5.0], "q_max":[20.0, 5.0],
              "dt": 5.0,  "nsteps":300, "save_every_n_steps":10,
              "mass": [2000.0, 2000.0],
              "potential_fn":Martens_model, "potential_fn_params":{"Va":0.00625, "Vb":0.0106, "Vc":0.0},
              "psi0_fn": compute.gaussian_wavepacket,
              "psi0_fn_params":{ "mass": [2000.0, 2000.0], "omega":[0.004, 0.004], "q0":[-1.0, 0.0], "p0":[4.0, 0.0] }
             }


params_II_1 = { "grid_size":[256, 256], "prefix":"MartensII-case1-exact",
              "q_min":[-5.0, -5.0], "q_max":[5.0, 5.0],
              "dt": 5.0,  "nsteps":300, "save_every_n_steps":10,
              "mass": [2000.0, 2000.0],
              "potential_fn":Martens_model, "potential_fn_params":{"Va":0.00625, "Vb":0.0106, "Vc":0.4},
              "psi0_fn": compute.gaussian_wavepacket,
              "psi0_fn_params":{ "mass": [2000.0, 2000.0], "omega":[0.004, 0.004], "q0":[-1.0, 0.0], "p0":[3.0, 0.0] }
             }

params_II_2 = { "grid_size":[256, 256], "prefix":"MartensII-case2-exact",
              "q_min":[-5.0, -5.0], "q_max":[5.0, 5.0],
              "dt": 5.0,  "nsteps":300, "save_every_n_steps":10,
              "mass": [2000.0, 2000.0],
              "potential_fn":Martens_model, "potential_fn_params":{"Va":0.00625, "Vb":0.0106, "Vc":0.4},
              "psi0_fn": compute.gaussian_wavepacket,
              "psi0_fn_params":{ "mass": [2000.0, 2000.0], "omega":[0.004, 0.004], "q0":[-1.0, 0.0], "p0":[4.0, 0.0] }
             }



solver1 = compute.exact_tdse_solver(params_I_1)
solver1.solve()

solver2 = compute.exact_tdse_solver(params_I_2)
solver2.solve()

solver3 = compute.exact_tdse_solver(params_II_1)
solver3.solve()

solver4 = compute.exact_tdse_solver(params_II_2)
solver4.solve()

