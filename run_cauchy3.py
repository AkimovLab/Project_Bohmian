import math
import argparse
import torch
import libra_py.dynamics.bohmian.compute as compute
import libra_py.dynamics.bohmian.plot as plot
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

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


def double_well_2D(q, params):
    """
    2D double well with 4 minima and coupling term.
    
    q : Tensor (ntraj, 2)
    params : dict, e.g. {"a": 0.005, "b": 0.015, "c": 0.005}
    """
    a = params.get("a", 0.005)
    b = params.get("b", 0.015)
    c = params.get("c", 0.005)
    
    x = q[:, 0]
    y = q[:, 1]
    
    return torch.sum( a * (x**2 - 1.0)**2 + b * (y**2 - 1.0)**2 + c * x**2 * y**2, 0)




def pick_initial_conditions(etarget, ham_function, ham_params, Q_mean, P_mean, mass=2000.0, omega=0.004, ntraj = 400, ntrials=500):
    """
    etarget - target quantum energy
    ntraj - how many trajectories to generate
    icond - 1 -for p = 3, 2 -for p=4
    ntrials - how many different samples of ntraj to try
    ham_function - classical potential
    ham_params - parameters for that function
    """
    
    # The standard deviation of the initial wavepacket 
    #mass = 200.0
    #omega = 0.01
    sigma_q = np.sqrt(0.5/(mass*omega))
    print(sigma_q)
    
    # Use the Scott's rule for kernel bandwidth
    d = 2
    sigma = sigma_q * ntraj**(-1.0/(d+4)) # Scott's rule
    #print(F"sigma = {sigma}")

    etot, ekin, epot = [], [], []
    Q, P, M = [], [], []
    """
    Q_mean, P_mean = None, None

    if model==1:
        if icond==1:
            Q_mean = [-1.0, 0.0]
            P_mean = [ 3.0, 0.0]
        elif icond==2:
            Q_mean = [-1.0, 0.0]
            P_mean = [ 4.0, 0.0]
    elif model==2:
            Q_mean = [-2.0,-1.0]
            P_mean = [ 3.0, 0.0]
        elif icond==2:
            Q_mean = [-2.0,-1.0]
            P_mean = [ 4.0, 0.0]
    """
    for _ in range(ntrials):
        #q, p, masses = compute.init_variables(ntraj, icond) # 1 is for icond = 1, 2 is for icond = 2

        q, p, masses = compute.init_variables(ntraj, mass, omega, Q_mean, P_mean)
        Q.append(q)
        P.append(p)
        M.append(masses)
        #u = compute.quantum_potential(q, sigma, masses, compute.rho_mv_cauchy)
        #equant.append(u.detach())
        E_kin = torch.sum( 0.5 * p**2/ masses)
        ekin.append(E_kin.detach())
        E_pot, _ = compute.compute_derivatives(q, ham_function, ham_params)
        epot.append( E_pot.detach())
        print(F"Ekin = {E_kin/ntraj}  Epot = {E_pot/ntraj} Total = {(E_kin + E_pot)/ntraj}")
        etot.append( ((E_kin + E_pot)/ntraj).detach() )
        
    indx = np.argmin( np.abs(np.array(etot) - np.array([etarget]*ntrials) ) )
    print("Energies at the selected point = ", etot[indx], ekin[indx]/ntraj, epot[indx]/ntraj)
    
    return Q[indx], P[indx], M[indx], sigma #, etot[indx], ekin[indx]/ntraj, epot[indx]/ntraj, equant[indx]/ntraj



def match_energy(etarget, ham_function, ham_params, Q_mean, P_mean, mass=2000.0, omega=0.004, ntraj = 400):
    
    # The standard deviation of the initial wavepacket 
    #mass = 200.0
    #omega = 0.01
    sigma_q = np.sqrt(0.5/(mass*omega))
    print(sigma_q)
    
    # Use the Scott's rule for kernel bandwidth
    d = 2
    sigma = sigma_q * ntraj**(-1.0/(d+4)) # Scott's rule
    #print(F"sigma = {sigma}")
    
    max_iter = 10
    iter = 0
    is_stop = False

    E_kin, E_pot = 0.0, 0.0
    q, p, masses = None, None, None
    """
    Q_mean, P_mean = None, None
    if model==1:
        if icond==1:
            Q_mean = [-1.0, 0.0]
            P_mean = [ 3.0, 0.0]
        elif icond==2:
            Q_mean = [-1.0, 0.0]
            P_mean = [ 4.0, 0.0]
    elif model==2:
            Q_mean = [-2.0,-1.0]
            P_mean = [ 3.0, 0.0]
        elif icond==2:
            Q_mean = [-2.0,-1.0]
            P_mean = [ 4.0, 0.0]
    """
    while is_stop==False and iter < max_iter:
        #q, p, masses = compute.init_variables(ntraj, icond) # 1 is for icond = 1, 2 is for icond = 2
        q, p, masses = compute.init_variables(ntraj, mass, omega, Q_mean, P_mean)

        E_kin = torch.sum( 0.5 * p**2/ masses).detach().item()/ntraj
        E_pot, _ = compute.compute_derivatives(q, ham_function, ham_params)
        E_pot = E_pot.detach().item()/ntraj
        iter = iter + 1
        if E_kin + E_pot < etarget:
            is_stop = True
        
    def error_funct(_sigma):
        u = compute.quantum_potential(q, _sigma, masses, compute.rho_mv_cauchy)
        u = u.detach().item()/ntraj
        etot = E_kin + E_pot + u
        return abs((etot - etarget)/etarget)
    
    res = minimize_scalar(lambda s: error_funct(s), bracket=( 0.001 , 1.0), method='Brent', options={"xtol": 1e-6, "maxiter": 1000})
    #res = minimize_scalar(lambda s: error_funct(s), bounds=( 0.001 , 1.0), method='bounded', options={"xatol": 1e-6, "maxiter": 1000})    

    print(F"optimized sigma = {res.x}")
    sigma = res.x
    print("Objective at sigma:", res.fun)
    print("Converged?", res.success, res.message)
    
    # Compute the energies:
    u = compute.quantum_potential(q, sigma, masses, compute.rho_mv_cauchy)
    u = u.detach().item()/ntraj
    E_tot = E_kin + E_pot + u
    
    print(E_kin, E_pot, u, E_tot)
    
    return q, p, masses, sigma 



# -------------------------
# Cauchy mixture on grid
# -------------------------
def rho_mv_cauchy_grid(points, Q, sigma):
    """
    Multivariate Cauchy mixture on grid (torch).
    points: (npts, ndof)
    Q: (ntraj, ndof)
    sigma: float
    returns: density values (npts,)
    """
    ntraj, ndof = Q.shape
    num = torch.lgamma(torch.tensor((ndof + 1) / 2.0))
    denom = torch.lgamma(torch.tensor(0.5)) + 0.5 * ndof * math.log(math.pi) + ndof * math.log(sigma)
    norm = torch.exp(num - denom)

    diff = points[:, None, :] - Q[None, :, :]  # (npts, ntraj, ndof)
    r2 = (diff**2).sum(dim=2)
    LZ = (1.0 + r2 / sigma**2) ** (-(ndof + 1) / 2.0)
    rho = (1.0 / ntraj) * norm * LZ.sum(dim=1)
    return rho

# Target Gaussian
def gaussian_2d(points, center, sigma):
    diff = points - center.unsqueeze(0)
    r2 = (diff**2).sum(dim=1)
    norm = 1.0 / (2 * math.pi * sigma**2)
    return norm * torch.exp(-0.5 * r2 / sigma**2)


def match_density(Q_mean, mass=2000.0, omega=0.004, ntraj=100):
    ndof = 2
    target_center = torch.tensor([-1.0, 0.0])
    #Q, p, masses = compute.init_variables(ntraj, 1) # 1 is for icond = 1, 2 is for icond = 2
    
    #target_sigma = 0.25
    #mass = 200.0
    #omega = 0.01
    target_sigma = np.sqrt(0.5/(mass*omega))

    Q, p, masses = compute.init_variables(ntraj, mass, omega, Q_mean, [3.0, 0.0])

    # -------------------------
    # Grid setup
    # -------------------------
    nx = ny = 120
    X, Y = target_center[0], target_center[1] 
    xlim, ylim = (-1.5 + X, 1.5 + X), (-1.5 + Y, 1.5 + Y)
    xs = torch.linspace(*xlim, nx)
    ys = torch.linspace(*ylim, ny)
    dx, dy = (xs[1] - xs[0]).item(), (ys[1] - ys[0]).item()
    Xg, Yg = torch.meshgrid(xs, ys, indexing="xy")
    grid_pts = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)

    target_vals = gaussian_2d(grid_pts, target_center, target_sigma)
    target_vals = target_vals / (target_vals.sum() * dx * dy)

    # -------------------------
    # Loss function
    # -------------------------
    def loss_for_sigma(sigma: float) -> float:
        est = rho_mv_cauchy_grid(grid_pts, Q, sigma)
        est = est / (est.sum() * dx * dy)
        return float(torch.mean((est - target_vals) ** 2).item())

    # -------------------------
    # Optimize sigma (scipy on torch loss)
    # -------------------------
    res = minimize_scalar(loss_for_sigma, bounds=(1e-3, 2.0), method="bounded")
    opt_sigma = res.x
    print("Optimized sigma:", opt_sigma, "loss:", res.fun)
    
    return Q, p, masses, opt_sigma



parser = argparse.ArgumentParser(description='Bohmian...')
#parser.add_argument('--opt', type=int)
parser.add_argument('--ntraj', type=int, default=100)
parser.add_argument('--dt', type=float, default=2.5)
parser.add_argument('--nsteps', type=int, default=800)
parser.add_argument('--icase', type=int, default=1)
parser.add_argument('--model', type=int, default=1)
parser.add_argument('--do_bohmian', type=int, default=1)

args = parser.parse_args()

ntraj = args.ntraj
dt = args.dt
nsteps = args.nsteps
icase = args.icase   # 1 - p=3,  2 - p=4 
model = args.model   # 1 = Martes I, 2 = Martens II
do_bohmian = args.do_bohmian

print(F"Running script for: ntraj = {ntraj} dt = {dt} nsteps = {nsteps} model={model} icase = {icase}")


mdl = ""
ham_function, ham_params = None, {}
Q_mean, P_mean = None, None

if model==1:
    mdl = "MartensI"
    ham_params = {"Va": 0.00625, "Vb": 0.0106, "Vc":0.0 }
    ham_function = Martens_model2
    Q_mean = [-5.0, 0.0]

    if icase==1:
        P_mean = [1.0, 0.0]
    elif icase==2:
        P_mean = [5.0, 0.0]
    
elif model==2:
    mdl = "MartensII"
    ham_params = {"Va": 0.00625, "Vb": 0.0106, "Vc":0.4 }
    ham_function = Martens_model2
    Q_mean = [-2.0, -1.0]

    if icase==1:
        P_mean = [1.0, 1.0]
    elif icase==2:
        P_mean = [2.0, 2.0]

elif model==3:
    mdl = "Double_well_2D"
    ham_params = {"a": 0.005, "b": 0.015, "c":0.005 }
    ham_function = double_well_2D

    Q_mean = [-1.0, -1.0]
    if icase==1:
        P_mean = [1.0, 2.0]
    elif icase==2:
        P_mean = [2.0, 1.0]


x = torch.load(F"{mdl}-case{icase}-exact.pt")
etarget = x["total_energy"][0].item()
print("Target exact total energy = {etarget}")


params = {"nsteps": nsteps, "dt": dt, "print_period":8,
          "ham": ham_function,  "ham_params": ham_params,
          "do_bohmian": do_bohmian,  "tbf_type":compute.rho_mv_cauchy
         }


print("Picking initial conditions...")
q, p, masses, sigma = None, None, None, 0.01
prefix = ""
if do_bohmian==0:
    #q, p, masses, sigma = pick_initial_conditions(etarget, ham_function, ham_params, Q_mean, P_mean, 20000.0, 0.004, ntraj, ntrials=500)
    q, p, masses = compute.init_variables(ntraj, 2000.0, 0.02, Q_mean, P_mean )
    prefix = F"results/classical-{mdl}-case{icase}-ntraj{ntraj}-dt{dt}"
else:
    # Matching energy
    q, p, masses, sigma = match_energy(etarget, ham_function, ham_params, Q_mean, P_mean, 2000.0, 0.02, ntraj)
    prefix = F"results/bohmian-mv_cauchy-{mdl}-case{icase}-ntraj{ntraj}-dt{dt}"

    #q, p, masses, sigma = match_density(ntraj)
    #prefix = F"results/dm-new-bohmian-mv_cauchy-{mdl}-case{icase}-ntraj{ntraj}-dt{dt}"


params.update( {"qpot_sigmas":sigma, "prefix":prefix} )

print(F"sigma = {sigma}") #  etot={etot}  ekin={ekin} epot={epot} equant={equant}")

#########################################################################################
print(F"Running MD with params = {params}")
torch.save(params, F"{prefix}-params.pt")
compute.md(q, p, masses, params)


########################################################################################
print(F"Plotting...")
which = [0, nsteps/4, nsteps/2, 3*nsteps/4, nsteps-1]
plot.plot({ "filename":F"{prefix}.pt", "prefix":prefix, "do_show":False, "which_timesteps":which })


