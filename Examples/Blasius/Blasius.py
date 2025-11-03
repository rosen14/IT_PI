#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numpy.random as rng
from scipy.integrate import odeint
from scipy.optimize import minimize, root
import matplotlib.pyplot as plt

import sys
import os
current_notebook_dir = os.getcwd()
# Replace with the path to your IT_PI.py function
project_root_dir = os.path.join(current_notebook_dir, '..', '..')
it_pi_module_dir = project_root_dir
sys.path.append(it_pi_module_dir)
import IT_PI_parallel as IT_PI
from numpy.linalg import inv, matrix_rank
from pprint import pprint
plt.rcParams['font.family'] = 'Times New Roman'  # Set the font to Times New Roman
plt.rcParams['text.usetex'] = True 


# In[3]:


##Generate the data
eta_inf = 10;  # Arbitrary "infinite" upper limit for domain
d_eta = 0.01;   # Step size
eta = np.arange(0, eta_inf, d_eta)

F_init = [0, 0, 0] # Initial guess for unknown initial condition
def blasius_rhs(f):
    """RHS of Blasius equation recast as first order nonlinear ODE
    f[0] = f
    f[1] = f'
    f[2] = f''
    """
    return np.array([f[1], f[2], -f[0]*f[2]/2])

def bc_fn(f0):
    """Solve with unknown initial condition as guess and evaluate at upper boundary"""
    global eta
    f = odeint(lambda f, t: blasius_rhs(f), f0, eta)
    # return discrepancy between upper boundary and desired f[2] = 1
    return [f0[0], f0[1], f[-1, 1] - 1]
# Solve root-finding problem for unknown initial condition
opt_res = root(bc_fn, F_init, tol=1e-4)
F0 = [0, 0, opt_res.x[2]]

# Evaluate with resulting initial conditions
f = odeint(lambda y, t: blasius_rhs(y), F0, eta)
x = np.linspace(1e-3, 1e-1, 100)
nu = 1e-6       # Viscosity of water near room temperature  (m^2/s)
U_inf = 0.01    # m/s

Re = (U_inf/nu)*x
delta = 1.72 * np.sqrt(x*nu/U_inf)

plt.figure(figsize=(6, 2))
plt.plot(x, delta)
plt.grid()
plt.xlabel('$x$')
plt.ylabel(r'$\delta$')

y = np.linspace(1e-4, 2*max(delta), 100)

yy, xx = np.meshgrid(y, x)

u = np.zeros([len(x), len(y)])
v = np.zeros(u.shape)

eta = yy*np.sqrt(U_inf/(xx*nu))  # Exact value of eta

for i in range(len(x)):
    f = odeint(lambda y, t: blasius_rhs(y), F0, eta[i, :])
    u[i, :] = U_inf * f[:, 1]
    v[i, :] = (0.5*U_inf/np.sqrt(Re[i])) * (eta[i, :]*f[:, 1] - f[:, 0])


# In[4]:


plt.figure(figsize=(6, 2))
plt.pcolor(x, y, u.T, shading='auto', cmap='bone')
plt.plot(x, delta, c='w')
plt.xlabel("$x$")
plt.ylabel("$y$")


# In[5]:


q = np.vstack([u.flatten()/U_inf, v.flatten()/U_inf]).T
# All parameters
p = np.vstack([np.full(u.shape, U_inf).flatten(),
               np.full(u.shape, nu).flatten(),xx.flatten(),
               yy.flatten()]).T


# In[6]:


Y                = q[:,0]
Y = Y.reshape(-1, 1)
X                = p
variables_tauw   = [ 'U','\\nu','x','y'];                       #Define variable name
D_in             = np.matrix(' 1 2 1 1; -1 -1 0 0')                           #Define D_in matrix 
num_input        = 1


# In[7]:


print("Rank of D_in:", matrix_rank(D_in))
print("D_in matrix:\n", D_in)
num_rows          = np.shape(D_in)[0]
num_cols          = np.shape(D_in)[1]
# Function to calculate basis matrices

# Generate basis matrices
num_basis        = D_in.shape[1] -matrix_rank(D_in)
basis_matrices   = IT_PI.calc_basis(D_in, num_basis)
print("Basis vectors:")
pprint(basis_matrices)


# In[8]:


# Run dimensionless learning
results = IT_PI.main(
    X,
    Y.reshape(-1, 1),
    basis_matrices,
    num_input=num_input,
    estimator="binning",
    estimator_params={"num_bins": 60},
    seed=42
)


input_PI = results["input_PI"]
epsilon  = results["irreducible_error"]
uq       = results["uncertainty"]


# In[9]:


output_PI = results["output_PI"]
#IT_PI.plot_scatter(input_PI,output_PI)
fig = plt.figure(figsize=(4, 4))
plt.scatter((input_PI[:500]),(output_PI[:500]),s=10)
plt.xlabel(r" $\Pi^*$", fontsize=25, labelpad=10)  
plt.ylabel(r" $\Pi_o^*$", fontsize=25, labelpad=10)
plt.xlim([0,10])
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
fig.savefig("Blasius.eps", format="eps", bbox_inches="tight",transparent = "True")


# In[10]:


# import pickle

# workspace = {}
# globals_copy = dict(globals())

# for key, value in globals_copy.items():
#     if not key.startswith('_') and not callable(value) and not hasattr(value, '__module__'):
#         try:
#             pickle.dumps(value)  # Test if it can be pickled
#             workspace[key] = value
#         except (TypeError, AttributeError):
#             print(f"Skipping {key} - cannot pickle")

# with open('output.pkl', 'wb') as f:
#     pickle.dump(workspace, f)

