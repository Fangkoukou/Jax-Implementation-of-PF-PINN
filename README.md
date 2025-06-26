# Jax-Implementation-of-PF-PINN
This is the jax implementation of PF-PINN, solving 1D coupled AC-CH phase field equation

This work implementes the PF-PINN described in the following paper 
"PF-PINNs: Physics-informed neural networks for solving coupled Allen-Cahn and Cahn-Hilliard phase field equations"
link to the paper: https://www.sciencedirect.com/science/article/pii/S0021999125001263
The original code can be found at https://github.com/NanxiiChen/PF-PINNs/tree/main

This rewrite contains the following modules:
utils: contains all the helper functions used for pde residual calculation, as well as for initial/boundary conditions.
variable: a wrapper class of SimpleNamespace, designed to hold data while providing easy access to different sets (ic/bc/colloc/adapt)
sampler: designed for sampling purpose
residual: designed for residual computation
params: a set of model parameters, used as global variables
