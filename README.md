# JAX Implementation of PF-PINN

This repository contains a JAX implementation of **PF-PINN**, a Physics-Informed Neural Network framework designed to solve the **1D coupled Allen-Cahn (AC) and Cahn-Hilliard (CH) phase field equations**.

## Reference Paper

This work implements the PF-PINN methodology as described in the paper:

> **PF-PINNs: Physics-informed neural networks for solving coupled Allen-Cahn and Cahn-Hilliard phase field equations**  
> [Journal of Computational Physics, 2025]  
> 📎 [Read the paper](https://www.sciencedirect.com/science/article/pii/S0021999125001263)

Original repository by the authors:  
[GitHub: NanxiiChen/PF-PINNs](https://github.com/NanxiiChen/PF-PINNs/tree/main)

---

## Project Structure

This rewritten version is modularized into the following components:

### `pde`
- A PDE module for coupled AC-CH equations, initialized using the set of physical parameters and numerical parameters
    - The physical parameters includes: alpha_phi, omega_phi, L, M, A, C_{Se}, C_{Le}
    - The numerical parameters includes: x_range, t_range (in physical units) and nx, nt, the number of discritization
- Contains a `solve` function that solves the system using `diffrax`
- Contains two graphing function `draw_heatmap` and `draw_profiles` for solution visulization

### `model`
- The model module, a simple Multilayer perceptron implemented using Equinox
    - Maps the normalized coordinates (x,t) to physical (phi, c)

### `derivative`
- A set of derivative functions implemented using JAX automatic differentiation
    - Computes the derivative of model wrt the input x,t
    - Takes care of normalization, the returned derivatives are in physical units

### `residual`
- Responsible for residual, loss, loss weight computations, depends on `derivative` and `pde`
  
### `sampler`
- Handles sampling logic for training, depends on `residual`

### `train`
- the training file, contains the training loop
---

## Features

- Built using **JAX** for high-performance automatic differentiation and GPU acceleration.
- Modular and extendable design, modify residual class for different 1D differential equations.
- Supports adaptive sampling and residual-based training.
