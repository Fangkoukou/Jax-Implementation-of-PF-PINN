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

### `params`
- A global configuration module storing all the parameters
  
### `derivative`
- A set of derivative functions implemented using JAX automatic differentiation

### `pde`
- A PDE module that contains PDE parameters and a set of helper functions in residual computation

### `residual`
- Responsible for residual, loss, loss weight computations, depends on `derivative` and `pde`
- 
### `sampler`
- Handles sampling logic for training, depends on `residual`

### `train`
- the training file:
  - contains model definition and training loop
---

## Features

- Built using **JAX** for high-performance automatic differentiation and GPU acceleration.
- Modular and extendable design, modify residual class for different 1D differential equations.
- Supports adaptive sampling and residual-based training.
