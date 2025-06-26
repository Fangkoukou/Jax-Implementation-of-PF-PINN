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
- Stores all model parameters (for PDE)
- Acts as a global configuration module
  
### `utils`
- Contains helper functions for:
  - PDE residual calculation
  - Initial and boundary condition evaluation

### `variable`
- A wrapper class built on `SimpleNamespace`
- Facilitates organized access to:
  - Initial conditions (IC)
  - Boundary conditions (BC)
  - Collocation points
  - Adaptive sampling points

### `sampler`
- Handles sampling logic for training:
  - Domain sampling
  - Adaptive sampling strategies (if applicable)

### `residual`
- Responsible for computing the residuals of:
  - Allen-Cahn equation
  - Cahn-Hilliard equation
- Utilizes automatic differentiation in JAX for accurate gradients

### 'train'
- the training file:
  - contains model definition and training loop
---

## Features

- Built using **JAX** for high-performance automatic differentiation and GPU acceleration.
- Modular and extendable design, modify residual class for different 1D differential equations.
- Supports adaptive sampling and residual-based training.
