# Solving the 1D Viscous Burgers' Equation using Physics-Informed Neural Networks (PINNs)

This repository contains a clean, self-contained, and Google Colab-optimized implementation of a **Physics-Informed Neural Network (PINN)** to solve the one-dimensional viscous Burgers' equation. By embedding the governing partial differential equation (PDE) directly into the neural network's loss function, the model acts as a data-efficient, mesh-free universal function approximator.

---

## Project Overview

Traditional numerical methods—such as Finite Difference (FDM) or Finite Element Methods (FEM)—require grid discretization and can struggle or become computationally expensive when handling sharp gradients or shock formations. 

This project implements an AI-driven, mesh-free alternative based on the foundational methodology introduced by Raissi et al. (2017):
> *Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2017). Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations. arXiv preprint arXiv:1711.10561.*

### The Governing Physics: Viscous Burgers' Equation
The 1D viscous Burgers' equation is a fundamental non-linear PDE used to model fluid mechanics, nonlinear acoustics, and gas dynamics:

$$u_t + u u_x - \nu u_{xx} = 0, \quad x \in [-1,1], \quad t \in [0,1]$$

**Problem Configurations:**
* **Latent Solution $u(t, x)$:** Represents the fluid velocity profile at any given time ($t$) and spatial coordinate ($x$).
* **Kinematic Viscosity ($\nu$):** Set to $\frac{0.01}{\pi}$. Small viscosity values make this problem notoriously hard for classical methods due to sudden shock formation.
* **Initial Condition (IC):** $u(0, x) = -\sin(\pi x)$
* **Dirichlet Boundary Conditions (BC):** $u(t, -1) = u(t, 1) = 0$

---

## Core Methodology & Architecture

Instead of relying purely on large labeled datasets, a PINN uses the underlying physical law as a **regularization agent** to constrain the network's mathematical search space.

### The Composite Loss Function
The network parameters are optimized by minimizing a total Mean Squared Error ($MSE$) loss:

$$\text{Total Loss} = MSE_u + MSE_f$$

1. **Supervised Data Loss ($MSE_u$):** Evaluates how well the network matches the known initial and boundary conditions from a small sample of points ($N_u = 100$).
2. **Physics Residual Loss ($MSE_f$):** Evaluates how well the network respects the fluid equation across thousands of unseen coordinates ($N_f = 10,000$) called **collocation points**. 

### Grid Independence via Automatic Differentiation
Using TensorFlow's `tf.GradientTape`, the model calculates the exact partial derivatives ($u_t, u_x, u_{xx}$) using the mathematical chain rule. This bypasses the need for any spatial discretization or mesh generation, allowing the model to predict fluid behavior at any continuous coordinate $(t, x)$.

---

## 📂 Code & Repository Structure

To eliminate common runtime bugs (such as `ModuleNotFoundError` or file streaming blocks), this repository uses a completely **self-contained data generation workflow** utilizing native Python libraries.
