# SwiftDiffusion
*Equivariant Graph Diffusion Network Inference optimized for Apple Silicon*
                                                                
SwiftDiffusion is a low-level from-scratch implementation of an Equivariant Graph Neural Network (EGNN) paired with a Denoising Diffusion Probabilistic Model (DDPM), written entirely in Swift and Metal Shading Language.
It generates 3D molecular structures by simulating atomic forces and thermodynamics directly on the GPU, with zero reliance on heavy Python frameworks during inference.

## Features
- **Pure Metal Compute Pipeline:** Custom MSL kernels for graph message passing, continuous time embeddings, and linear neural network layers.
- **Langevin Thermodynamics:** Native GPU implementation of the Box-Muller transform for stochastic Gaussian noise generation and simulated annealing.
- **Hardware-Level Physics:** Incorporates hard-coded physical boundary conditions, including Pauli Exclusion (nuclear repulsion) and containment fields, to prevent singularity collapse and kinetic escape. Keeps the central atom (Carbon) locked to the origin (0.0, 0.0, 0.0) to provide a stable frame of reference for the expanding hydrogen cloud, effectively isolating the rigid body translation.
- Runs natively on MacOS (only) using standard Apple Metal libraries


## Example Output
Generating a 5-node molecular graph (Methane, CH4) from pure noise, matching the equilibrium state of PyTorch weights on Apple Silicon.

```
Node 0 [Carbon]: ( 0.0003,  0.0001, -0.0000)
Node 1 [Hydrogen]: ( 0.8087, -0.5723,  1.4417)
Node 2 [Hydrogen]: (-1.7979,  0.2439, -0.0153)
Node 3 [Hydrogen]: ( 0.1712,  0.0805, -1.5560)
Node 4 [Hydrogen]: ( 0.8177,  0.2478,  0.1297)
```

## Architecture
- **Weight Loading:** Reads raw .bin weight buffers seamlessly exported from a [trained PyTorch model](https://github.com/Raiden-Makoto/QuantumEDM/tree/main)
- **Graph Setup:** Initializes nodes, edges, and atom types into shared Apple Silicon memory.
- **Compute Loop:** Dispatches 2,500 iterations of message-passing, coordinate updates, and residual feature additions in a highly optimized GPU loop.
