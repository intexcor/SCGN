# SCGN: Spherical Chamfer Generative Networks

**A 1-Step, Non-Adversarial, Unsupervised Generative Model based on Joint Hypersphere Topology.**

![MNIST Results](samples_step_50000.png)

## Overview
Current generative models (GANs, Diffusion, Flow Matching) rely heavily on adversarial networks, computationally expensive multi-step ODE/SDE solvers, or discrete Optimal Transport algorithms (like Sinkhorn). 

**SCGN (Spherical Chamfer Generative Networks)** challenges this paradigm by introducing a purely geometric approach to generative modeling. It maps random noise directly to data in a **single forward pass** without any adversarial components, using a novel **Joint Space Chamfer Distance** on a hypersphere.

## The Mathematical Breakthrough
When mapping noise to images using pure pixel-wise distance metrics (like MSE or Chamfer), networks traditionally collapse into producing "Gray Mush" (averaging the dataset) or experience severe Mode Collapse.

SCGN solves this through two topological constraints:

1. **The Spherical Manifold:** 
   By projecting both the generated images and the real images onto an $N$-dimensional hypersphere of radius $R = \sqrt{C \times H \times W}$, we prevent the network from collapsing into the mean (the origin). The network is forced to maintain high variance and contrast.

2. **Joint Space Unconditional Chamfer Distance:**
   To induce semantic clustering and avoid generating generic "safe" smudges, SCGN computes the distance matrix in a *Joint Space* consisting of the image pixels concatenated with a continuous condition embedding (e.g., CLIP embeddings or pseudo-labels). 
   
   $$ C_{ij} = (1 - \cos(\theta_{img})) + \gamma (1 - \cos(\theta_{cond})) $$

   By minimizing the bi-directional Chamfer distance (Min-Min Cosine Distance) over this joint space, the continuous neural network **spontaneously quantizes** its latent space. It learns to map specific continuous embeddings to highly specific macro-structures (like distinct MNIST digits) without explicit classification losses or batch-pairing masks.

## Features
- **1-Step Generation:** No multi-step diffusion.
- **No Discriminator:** No unstable adversarial training.
- **No Explicit Optimal Transport:** No CPU-bottlenecked sorting or unstable Sinkhorn exponentials.
- **Self-Organizing:** Automatically clusters the continuous latent space into semantic features.

## Usage
Train the unified SCGN model on various datasets:

```bash
# Train on MNIST (perfectly separates all 10 digits)
python train_scgn.py --dataset mnist

# Train on Fashion-MNIST
python train_scgn.py --dataset fmnist

# Train on CIFAR-10 (tests complex variance tracking)
python train_scgn.py --dataset cifar
```

## How it works (The Code)
The entire logic boils down to a few lines of PyTorch matrix multiplications:

```python
# Project generated and real to Joint Sphere
sim_img = torch.mm(gen_img_norm, real_img_norm.t())
sim_cond = torch.mm(gen_cond, real_cond.t())        

# Cost Matrix
C = (1.0 - sim_img) + gamma * (1.0 - sim_cond)

# Bidirectional Chamfer
loss_gen_to_real = C.min(dim=1)[0].mean()
loss_real_to_gen = C.min(dim=0)[0].mean()
loss = loss_gen_to_real + loss_real_to_gen
```

## Discovery
This architecture was discovered and mathematically formulated during a deep-dive research session into the limits of Optimal Transport and unsupervised learning.
