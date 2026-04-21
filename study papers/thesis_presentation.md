# Thesis Meeting — Study Guide, Presentation Outline & Paper Summaries
## Topic: Neural Radiance Fields → 3D Gaussian Splatting and Extensions

---

## PART 1 — STUDY GUIDE

### Core Concepts to Understand

**Novel View Synthesis (NVS)**
The task of generating photorealistic images from viewpoints not present in the training set, given a set of input images with known camera poses.

**Volume Rendering**
A technique to accumulate color along a camera ray by integrating color × density contributions. The key equation:
```
C(r) = ∫ T(t) · σ(r(t)) · c(r(t), d) dt
where T(t) = exp(−∫ σ(r(s)) ds)   [transmittance]
```
T(t) is the probability the ray travels from t_n to t without hitting anything. This integral is approximated discretely as alpha-compositing:
```
Ĉ = Σ T_i · (1 − exp(−σ_i δ_i)) · c_i
```

**Alpha Compositing (α-blending)**
The discrete form of volume rendering. Each point has α_i = 1 − exp(−σ_i δ_i). Color is accumulated front-to-back weighted by transmittance.

**Structure-from-Motion (SfM)**
An algorithm (e.g., COLMAP) that estimates camera poses and produces a sparse 3D point cloud from a set of unordered photos. Both NeRF and 3DGS use SfM as input preprocessing.

**Positional Encoding**
Maps low-dimensional inputs to higher-dimensional space using sinusoidal functions so that MLPs can learn high-frequency functions:
```
γ(p) = (sin(2⁰πp), cos(2⁰πp), ..., sin(2^(L−1)πp), cos(2^(L−1)πp))
```
Without it, networks are biased toward smooth/blurry outputs.

**Spherical Harmonics (SH)**
A set of basis functions on the sphere used to encode view-dependent color effects (specularity, reflections). Each Gaussian in 3DGS stores SH coefficients instead of a single RGB value, allowing color to change with viewing direction.

**Inverse Rendering**
The problem of decomposing a scene from images into its intrinsic properties: geometry, materials (albedo, roughness, metalness), and lighting. Downstream applications: relighting, material editing, sim-to-real transfer.

**Bouguer-Beer-Lambert Law**
A physics law from radiative transfer: light intensity decreases exponentially when passing through an absorbing material. The extinction (attenuation) depends on particle density n and cross-section σ_ν (which is material-dependent):
```
I(s) = I(0) · e^(−nσ_ν s)   →   α = 1 − e^(−nσ_ν s)
```

**Sparse-View NVS**
Novel view synthesis with very few (e.g., 3) training images. 3DGS overfits to training views because Gaussians far from the camera receive weak gradient signals.

### Key Metrics
| Metric | Better | What it measures |
|--------|--------|-----------------|
| PSNR (dB) | Higher | Per-pixel average error (log scale) |
| SSIM | Higher (max 1) | Structural similarity (luminance, contrast, texture) |
| LPIPS | Lower | Perceptual similarity using deep features |

### The Progression of Methods
```
NeRF (2020)
  ↓  Slow (1–2 days train, 1s/frame), implicit MLP
3DGS (2023)
  ↓  Real-time (≥30fps), explicit Gaussians, fast train
    ├── OMG (2025): adds physics-based opacity for inverse rendering
    ├── DropGaussian (2025): fixes overfitting with sparse views
    └── SMW-GS (2025): handles unconstrained/large-scale scenes
```

---

## PART 2 — PRESENTATION OUTLINE

**Title:** From Neural Fields to Gaussian Splatting: A Review of Novel View Synthesis Methods

**Suggested duration:** 20–25 minutes + questions

---

### Slide 1 — Motivation
- What is novel view synthesis? Why does it matter?
- Applications: VR/AR, robotics, digital twins, macro photography, cultural heritage
- Show: Dany Bittel's macro Gaussian splatting demo (https://danybittel.ch/macro.html) as a striking visual opener

### Slide 2 — The Fundamental Problem
- Input: a set of photos with known camera poses (from SfM/COLMAP)
- Goal: render the scene from any new viewpoint
- Two families: **implicit** (neural networks encode scene) vs **explicit** (store scene geometry directly)
- NeRF = implicit. 3DGS = explicit.

### Slide 3 — NeRF: The Implicit Baseline
- Scene = continuous 5D function: (x, y, z, θ, φ) → (RGB, density σ)
- MLP: 8 layers, 256 channels, ReLU
- Render by marching rays through volume, numerically integrate
- Key tricks: **positional encoding** (high-freq details), **hierarchical sampling** (coarse+fine networks)
- Limitation: 1–2 days to train, ~1s per frame to render

### Slide 4 — 3DGS: The Explicit Revolution
- Replace MLP with ~1–5 million 3D Gaussian blobs
- Each Gaussian: position μ, covariance Σ (→ shape/size/orientation), opacity α, SH color coefficients
- Initialized from SfM sparse point cloud
- Render: project Gaussians to 2D, sort by depth, tile-based GPU rasterizer → **real-time ≥30fps at 1080p**
- Train: ~7min (7K iter) to ~35min (30K iter)

### Slide 5 — 3DGS: Adaptive Density Control (key mechanism)
- Problem: SfM gives sparse, imperfect initialization
- Solution: interleaved clone/split strategy during optimization
  - **Clone**: small Gaussian in under-reconstructed area → duplicate + move
  - **Split**: large Gaussian covering too much area → split into 2
  - **Prune**: Gaussians with opacity < threshold → remove
- This self-organizing process produces compact, accurate representations

### Slide 6 — 3DGS vs NeRF: Numbers
| | NeRF | 3DGS (30K) |
|---|---|---|
| Train time | 1–2 days | ~35 min |
| Render speed | ~0.07 fps | 137 fps |
| PSNR (MipNeRF360) | 27.69 | 27.21 |
| Memory | low | ~734 MB |

Quality comparable, speed dramatically better.

### Slide 7 — Extension 1: OMG — Physics-Based Opacity (ICLR 2025)
- Problem: 3DGS-based inverse rendering ignores that **opacity depends on material**
- Glass vs gas: same density, completely different light absorption
- The fix: apply Bouguer-Beer-Lambert law → α_i = 1 − exp(−o_i · G_i(x) · f(m_i))
  - f(·) is a small neural network that maps material properties to cross-section σ_ν
- Material properties now receive gradients from **both color loss and opacity** → better constrained
- Plug-and-play improvement on top of existing baselines: +0.4 dB PSNR, better albedo/roughness

### Slide 8 — Extension 2: DropGaussian — Sparse-View Robustness (CVPR 2025)
- Problem: 3DGS overfits to sparse inputs (3–9 views) because far Gaussians get weak gradients
- Analogy to **dropout**: randomly remove a fraction r of Gaussians during each training step
- Remaining Gaussians have their opacity scaled up by 1/(1−r) to compensate
- Far-from-camera Gaussians now get better gradient coverage → whole scene reconstructed
- Drop rate increases progressively (r_t = γ · t/t_total), stronger regularization as training progresses
- Result: best PSNR on LLFF 3-view (20.76), prior-free, zero added inference complexity

### Slide 9 — Extension 3: SMW-GS — Large-Scale Unconstrained Scenes (arXiv 2025)
- Problem: internet photo collections have varying illumination, large unbounded scenes (cities)
- Three innovations:
  1. **Appearance decoupling**: Global (f_g, scene-wide tone) + Refined (f_r, local textures) + Intrinsic (f_v, material) features
  2. **Micro-macro Wavelet Sampling**: samples 2D feature maps at both fine (narrow frustum) and broad (wide frustum) scales using DWT, capturing high-frequency texture AND regional lighting
  3. **PSG Camera Partitioning**: for large scenes, intelligently assigns cameras to spatial blocks by maximizing per-Gaussian supervision → scalable training
- Results: SOTA on Brandenburg Gate, Sacre Coeur, Trevi Fountain; 1.5× faster than GS-W

### Slide 10 — How These Papers Relate to Macro Gaussian Splatting
- Connect the papers to the thesis context
- Macro photography = extreme close-up scenes → lighting variations, material details critical
- OMG: accurate material modeling at macro scale
- DropGaussian: few training views (macro setups often have limited angles)
- SMW-GS: handling variable illumination and detail at multiple scales
- Open question: how do these methods handle the depth-of-field blur typical of macro lenses?

### Slide 11 — Summary & Research Directions
| Challenge | Paper | Approach |
|-----------|-------|----------|
| Speed | 3DGS | Explicit Gaussians + GPU rasterizer |
| Material accuracy | OMG | Beer-Lambert opacity |
| Sparse views | DropGaussian | Stochastic Gaussian removal |
| Illumination variation | SMW-GS | Wavelet multi-scale features |
| Macro scenes | ? | Open research question |

---

## PART 3 — PAPER SUMMARIES

---

### Paper 1: NeRF — Representing Scenes as Neural Radiance Fields for View Synthesis
**Authors:** Mildenhall, Srinivasan, Tancik, Barron, Ramamoorthi, Ng (UC Berkeley, 2020)
**arXiv:** 2003.08934 | **GitHub:** https://github.com/bmild/nerf

**Problem:** Synthesize novel views of complex scenes from a sparse set of input images.

**Core idea:** Represent a static scene as a continuous 5D function F_Θ: (x,y,z,θ,φ) → (RGB, σ). A fully-connected MLP (8 layers × 256 channels, ReLU) encodes this function in its weights. To render, cast rays from the camera and numerically integrate the volume rendering equation along each ray.

**Key technical contributions:**

1. **Positional Encoding:** Raw (x,y,z,θ,φ) inputs fail because MLPs are biased toward low-frequency functions. Map inputs to higher-dimensional space: γ(p) = [sin(2⁰πp), cos(2⁰πp), ..., sin(2^(L−1)πp), cos(2^(L−1)πp)]. L=10 for position, L=4 for direction. This allows representing high-frequency geometry and texture.

2. **Hierarchical Sampling:** Densely sampling along every ray is wasteful (empty space). Train two networks: "coarse" sampled uniformly, "fine" sampled proportionally to coarse density. Allocates compute to where scene content actually exists.

3. **View-dependent color:** Density σ depends only on position (multiview consistent geometry), RGB color c depends on both position and viewing direction (captures specularity/reflections).

**Results:** PSNR 40.15 on synthetic scenes, outperforms all prior neural rendering methods. Renders at ~0.07 fps; training takes 100K–300K iterations (~1–2 days on a V100 GPU).

**Limitations:** Extremely slow training and inference. Requires known camera poses. Represents one scene per network — no generalization.

---

### Paper 2: 3D Gaussian Splatting for Real-Time Radiance Field Rendering
**Authors:** Kerbl, Kopanas, Leimkühler, Drettakis (Inria, 2023)
**arXiv:** 2308.04079 | **GitHub:** https://github.com/graphdeco-inria/gaussian-splatting

**Problem:** NeRF-quality novel view synthesis at real-time rendering speeds.

**Core idea:** Represent the scene explicitly as a collection of 1–5 million anisotropic 3D Gaussians. Each Gaussian is defined by:
- **Position (mean) μ ∈ ℝ³**
- **Covariance Σ** (factored as Σ = RSS^T R^T using a scale vector s and rotation quaternion q)
- **Opacity α ∈ [0,1]**
- **Spherical Harmonic coefficients** for view-dependent color (4 bands = 48 coefficients)

Initialized from the SfM sparse point cloud. All parameters are optimized jointly via gradient descent.

**Rendering pipeline:**
1. Project 3D Gaussians to 2D splats using camera transform
2. Sort by depth using a fast GPU Radix sort
3. Tile-based rasterizer: split screen into 16×16 tiles, each tile processes its Gaussians front-to-back with α-blending
4. Backward pass traverses tile lists back-to-front, recovering accumulated α for gradient computation

**Adaptive Density Control (key innovation):**
- Every 100 iterations: examine Gaussians with large view-space positional gradients
- **Clone** small Gaussians (under-reconstruction) → copy + move in gradient direction
- **Split** large Gaussians (over-reconstruction) → replace with 2 smaller ones, scale ÷ 1.6
- **Prune** Gaussians with α < ε_α every 3000 iterations

**Results:** 135 fps at 1080p (7K iter), 93 fps (30K iter). PSNR 27.21 on MipNeRF360, comparable to Mip-NeRF360 which takes 48 hours to train. Training: ~35 min for 30K iterations on a single A6000 GPU. Memory: ~700 MB–2 GB.

**Ablation highlights:** Removing anisotropic covariance (→ isotropic) significantly hurts quality. Removing SH hurts view-dependent effects. Removing clone/split severely degrades background quality.

**Limitations:** Popping artifacts with view-dependent Gaussians. Struggles in poorly observed regions. No regularization applied.

---

### Paper 3: OMG — Opacity Matters in Material Modeling with Gaussian Splatting
**Authors:** Yong, Manivannan, Kerbl, Wan, Stepputtis, Sycara, Xie (Carnegie Mellon, ICLR 2025)
**GitHub:** https://github.com/SilongYong/OMG

**Problem:** 3DGS-based inverse rendering (decomposing images into geometry + materials + lighting) treats opacity as a standalone parameter independent of material properties. This violates physics.

**Insight:** The Bouguer-Beer-Lambert law states that light attenuation depends on the material's cross-section σ_ν (how likely a particle is to interact with a photon) and particle density n. In 3DGS, the Gaussian blob plays the role of an absorbing body. Therefore opacity should be:
```
α_i = 1 − exp(−n_i · σ_ν · s)  →  α_i = 1 − exp(−o_i · G_i(x) · f(m_i))
```
where f(·) is a small MLP (2 hidden layers, 128 units, ReLU, sigmoid output) that maps material properties m_i (albedo, roughness) to cross-section σ_ν.

**Why this matters:**
- Material properties now receive gradients from **two sources**: color (via PBR loss) AND opacity (via the α term)
- This acts as a physics-informed regularization — better constrained optimization
- Taylor expansion analysis shows the standard α formula is a first-order approximation of the OMG formula (confirms mathematical correctness)

**Key results (plug-and-play on 3 baselines):**
- R3DG: +0.4 dB PSNR NVS, +0.6 dB albedo PSNR, albedo MSE reduced from 0.011 → 0.007
- GaussianShader: +0.3 dB PSNR on Shiny Blender
- GS-IR: +0.5 dB PSNR on MipNeRF360 real-world data, sharper normal estimation

**Limitation:** The improvement in geometry is a side-effect (opacity now jointly represents geometry and material), which is beneficial in most cases but may complicate disentanglement.

---

### Paper 4: DropGaussian — Structural Regularization for Sparse-view Gaussian Splatting
**Authors:** Park, Ryu, Kim (Konkuk University, CVPR 2025)
**GitHub:** https://github.com/DCVL-3D/DropGaussian_release

**Problem:** 3DGS overfits to training views when only sparse inputs (3–9 views) are available. Gaussians far from the camera have low transmittance (occluded by other Gaussians), receive weak gradient signals, and fail to represent unobserved regions. This leads to good reconstruction of training views but poor generalization to novel views.

**Core idea:** During each training step, randomly drop a fraction r of Gaussians. For the remaining Gaussians, compensate their opacity:
```
õ_i = M(i) · o_i    where M(i) = 1/(1−r) for remaining, 0 for dropped
```

This has two effects:
1. Far/occluded Gaussians are no longer always blocked → they receive larger gradient signals → the model learns the full scene
2. The representation is forced to be distributed across many Gaussians rather than concentrated in a small training-view-visible subset

**Progressive dropping:** Overfitting intensifies in later training. The drop rate increases linearly:
```
r_t = γ · (t / t_total)
```
γ = 0.2 by default. This avoids disrupting early-stage geometry formation.

**Dropping strategy:** Random removal is better than selective (by gradient or distance). Selective schemes risk permanently removing Gaussians critical for unseen regions.

**Results on LLFF dataset (3-view):**
- 3DGS baseline: PSNR 19.22
- DropGaussian: **PSNR 20.76** — best among all compared methods including prior-based approaches (CoR-GS 20.43, FSGS 20.43)
- MipNeRF360 (12-view): PSNR 19.74 vs 3DGS 18.52
- Blender (3-view): PSNR 25.42

**Advantages:** No additional modules, no pretrained priors needed, zero inference overhead (all Gaussians used at test time), simple one-hyperparameter method.

**Limitation:** Sensitivity to γ hyperparameter; optimal value varies by dataset.

---

### Paper 5: Micro-macro Gaussian Splatting with Enhanced Scalability for Unconstrained Scene Reconstruction (SMW-GS)
**Authors:** Li, Lv, Yang, Huang (2025)
**arXiv:** 2506.13516 | **GitHub:** https://github.com/Kidleyh/SMW-GS

**Problem:** Reconstructing scenes from uncontrolled internet photo collections (varying illumination, atmospheric conditions) and scaling to large urban environments (cities). Existing 3DGS methods use global appearance embeddings that fail to capture per-point local lighting variations; large-scale methods rely on block-level supervision that misses Gaussian-level detail.

**Architecture overview (builds on Scaffold-GS):**

**1. Structured Appearance Decoupling:**
Each Gaussian's appearance is decomposed into three components:
- **Global feature f_g** ∈ ℝ^n_g: Scene-wide tone/lighting (from CNN-encoded reference image, pooled globally)
- **Refined feature f_r** ∈ ℝ^n_r: Local textures, highlights, shadows (sampled from 2D feature maps)
- **Intrinsic feature f_v** ∈ ℝ^n_v: Material invariants (optimized directly during training)

**2. Micro-macro Wavelet Sampling (MWS) — key innovation:**
Standard 2D projection causes different 3D points on the same ray to sample identical features.
- **Micro-Projection (MP):** Project each Gaussian into a narrow conical frustum with learnable offsets {nc_i} → captures fine-grained textures
- **Broad Projection:** Sample from a wide conical frustum (radius ∝ 1/distance) → captures regional lighting
- **Wavelet-based Sampling (WS):** Apply 1-level Discrete Wavelet Transform to the feature map (→ 4 sub-bands: LL, LH, HL, HH). Low-pass captures smooth regions, high-pass captures edges/textures. Multi-scale sampling enriches feature diversity.

The micro and broad features are combined via learnable weights, then all scales are concatenated: f_r = f^n_{r,0} ⊕ f^b_{r,0} ⊕ ... ⊕ f^n_{r,M} ⊕ f^b_{r,M}

**3. Hierarchical Residual Fusion Network (HRFN):**
4 MLPs fuse f_g, f_r, f_v with positional encoding γ(x_i) and view direction d̄_ic to predict per-Gaussian colors. Residual connections preserve feature fidelity and improve gradient flow.

**4. Large-Scale Scene Promotion (PSG Camera Partitioning):**
For city-scale scenes, partition into M×N spatial blocks. Use **Point-Statistics-Guided** camera assignment: count visible camera observations per Gaussian → assign under-supervised cameras to blocks where they maximally help → ensures all Gaussians receive sufficient training signal. Rotational Block Training cycles GPUs across blocks to maintain global consistency.

**Results:**
- Classical unconstrained (Brandenburg Gate): PSNR 29.37 vs GS-W 27.96
- Trevi Fountain: PSNR 24.07 vs GS-W 23.24
- Large-scale Rubble: PSNR 29.03 vs Momentum-GS 26.87
- Rendering speed: 64–92 fps vs GS-W 54–60 fps (1.5× faster)
- Training time: 2.6h vs GS-W 2.83h (slightly faster)

**Limitation:** Rendering speed drops to ~30 fps on some large-scale scenes due to CNN feature extraction overhead.

---

## PART 4 — QUICK REFERENCE

### GitHub Links
| Paper | GitHub |
|-------|--------|
| NeRF | https://github.com/bmild/nerf |
| 3D Gaussian Splatting | https://github.com/graphdeco-inria/gaussian-splatting |
| OMG | https://github.com/SilongYong/OMG |
| DropGaussian | https://github.com/DCVL-3D/DropGaussian_release |
| SMW-GS | https://github.com/Kidleyh/SMW-GS |
| Macro GS demo | https://danybittel.ch/macro.html |

### The Single Most Important Idea from Each Paper
- **NeRF:** A scene can be encoded entirely in the weights of an MLP that maps 5D coordinates to color+density.
- **3DGS:** Replacing an MLP with explicit 3D Gaussians + a GPU rasterizer achieves comparable quality at 1000× the rendering speed.
- **OMG:** Opacity in a Gaussian blob should physically depend on what material fills that region — modeling this correctly constrains material estimation.
- **DropGaussian:** Randomly removing Gaussians during training (like dropout) forces all Gaussians to learn from more viewpoints, preventing overfitting in sparse-view settings.
- **SMW-GS:** Multi-scale wavelet sampling of 2D feature maps lets each Gaussian capture both fine local textures and broad regional lighting, enabling robust reconstruction under varying illumination.
