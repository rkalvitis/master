# Benchmark Summary Across Papers

## Papers Covered

| # | Paper | Venue |
|---|-------|-------|
| 1 | **NeRF** — Representing Scenes as Neural Radiance Fields for View Synthesis (Mildenhall et al.) | arXiv 2020 |
| 2 | **3DGS** — 3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al.) | ACM TOG 2023 |
| 3 | **OMG** — Opacity Matters in Material Modeling with Gaussian Splatting (Yong et al.) | ICLR 2025 |
| 4 | **DropGaussian** — Structural Regularization for Sparse-view Gaussian Splatting (Park et al.) | CVPR 2025 |
| 5 | **SMW-GS** — Micro-Macro Gaussian Splatting with Enhanced Scalability (Li et al.) | arXiv 2025 |

---

## Metrics Used

All five papers use the same three core image quality metrics:

| Metric | Direction | What it measures |
|--------|-----------|-----------------|
| **PSNR** (Peak Signal-to-Noise Ratio) | ↑ higher is better | Pixel-level accuracy |
| **SSIM** (Structural Similarity Index) | ↑ higher is better | Perceptual structure, contrast, luminance |
| **LPIPS** (Learned Perceptual Image Patch Similarity) | ↓ lower is better | Deep-feature perceptual distance |

Additional metrics used by specific papers:
- **FPS** (Frames Per Second) — 3DGS, SMW-GS (render speed)
- **Memory / Storage** — 3DGS, SMW-GS
- **MSE** — OMG only (albedo/roughness estimation error)
- **Relighting PSNR/SSIM/LPIPS** — OMG only

---

## Datasets Used Per Paper

### 1. NeRF
| Dataset | Description | Scenes / Images |
|---------|-------------|-----------------|
| **Diffuse Synthetic 360°** (DeepVoxels) | 4 Lambertian objects, simple geometry | 512×512 px; 479 train / 1000 test views |
| **Realistic Synthetic 360°** | 8 path-traced objects, non-Lambertian materials | 800×800 px; 100 train / 200 test views |
| **Real Forward-Facing** (LLFF) | 8 real-world handheld captures | 1008×756 px; 20–62 images per scene |

Baselines: SRN, Neural Volumes (NV), LLFF

---

### 2. 3DGS
| Dataset | Description | Scenes used |
|---------|-------------|-------------|
| **Mip-NeRF360** | Unbounded indoor + outdoor 360° scenes | Bicycle, Garden, Stump, Counter, Room |
| **Tanks & Temples** | Real-world large object captures | Truck, Train |
| **Deep Blending** | Indoor forward-facing scenes | Playroom, DrJohnson |
| **Synthetic Blender** | 13 synthetic scenes (same as NeRF's Realistic Synthetic) | Used in ablation (Table 2) |

Baselines: Mip-NeRF360, InstantNGP, Plenoxels

---

### 3. OMG
| Dataset | Description | Task |
|---------|-------------|------|
| **Synthetic4Relight** | Synthetic objects with known material ground truth | Material modeling + relighting |
| **Shiny Blender** | Reflective/specular synthetic objects | Novel view synthesis |
| **Glossy Synthetic** | Glossy surface synthetic objects | Novel view synthesis |
| **Mip-NeRF 360** | Real-world unbounded scenes (9 scenes) | Novel view synthesis |

Baselines: R3DG, GaussianShader, GS-IR

---

### 4. DropGaussian
| Dataset | Description | Sparse-view setting |
|---------|-------------|---------------------|
| **LLFF** | Forward-facing real scenes | 3-view, 6-view, 9-view |
| **Mip-NeRF360** | Unbounded 360° scenes | 12-view, 24-view |
| **Blender** | Synthetic bounded scenes | 8 training views |
| **Replica** | Indoor synthetic scenes | 2-view (feed-forward comparison) |

Baselines (NeRF): Mip-NeRF, DietNeRF, RegNeRF, FreeNeRF, SparseNeRF  
Baselines (3DGS): 3DGS, DNGaussian, FSGS, CoR-GS  
Baselines (feed-forward): pixelSplat, MVSplat, FreeSplat

---

### 5. SMW-GS (Micro-Macro)
| Dataset | Description | Scenes |
|---------|-------------|--------|
| **Phototourism** (classical unconstrained) | In-the-wild internet photo collections | Brandenburg Gate, Sacre Coeur, Trevi Fountain |
| **UrbanScene3D** (real large-scale) | Aerial drone captures of urban environments | Rubble, Building, Residence, Sci-Art |
| **MatrixCity** (synthetic large-scale) | Synthetic city-scale scenes with appearance variations | Block_A, Block_E, Block_A*, Block_E* |

Baselines: Ha-NeRF, CR-NeRF, WildGaussians, GS-W, VastGaussian, CityGaussian, Momentum-GS

---

## Benchmark Overlap — Which Datasets Are Shared?

| Dataset | NeRF | 3DGS | OMG | DropGaussian | SMW-GS |
|---------|:----:|:----:|:---:|:------------:|:------:|
| **Mip-NeRF360** | — | ✓ | ✓ | ✓ | — |
| **Blender (Realistic Synthetic)** | ✓ | ✓ (ablation) | — | ✓ | — |
| **LLFF / Real Forward-Facing** | ✓ | — | — | ✓ | — |
| **Tanks & Temples** | — | ✓ | — | — | — |
| **Deep Blending** | — | ✓ | — | — | — |
| **Synthetic4Relight** | — | — | ✓ | — | — |
| **Shiny Blender** | — | — | ✓ | — | — |
| **Glossy Synthetic** | — | — | ✓ | — | — |
| **Replica** | — | — | — | ✓ | — |
| **Phototourism** | — | — | — | — | ✓ |
| **UrbanScene3D** | — | — | — | — | ✓ |
| **MatrixCity** | — | — | — | — | ✓ |

**Key observations:**
- **Mip-NeRF360** is the most shared benchmark — used by 3DGS, OMG, and DropGaussian. It has become the de facto standard for evaluating novel view synthesis on real unbounded scenes.
- **Blender (Realistic Synthetic)** is shared by NeRF, 3DGS, and DropGaussian — a common synthetic benchmark inherited from the original NeRF paper.
- **LLFF** links NeRF and DropGaussian — the original real-world benchmark from 2019, still used for sparse-view evaluation.
- **OMG** uses unique inverse-rendering-specific datasets (Synthetic4Relight, Shiny Blender, Glossy Synthetic) that no other paper uses — this reflects its different task (material decomposition, not just novel view synthesis).
- **SMW-GS** uses entirely different datasets from all others — Phototourism, UrbanScene3D, and MatrixCity — because its focus is large-scale unconstrained reconstruction rather than object-centric or bounded scenes.

---

## Are the Benchmarks the Same?

**Short answer: Partially, but not fully.**

- Papers sharing the same task (novel view synthesis) tend to converge on **Mip-NeRF360** and **Blender** as common ground.
- Papers with different focuses (inverse rendering, large-scale, sparse-view) use **task-specific datasets** that the others do not.
- Even when the same dataset is used (e.g., Mip-NeRF360 in 3DGS, OMG, and DropGaussian), the **split, number of training views, and what is being measured** can differ — for example, DropGaussian uses 12/24-view sparse splits of Mip-NeRF360, while 3DGS uses the standard full split.
- The metrics (PSNR, SSIM, LPIPS) are **consistent across all papers**, making cross-paper number comparisons possible where datasets overlap.
