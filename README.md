# Simulation‑Based Mechanism Classification
### Automatic identification of reaction mechanisms from concentration–time profiles using simulated data and multimodal deep learning.

This repository implements a workflow for simulation‑based mechanism classification in complex catalytic systems. It comprises (i) a simulator that generates labelled kinetic profiles for 20 reaction mechanisms (M1–M20); (ii) a two‑branch neural architecture that fuses static initial‑condition descriptors with dynamic time‑series; and (iii) a comparative study of four feature‑fusion strategies—late averaging, attention‑based reweighting, gate‑weighted additive fusion, and Hadamard (bilinear) fusion.

## Project Goal
The project is designed to address limitations of classical kinetic analysis (initial‑rate methods, linearising transforms, integrated rate laws) when networks feature multiple resting states, activation/deactivation, or time‑dependent speciation. By training on diverse synthetic corpora, the classifier learns discriminative temporal features (e.g., induction, saturation, deactivation plateaus) that reliably fingerprint mechanism classes.

## Highlights
- **End-to-end pipeline**: ODE formalism → parameter sampling and mechanism-specific filtering → LSODA integration → time-resolved tensor construction → model training and evaluation.
- **Data Generation**: millions of labelled samples, with controlled covariate shift and sparse test sampling to stress-test generalisation. 
- **Multimodal classifier**: a static branch (initial conditions, e.g., catalyst loading) and a dynamic branch (trajectories of S and P) fused by differebr strategies. 
- **Evaluation**: top-1/top-3 accuracy, mean cross-entropy, entropy-based uncertainty, 95% credible-set size distributions, and confusion analyses to separate model errors from fundamental identifiability limits. 

## Simulation and Inference Framework
### 1) Mechanistic Simulator
- **ODE formalism**: Each mechanism $M_i$ is encoded as mass-action ODEs  
  $\frac{dC}{dt} = F(C; \theta)$, with species $[S], [P], [\text{cat}], [\text{cat}S], \dots$  
- **Parameter sampling**: Rate constants sampled log-uniformly over $10^{-5}$–$10^5$ and rounded to 3 s.f.; additional mechanism-specific post-filters ensure informative kinetics (e.g., minimum deactivation, dimer accumulation).  
- **Numerical integration**: SciPy LSODA with \texttt{rtol=1e-6}, \texttt{atol=1e-9}, time horizon $t \in [0, 100]$, dense grid of 5,000 points; non-physical or unstable solutions rejected.  

### 2) Initial Conditions and Data Design
- Per parameter set: four trajectories  
  – three at standard $[S]_0 = 1.0$, $[P]_0 = 0$ with distinct catalyst loadings sampled from $[0.01, 0.10]$;  
  – one partially converted start with $S_0 \sim \mathcal{U}(0.4, 0.8)$, $P_0 = 1 - S_0$.  
- **Temporal subsampling**: 21 points for train/val; 7 points for test (sparse regime). Time columns min–max scaled to $[0,1]$; concentrations kept in physical units.  

### 3) Dataset Availability (after filtering and packing)
The full training, validation, and test datasets are hosted externally due to size constraints. All files can be accessed and downloaded from OneDrive via the following link:
[🔗 Download Data (OneDrive)](https://1drv.ms/f/s!AtSPOuyiZcMKgQJpXgPnEHD2dFKX?e=dfRsQG)

The repository contains:
- Training set: 4.95M labelled samples
- Validation set: 50k samples
- Test set: 100k samples (sparse subsampling, distinct parameter sets)
- **Per sample input**:  
  – Static vector $x_1 \in \mathbb{R}^4$ (four loadings)  
  – Dynamic block $X_2 \in \mathbb{R}^{N \times 12}$ (time, $S$, $P$ for four trajectories)  

### 4) Classifier Architecture
- **Static branch**: Dense encoder (ReLU, 64-dim) for $x_1$.  
- **Dynamic branch**: stacked LSTM–LSTM encoder (64-dim) for $X_2$.  
- **Fusion strategies (comparative)**:  
  1. *Baseline Late-Average* – independent branch posteriors averaged at output level.  
  2. *Attention-Based Fusion* – concatenate embeddings → channel-wise attention (softmax) → fusion head.  
  3. *Gate-Weighted Additive* – project each modality → sigmoid gates → weighted sum → fusion head.  
  4. *Hadamard (Bilinear)* – project to shared space → element-wise product → fusion head.  
- **Training**: cross-entropy, Adam (1e-4) optimiser, random time subsampling (always keep $t=0$), optional Gaussian noise on $S, P$ (not on time).
  
## Case Studies
This study builds on the 2023 *Nature* paper ["Organic reaction mechanism classification using machine learning"](https://www.nature.com/articles/s41586-022-05639-4) and explores alternative SBI methods as outlined in the 2024 arXiv review ["A Comprehensive Guide to Simulation-based Inference in Computational Biology"](https://arxiv.org/abs/2409.19675).

## References
- Burés & Larrosa (2023). *Organic reaction mechanism classification using ML.* *Nature*, 613: 689–695.
- Schwarz (1978). *Estimating the dimension of a model.* *The Annals of Statistics*, 6(2): 461–464. DOI: 10.1214/aos/1176344136.
- Gutenkunst *et al.* (2007). *Universally sloppy parameter sensitivities in systems biology models.* *PLoS Computational Biology*, 3(10): e189. DOI: 10.1371/journal.pcbi.0030189.
- Hindmarsh (1983). *ODEPACK: a systematized collection of ODE solvers.* *Scientific Computing*.
- Goodwin *et al.* (2017). *Cantera: An object-oriented software toolkit for chemical kinetics, thermodynamics, and transport processes.* *Zeitschrift für Physikalische Chemie*.
- Papamakarios *et al.* (2019). *Neural density estimation and likelihood-free inference.* *arXiv preprint* (arXiv:).
- Cranmer *et al.* (2020). *The frontier of simulation-based inference.* *PNAS*.
- Martínez-Carrión *et al.* (2019). *Variable-time normalisation analysis.* *Chemical Science*.
- Akaike (1974). *A new look at the statistical model identification.* *IEEE Transactions on Automatic Control*.
- SBI Toolkit Documentation: [https://sbi-dev.github.io](https://sbi-dev.github.io)

---

> Repository maintained by [DearKarl](https://github.com/DearKarl). Contributions and feedback welcome.
