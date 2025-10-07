# Simulation-Based Deep Learning for Chemical Reaction Mechanism Classification

A multimodal neural-network framework that learns catalytic reaction mechanisms from simulated concentration–time data by combining static descriptors and dynamic trajectories.

## Abstract

This repository implements a workflow for simulation‑based mechanism classification in complex catalytic systems. It comprises (i) a simulator that generates labelled kinetic profiles for 20 reaction mechanisms (M1–M20); (ii) a two‑branch neural architecture that fuses static initial‑condition descriptors with dynamic time‑series; and (iii) a comparative study of four feature‑fusion strategies—late averaging, attention‑based reweighting, gate‑weighted additive fusion, and Hadamard (bilinear) fusion.

## Project Goal

The project aims to overcome the limitations of classical kinetic analysis—such as initial-rate methods, linearising transformations, and integrated rate laws—which struggle to describe systems with multiple catalyst resting states. The neural network is trained on diverse simulation-generated concentration–time profiles, allowing it to learn complex temporal behaviours and accurately classify catalytic mechanisms beyond the scope of traditional kinetic models.

## Highlights

- **Simulation-based dataset generation:** millions of labelled samples are produced by numerically integrating twenty ODE-defined catalytic mechanisms under diverse kinetic parameters and initial conditions. 
- **Multimodal neural architecture:** A two-branch neural network combines a dense encoder for static catalyst descriptors with stacked LSTMs for dynamic trajectories, enabling the model to learn temporal features such as induction, saturation, and deactivation from full concentration–time data.
- **Feature-fusion strategies:** The neural architecture explores multiple deep learning fusion schemes, including attention mechanisms, gate-weighted additive fusion, and Hadamard (bilinear) coupling, to optimise feature integration within the network and strengthen cross-modal representation learning.
- **Comprehensive model evaluation:** The models are evaluated through top-1 and top-3 accuracy, cross-entropy loss, predictive entropy, and 95% credible-set distributions, complemented by confusion-matrix analysis to differentiate model errors from intrinsic mechanistic non-identifiability.

## Project Structure

```
Simulation_Based_Classification/
│
├── ODE_solver_exploration.ipynb                  # Exploration of LSODA solver behaviour
│
├── simulation_data_generation                    # Simulation scripts and kinetic dataset generation
│   ├── simulated_kinetic_profiles_example.ipynb
│   └── simulation_data_generation.ipynb
│
├── model_architecture_overview                   # Visual overview of the model architecture
│   ├── model_average_fusion_arch
│   ├── model_attention_fusion_arch
│   ├── model_gating_fusion_arch
│   └── model_hadamard_fusion_arch
│
├── train_model                                   # Trained models with associated training notebooks and saved weights
│   ├── trained_model_baseline
│   │   ├── M1_20_model_classification_baseline
│   │   ├── mechanism_classifier_baseline_model.ipynb
│   │   ├── training_history_baseline.pkl
│   │   └── best_model_weights_baseline.index
│   │
│   ├── trained_model_attention
│   │   ├── M1_20_model_classification_attention
│   │   ├── mechanism_classifier_attention_model.ipynb
│   │   ├── training_history_attention.pkl
│   │   └── best_model_weights_attention.index
│   │
│   ├── trained_model_gatefusion
│   │   ├── M1_20_model_classification_gatefusion
│   │   ├── mechanism_classifier_gatefusion_model.ipynb
│   │   ├── training_history_gatefusion.pkl
│   │   └── best_model_weights_gatefusion.index
│   │
│   └── trained_model_hadamard
│       ├── M1_20_model_classification_hadamard
│       ├── mechanism_classifier_hadamard_model.ipynb
│       ├── training_history_hadamard.pkl
│       └── best_model_weights_hadamard.index
│
├── model_evaluation                              # Model performance evaluation and uncertainty analysis notebooks
│   ├── euclidean_distance_analysis.ipynb
│   ├── model_performance_evaluation.ipynb
│   ├── simulation_distribution_evaluation.ipynb
│   └── training_history_plots.ipynb
│
└── catalytic_mechanisms_types_M1_to_M20.pdf      # Mechanism overview diagram
```

## Dataset Availability
The full training, validation, and test datasets are hosted externally due to size constraints. All files can be accessed and downloaded from OneDrive via the following link:
[🔗 Download Data (OneDrive)](https://1drv.ms/f/s!AtSPOuyiZcMKgQJpXgPnEHD2dFKX?e=dfRsQG)

The repository contains:
- Training set: 4.95M labelled samples
- Validation set: 50k samples
- Test set: 100k samples (sparse subsampling, distinct parameter sets)
- **Per sample input**:  
  – Static vector $x_1 \in \mathbb{R}^4$ (four loadings)  
  – Dynamic block $X_2 \in \mathbb{R}^{N \times 12}$ (time, $S$, $P$ for four trajectories)  

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
