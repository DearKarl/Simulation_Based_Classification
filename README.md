# Learning Chemical Reactions with Simulation Based Inference (SBI)

This project applies **Bayesian simulation-based inference (SBI)** to identify chemical reaction mechanisms from sparse kinetic time-series data. It builds on the 2023 *Nature* paper ["Organic reaction mechanism classification using machine learning"](https://www.nature.com/articles/s41586-022-05639-4) and explores alternative SBI methods as outlined in the 2024 arXiv review ["A Comprehensive Guide to Simulation-based Inference in Computational Biology"](https://arxiv.org/abs/2409.19675).

## Project Goal
To infer the underlying chemical reaction mechanism from time-series concentration data by using simulation-based inference pipelines, neural density estimators, and mechanistic simulators (ODE-based).

## Methods
- **Data Generation**:
- **Simulator & Prior Design**:
- **Neural Density Estimation**:
- **Inference Algorithms**:
- **Posterior Sampling**:
- **Model Diagnostics & Validation**:


## Case Studies
- **Case Study 1**: Reimplementation of the *Nature 2023* paper using an amortised LSTM-based classifier.
- **Case Study 2**: Comparison between neural and classical SBI techniques for biological systems using synthetic likelihoods.

## References
- Burés & Larrosa (2023). *Nature*, 613: 689–695.
- Wang *et al.* (2024). *arXiv preprint* 2409.19675.
- SBI Toolkit Documentation: [https://sbi-dev.github.io](https://sbi-dev.github.io)

---

> Repository maintained by [DearKarl](https://github.com/DearKarl). Contributions and feedback welcome.
