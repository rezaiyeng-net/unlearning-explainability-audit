# Beyond Accuracy: The Explanation Stability Gap in Machine Unlearning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the complete source code and experimental framework for the paper:

> **"Beyond Accuracy: The Explanation Stability Gap in Machine Unlearning"**  
> Hossein Rezaee  
> *Expert Systems with Applications* (under review)

---

## 📖 Overview

This repository provides a fully reproducible implementation of the **Feature Importance Distance (FID)** metric and the evaluation framework described in the paper. It is intended for researchers and practitioners who have read the paper and wish to:

- Reproduce all experimental results, including all tables (1–10) and figures (1–4)
- Compute FID on their own unlearned models
- Extend the framework to new datasets, model architectures, or unlearning algorithms

The code is modular, well-documented, and designed to support further research on explanation stability in machine unlearning.

---

## 📁 Repository Structure
├── code/
│ ├── main_part1_linear_tree.py # Phase 1: LR & RF (exact SHAP explainers)
│ ├── main_part2_advanced_models.py # Phase 2: GB & MLP (KernelExplainer)
│ └── mia_sensitivity_analysis.py # MIA threshold sensitivity analysis
├── results/
│ ├── part1_results.csv # Raw outputs for Phase 1 (15 seeds)
│ └── part2_results.csv # Raw outputs for Phase 2 (15 seeds)
├── tables/ # Generated LaTeX tables
├── figures/ # Generated vector-based figures
├── requirements.txt # Python dependencies
├── LICENSE # MIT License
└── README.md

---

## ⚙️ Requirements

The code requires **Python 3.8** or higher. Install all dependencies using:

```bash
pip install -r requirements.txt

