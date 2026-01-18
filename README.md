# ResMLP-GL: Signature-Aware Deep Learning for Cancer Driver Prediction

This repository contains the full implementation, trained models, and processed datasets for the study:

**"Signature-aware deep learning reveals distinct driver gene programs and mutational processes in glioblastoma and colon adenocarcinoma"**

## Repository Structure

ResMLP-GL/
├── data/ # Processed mutation feature tables (TCGA / ICGC)
├── Python-Code/ # Training and evaluation scripts
├── models/ # Pretrained ResMLP-GL model weights
├── Results/ # Figures and tables used in the manuscript
├── supplementary-files/ # Supplementary tables and figures
├── preprocessor.pkl # Feature preprocessing transformer
├── driver_prediction_model.keras
├── run_revision_pipeline.py
├── requirements.txt
└── README.md

## Requirements

- Python ≥ 3.9
- TensorFlow 2.15
- scikit-learn 1.4.1
- imbalanced-learn 0.12
- Optuna 3.6.1
- SHAP
- lifelines

Install dependencies:

```bash
pip install -r requirements.txt
Reproducibility

All experiments were run with fixed random seeds at Python, NumPy, and TensorFlow levels.
Processed feature tables and the preprocessing transformer are provided to ensure exact reproducibility.

Hardware Used

NVIDIA GeForce RTX 4090 (24GB VRAM)

Intel Core i9-13900K

125 GB RAM

Ubuntu 20.04 LTS
Citation

If you use this work, please cite the corresponding paper (under review at Computational Biology and Chemistry).
