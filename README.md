# ResMLP-GL
Deep Residual Multi-Layer Perceptron (ResMLP-GL) Identifies Novel Therapeutic Targets 
# Cross-Cancer Driver-Mutation Pipeline  
Deep-learning workflow for **identifying, interpreting, and benchmarking cancer driver mutations** across glioblastoma (GBM) and lung adenocarcinoma (LUAD).

---

## ✨ Key Features
* **End–to–end automation** – from raw TCGA‐style variant files to publication-quality figures.  
* **Deep Residual MLP** with Optuna hyper-parameter search and ADASYN class balancing.  
* **Explainability** – SHAP‐based feature attribution grouped by SBS signatures, pathogenicity scores, etc.  
* **Benchmarking** – head-to-head comparison with CHASMplus, OncodriveFML, and MutSigCV.  
* **Cross-cancer insights** – Venn diagrams and differential‐frequency plots (GBM vs LUAD).

---

## 🗂️ Repository Layout
| Stage | Script(s) | Purpose |
|-------|-----------|---------|
| **1. Pre-processing** | `1_preprocess_data.py` | Minimal GBM + LUAD merge, driver flag, COSMIC-ready mutation string. |
|  | `1A` – `1F` helper scripts | Optional extras: DP cleanup, VCF fixes, OpenCRAVAT TSV conversion, sanity checks. |
| **2. Feature engineering** | `2_preprocess_data.py` | Creates `mutations_variant_complete.tsv` with SBS, AlphaMissense, CADD, … |
| **3. Model training** | `3_model_training.py` | Residual MLP + Optuna search; saves `driver_prediction_model.keras`. |
| **4. Feature selection** | `4_shap_feature_selector.py` | SHAP value computation & grouped importance plots. |
| **5. 3rd-party tool inputs** | `5A`–`5C` | Generate CHASMplus / OncodriveFML / MutSigCV input files. |
| **6. Benchmark & viz** | `6_benchmark_and_visualization.py` | AUROC/PR, top-gene Venn, bar charts. |
| **7. Publication figures** | `7_model_evaluation_visuals.py` | High-res ROC/PR curves, SHAP heatmaps, etc. |
| **8. Cross-cancer analysis** | `8_Cross-Cancer-Comparison-(GBM-vs-LUAD).py` + `8A_updated-*` | Differential mutation frequency & enrichment stats. |

---

## 📦 Requirements
* **Python 3.10+**
* Pip packages (see quick install below)  
  `pandas numpy scikit-learn imbalanced-learn tensorflow optuna shap matplotlib seaborn matplotlib-venn joblib`
* **R (≥ 4.1)** only if you plan to run MutSigCV or additional CNV work.
* (Optional) **CHASMplus, OncodriveFML, MutSigCV** executables in `$PATH`.

```bash
# one-liner install
pip install pandas numpy scikit-learn imbalanced-learn tensorflow optuna shap \
            matplotlib seaborn matplotlib-venn joblib
📑 Input Files
File	Description
TCGA-GBM.somaticmutation_wxs.tsv	TCGA-GBM MAF-like table with effect, dna_vaf, etc.
TCGA-LUAD.somaticmutation_wxs.tsv	TCGA-LUAD equivalent.
Census_all*.csv	COSMIC Cancer Gene Census (for driver flag).
mutations_variant_complete.tsv	Auto-generated master feature table used by the model.

Place them in the repository root before running the pipeline.

🚀 Quick-Start
bash
Copy
Edit
# 1) Minimal preprocessing & feature build
python 1_preprocess_data.py        # creates combined & annotated TSV
python 2_preprocess_data.py        # adds SBS, AlphaMissense, etc.

# 2) Train the deep-learning classifier
python 3_model_training.py         # saves model + Optuna study

# 3) Interpret the model
python 4_shap_feature_selector.py  # writes SHAP plots to ./shap_results/

# 4) Generate benchmark input for external tools (optional)
python 5A_chasm_input_data.py
python 5B_oncodrive_input_data.py
python 5C_MutsigCV_input.py

# 5) Compare and visualise
python 6_benchmark_and_visualization.py
python 7_model_evaluation_visuals.py

# 6) Cross-cancer differential analysis
python 8_Cross-Cancer-Comparison-(GBM-vs-LUAD).py
Each script prints its output locations (tables, .keras model, PNG/SVG figures).

📊 Expected Outputs
driver_prediction_model.keras – trained TensorFlow model (~90 % AUROC on dev data).

shap_results/ – feature importance bar charts & heatmaps.

figures/ – ROC, PR curves, Venn diagrams, frequency dotplots.

*_input.tsv – ready-to-run files for CHASMplus / OncodriveFML / MutSigCV.

📝 Citation
If you use this code or figures in a publication, please cite:

sql
Copy
Edit
Zubair M. et al. “Cross-Cancer Analysis of Somatic Mutations Using Deep Learning Identifies Novel Therapeutic Targets in Glioblastoma and Lung Adenocarcinoma.” (2025)
🛠️ Troubleshooting
Memory errors during Optuna search – reduce n_trials in 3_model_training.py.

Missing columns – ensure your TSVs follow TCGA MAF column naming (effect, dna_vaf, …).

External tool paths – edit the top of 5A–5C scripts to point to your installations.

📄 License
Distributed under the MIT License (see LICENSE for details).

🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

✉️ Contact
Muhammad Zubair 
