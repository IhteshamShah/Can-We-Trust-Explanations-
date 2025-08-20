
# Can We Trust Explanations! 
**Evaluation of Model-Agnostic Explanation Techniques on Highly**
**Imbalanced, Multiclass-Multioutput Classification Problem**

This Paper is published at HealthInfo2025 conference (https://www.scitepress.org/Papers/2025/131574/131574.pdf)

This repository contains the implementation of experiments evaluating the **trustworthiness of machine learning explanations** in the medical domain.
We focus on **LIME** and **SHAP** explainers, and evaluate them along three key dimensions:

* **Fidelity** – how well the explanation reflects the model’s prediction.
* **Stability** – how consistent the explanation is across repeated runs.
* **Medical Guidelines Concordance** – how well the explanations align with expert medical knowledge.

---

## 📂 Project Structure

* **`main.py`** – Runs the experiments, executes explanation methods, and saves results.
* **`utils.py`** – Provides data loading, preprocessing, integration of medical guidelines, Random Forest training, and functions to compute explanation results across fidelity, stability, and guideline comparison.
* **`explainer.py`** – Implements **LIME** and **SHAP** explainers, functions to extract top contributing features, and methods for computing variance/stability indices.
* **`logging_config.py`** – Centralized logging configuration for capturing experiment outputs, warnings, and errors.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/IhteshamShah/Can-We-Trust-Explanations-.git
cd project

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the experiments with:

```bash
python main.py
```

Results (logs and plots) are saved automatically to the **`logs/project.log`** and  **`projects/results/`** directory.

---

## 📊 Results

The framework provides:

* **Plots** comparing SHAP and LIME explanations.
* **CSV outputs** containing treatment-level fidelity, stability, and guideline concordance results.
* **Logs** recording each step of the experimental workflow.

---

## 🧾 Citation

If you use this repository for your research, please cite:

```
@article{shahcan,
  title={Can We Trust Explanation! Evaluation of Model-Agnostic Explanation Techniques on Highly Imbalanced, Multiclass-Multioutput Classification Problem},
  author={Shah, Syed Ihtesham Hussain and ten Teije, Annette and Volders, Jos{\'e}}
}
```

---

## 📌 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.