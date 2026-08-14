# A-Z Bias-Variance Tradeoff Tutorial Series

An advanced, hands-on tutorial series exploring the **Bias-Variance Tradeoff** from the perspective of an Advanced Machine Learning Engineer.

> *Note: Deep Learning and Neural Network implications are strictly excluded. The focus is exclusively on standard statistical machine learning algorithms and methods.*

---

## 🛠️ Environment & Package Management

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible dependency and environment management.

### Installation & Setup

Sync the virtual environment and install all required dependencies:

```bash
uv sync
```

### Documentation Site Build

Build the static site documentation using MkDocs:

```bash
uv run mkdocs build
```

To preview the documentation locally with live reload:

```bash
uv run mkdocs serve
```

---

## 📚 Tutorial Series Structure

Each tutorial is written as a standalone Python file (`.py`) formatted in Jupyter Percent format (`# %%` code cells and `# %% [markdown]` explanation cells):

1. `01_irreducible_error_and_decomposition.py` - Mathematical Error Decomposition (Bias, Variance, Irreducible Noise)
2. `02_model_complexity_and_polynomials.py` - Model Complexity, Polynomial Regression, & Matrix Conditioning
3. `03_learning_curves_and_sample_complexity.py` - Learning Curves, Empirical vs. True Risk, & Sample Complexity
4. `04_regularization.py` - Penalty-Based Control (Ridge, Lasso, Elastic Net, LARS & Shrinkage Trajectories)
5. `05_cross_validation_strategies.py` - Cross-Validation Variance, LOOCV, Nested CV, & Nadeau-Bengio Correction
6. `06_feature_selection_and_dimensionality.py` - Feature Selection, Dimensionality, Stability Selection, & PCA vs. PLS
7. `07_ensembles_bagging_and_boosting.py` - Aggregation & Weak Learners (Random Forests vs. GBM & HistGradientBoosting)
8. `08_advanced_topics_temporally_complex_models.py` - Temporally Complex Models (VAR, ARIMA, & Stationarity Effects)
