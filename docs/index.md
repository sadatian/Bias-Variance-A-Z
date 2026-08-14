# A-Z Bias-Variance Tradeoff Tutorial Series

Welcome to the **A-Z Bias-Variance Tradeoff Tutorial Series**, designed specifically for **Advanced Machine Learning Engineers**.

This tutorial series explores the core statistical mechanics, empirical risk behavior, penalty trajectories, cross-validation variance, dimensional effects, ensemble decomposition, and temporal complexity involved in machine learning model evaluation and optimization.

---

## **Series Outline**

1. **[01. Mathematical Error Decomposition](01_irreducible_error_and_decomposition.md)**  
   Mathematical decomposition of expected prediction error into Bias, Variance, and Irreducible Noise. Practical estimation using bootstrapping and k-NN proxies.

2. **[02. Model Complexity & Polynomials](02_model_complexity_and_polynomials.md)**  
   Visualizing the classic tradeoff via polynomial regression, condition numbers of design matrices, SVD analysis, and interpolation thresholds.

3. **[03. Learning Curves & Sample Complexity](03_learning_curves_and_sample_complexity.md)**  
   Empirical Risk vs. True Risk, sample complexity bounds, VC-dimension context, and confidence interval bands across training set scaling.

4. **[04. Penalty-Based Control (Regularization)](04_regularization.md)**  
   Ridge, Lasso, Elastic Net, Bayesian priors (Laplace vs Gaussian), LARS/Coordinate Descent paths, and exact shrinkage trajectories.

5. **[05. Cross-Validation Strategies](05_cross_validation_strategies.md)**  
   Assessment variance, LOOCV, Nested CV, Nadeau & Bengio variance corrections, and Block Time-Series splitting.

6. **[06. Feature Selection & Dimensionality](06_feature_selection_and_dimensionality.md)**  
   Curse of dimensionality, Filter/Wrapper methods, RFE, Stability selection via randomized Lasso, PCA vs. PLS variance mechanics.

7. **[07. Aggregation & Ensembles](07_ensembles_bagging_and_boosting.md)**  
   Variance reduction in Random Forests vs. Bias reduction in Gradient Boosting Machines (GBM) and Histogram-based boosting.

8. **[08. Temporally Complex Models](08_advanced_topics_temporally_complex_models.md)**  
   Bias-variance tradeoff in time-series (VAR, ARIMA), stationarity impact, lag-length optimization, and expanding/rolling window CV.

---

## **Environment & Tools**

- **Package Manager**: [`uv`](https://github.com/astral-sh/uv)
- **Primary Code Format**: Python files in Jupyter Percent format (`# %%` and `# %% [markdown]`)
- **Site Generator**: MkDocs with Material Theme (`uv run mkdocs build` / `uv run mkdocs serve`)
