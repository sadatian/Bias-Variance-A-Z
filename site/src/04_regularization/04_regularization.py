# %% [markdown]
# # Module 04: Regularization (Penalty-Based Control)
#
# ## Foundational Theory: Shrinkage Paths & Bayesian Priors
#
# Regularization constrains the parameter space via penalty terms:
#
# $$
# \min_{w} \left\{ \frac{1}{2n} \|y - Xw\|_2^2 + \lambda_1 \|w\|_1 + \frac{\lambda_2}{2} \|w\|_2^2 \right\}
# $$
#
# Elastic Net bridges Lasso ($\ell_1$, Laplace prior) and Ridge ($\ell_2$, Gaussian prior), explicitly controlling coefficient variance and feature selection stability.

# %% [markdown]
# ### 1. Imports & Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso, ElasticNet, lars_path

sns.set_theme(style="whitegrid")
print("Module 04 environment initialized.")
