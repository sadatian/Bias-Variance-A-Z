# %% [markdown]
# # Module 07: Aggregation and Ensembles (Bagging & Boosting)
#
# ## Foundational Theory: Variance Reduction vs. Bias Reduction
#
# Bagging averages $B$ i.i.d. estimators with correlation $\rho$:
#
# $$
# \text{Var}\left( \frac{1}{B} \sum_{b=1}^B \hat{f}_b(x) \right) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2
# $$
#
# Boosting fits weak learners sequentially to negative gradients, iteratively reducing bias: $f_m(x) = f_{m-1}(x) + \eta h_m(x)$.

# %% [markdown]
# ### 1. Imports & Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor

sns.set_theme(style="whitegrid")
print("Module 07 environment initialized.")
