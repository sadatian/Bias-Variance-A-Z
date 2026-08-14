# %% [markdown]
# # Module 05: Cross-Validation Strategies
#
# ## Foundational Theory: Estimator Variance & Correction Methods
#
# Standard K-Fold Cross-Validation splits dataset $\mathcal{D}$ into $K$ partitions. The variance of the cross-validation error estimator is non-trivial due to overlap across training folds. We analyze the Nadeau & Bengio corrected variance estimate:
#
# $$
# \text{Var}_{\text{NB}}(\hat{\mu}) = \left( \frac{1}{K} + \frac{n_{\text{test}}}{n_{\text{train}}} \right) S^2
# $$

# %% [markdown]
# ### 1. Imports & Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, LeaveOneOut, TimeSeriesSplit

sns.set_theme(style="whitegrid")
print("Module 05 environment initialized.")
