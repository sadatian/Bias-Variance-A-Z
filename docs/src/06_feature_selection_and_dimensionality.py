# %% [markdown]
# # Module 06: Feature Selection and Dimensionality
#
# ## Foundational Theory: Curse of Dimensionality & Variance Expansion
#
# As feature dimension $p$ grows relative to sample size $n$, Euclidean distance becomes non-informative, and linear models suffer from inflated coefficient variance $\text{Var}(\hat{w}_j) = \frac{\sigma^2}{(1 - R_j^2) \sum (x_{ij} - \bar{x}_j)^2}$.

# %% [markdown]
# ### 1. Imports & Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

sns.set_theme(style="whitegrid")
print("Module 06 environment initialized.")
