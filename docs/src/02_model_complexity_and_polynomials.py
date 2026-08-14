# %% [markdown]
# # Module 02: Model Complexity and Polynomials
#
# ## Foundational Theory: Polynomial Degree & Design Matrix Conditioning
#
# As model complexity scales (e.g., increasing degree $d$ in polynomial regression $\hat{f}(x) = \sum_{j=0}^d w_j x^j$), the design matrix $X \in \mathbb{R}^{n \times (d+1)}$ becomes increasingly ill-conditioned. The condition number $\kappa(X) = \frac{\sigma_{\max}(X)}{\sigma_{\min}(X)}$ explodes, leading to numerical instability in the normal equations $(X^T X)^{-1} X^T y$ and severe variance inflation.

# %% [markdown]
# ### 1. Imports & Setup
#
# We import linear model utilities, polynomial feature generators, and SVD tools.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

sns.set_theme(style="whitegrid")
print("Module 02 environment initialized.")
