# %% [markdown]
# # Module 08: Temporally Complex Models
#
# ## Foundational Theory: Time Series Bias-Variance Dynamics
#
# In dynamic time-series models (VAR, ARIMA), non-stationarity and autocorrelation distort residual variance estimates. We analyze parameter inflation across lag lengths $p$ in $\mathbf{y}_t = c + \sum_{i=1}^p A_i \mathbf{y}_{t-i} + \boldsymbol{\epsilon}_t$.

# %% [markdown]
# ### 1. Imports & Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA

sns.set_theme(style="whitegrid")
print("Module 08 environment initialized.")
