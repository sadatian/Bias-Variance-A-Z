# %% [markdown]
# # Module 01: Mathematical Error Decomposition
#
# ## Foundational Theory & Mathematical Derivation
#
# In statistical learning theory, for a target variable $y = f(x) + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2)$, the expected prediction error of a regression model $\hat{f}(x)$ at a target query point $x$ can be decomposed cleanly into three orthogonal components:
#
# $$
# \mathbb{E}_{\mathcal{D}, \epsilon} \left[ \left( y - \hat{f}(x; \mathcal{D}) \right)^2 \right] = \underbrace{\left( f(x) - \mathbb{E}_{\mathcal{D}}[\hat{f}(x; \mathcal{D})] \right)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}_{\mathcal{D}} \left[ \left( \hat{f}(x; \mathcal{D}) - \mathbb{E}_{\mathcal{D}}[\hat{f}(x; \mathcal{D})] \right)^2 \right]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible Noise}}
# $$
#
# This tutorial module demonstrates this error decomposition empirically using bootstrap resampling on synthetic and benchmark data.

# %% [markdown]
# ### 1. Imports and Environment Setup
#
# We import numerical, statistical, machine learning, and plotting libraries.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid")
np.random.seed(42)
print("Environment initialized cleanly.")
