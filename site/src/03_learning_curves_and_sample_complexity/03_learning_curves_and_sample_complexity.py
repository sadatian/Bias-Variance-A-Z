# %% [markdown]
# # Module 03: Learning Curves and Sample Complexity
#
# ## Foundational Theory: Empirical Risk vs. True Risk & Sample Scaling
#
# Empirical Risk Minimization (ERM) minimizes the training error $\hat{R}_S(h) = \frac{1}{m} \sum_{i=1}^m L(h(x_i), y_i)$. As training sample size $m \to \infty$, the generalization gap $|R(h) - \hat{R}_S(h)|$ contracts at rate $\mathcal{O}\left(\sqrt{\frac{\text{VCdim}(\mathcal{H})}{m}}\right)$.

# %% [markdown]
# ### 1. Imports & Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve

sns.set_theme(style="whitegrid")
print("Module 03 environment initialized.")
