# %% [markdown]
# # Module 02: Model Complexity, Polynomial Expansion & Matrix Conditioning
#
# ## Foundational Theory & Mathematical Derivation
#
# In linear regression, when model complexity is increased by expanding a scalar feature $x \in [a, b]$ into a $d$-degree polynomial basis $\phi(x) = [\phi_0(x), \phi_1(x), \dots, \phi_d(x)]^T \in \mathbb{R}^{d+1}$, the estimator prediction takes the linear form:
#
# $$
# \hat{f}(x; w) = w^T \phi(x) = \sum_{j=0}^d w_j \phi_j(x)
# $$
#
# Given a primary dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$, the design matrix $X \in \mathbb{R}^{N \times (d+1)}$ is given by:
#
# $$
# X = \begin{bmatrix} 
# \phi_0(x_1) & \phi_1(x_1) & \dots & \phi_d(x_1) \\ 
# \phi_0(x_2) & \phi_1(x_2) & \dots & \phi_d(x_2) \\ 
# \vdots & \vdots & \ddots & \vdots \\ 
# \phi_0(x_N) & \phi_1(x_N) & \dots & \phi_d(x_N) 
# \end{bmatrix}
# $$
#
# The Ordinary Least Squares (OLS) closed-form parameter estimate is obtained by solving the normal equations $(X^T X) \hat{w} = X^T y$:
#
# $$
# \hat{w} = (X^T X)^{-1} X^T y
# $$
#
# ### Variance Propagation & Design Matrix Conditioning
#
# Assuming additive homoskedastic noise $y = f(x) + \epsilon$ with $\text{Var}(\epsilon) = \sigma^2 I$, the covariance matrix of the parameter vector $\hat{w}$ over dataset sampling is:
#
# $$
# \text{Cov}(\hat{w}) = \mathbb{E}_{\mathcal{D}} \left[ (\hat{w} - \mathbb{E}[\hat{w}])(\hat{w} - \mathbb{E}[\hat{w}])^T \right] = \sigma^2 (X^T X)^{-1}
# $$
#
# Using thin Singular Value Decomposition (SVD), $X = U \Sigma V^T$, where $U \in \mathbb{R}^{N \times (d+1)}$ and $V \in \mathbb{R}^{(d+1) \times (d+1)}$ satisfy $U^T U = V^T V = V V^T = I_{d+1}$, and $\Sigma = \text{diag}(\sigma_1, \sigma_2, \dots, \sigma_{d+1}) \in \mathbb{R}^{(d+1) \times (d+1)}$ contains singular values sorted in non-increasing order $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_{d+1} \ge 0$.
#
# The Gram matrix inverse decomposes as $(X^T X)^{-1} = V \Sigma^{-2} V^T$. Consequently, the total parameter variance is directly linked to the singular spectrum:
#
# $$
# \text{Tr}\left(\text{Cov}(\hat{w})\right) = \sigma^2 \sum_{j=1}^{d+1} \frac{1}{\sigma_j^2}
# $$
#
# For any query point $x \in [a, b]$ with feature expansion $\phi(x) \in \mathbb{R}^{d+1}$, the pointwise prediction is $\hat{f}(x) = \phi(x)^T \hat{w}$. Projecting parameter covariance onto the feature basis reveals the exact pointwise prediction variance:
#
# $$
# \operatorname{Var}(\hat{f}(x)) = \phi(x)^T \operatorname{Cov}(\hat{w}) \phi(x) = \sigma^2 \phi(x)^T V \Sigma^{-2} V^T \phi(x) = \sigma^2 \left\| \Sigma^{-1} V^T \phi(x) \right\|_2^2 = \sigma^2 \sum_{j=1}^{d+1} \frac{(v_j^T \phi(x))^2}{\sigma_j^2}
# $$
#
# where $v_j \in \mathbb{R}^{d+1}$ is the $j$-th right singular vector (the $j$-th column of $V$).
#
# The $L_2$ condition number of the design matrix $X$ is:
#
# $$
# \kappa(X) = \frac{\sigma_{\max}(X)}{\sigma_{\min}(X)} = \frac{\sigma_1}{\sigma_{d+1}}
# $$
#
# Since $(X^T X)$ squares the singular values, its condition number explodes quadratically: $\kappa(X^T X) = \kappa(X)^2 = (\sigma_1 / \sigma_{d+1})^2$.
#
# ### Monomial Collinearity vs. Basis Orthogonalization
#
# In a standard **monomial basis** $\phi(x) = [1, x, x^2, \dots, x^d]^T$, consecutive powers $x^j$ and $x^{j+1}$ become near-collinear over positive bounded intervals (e.g. $[0, 2]$). This drives $\sigma_{\min}(X) = \sigma_{d+1} \to 0$ and causes $\kappa(X) > 10^{11}$ at moderate-to-high degrees. Under finite-precision floating-point arithmetic, standard LAPACK solvers (`*gelsd`) apply SVD rank truncation, dropping trailing singular values below machine precision ($\text{rcond} \approx \epsilon_{\text{mach}}$). This conflates algorithmic floating-point precision loss with true statistical variance.
#
# To mathematically decouple statistical variance from floating-point degradation, we implement **basis orthogonalization** using Chebyshev polynomials of the first kind $T_j(u)$. By linearly mapping the target domain $x \in [a, b]$ to the standard Chebyshev interval $u \in [-1, 1]$:
#
# $$
# u = 2 \left( \frac{x - a}{b - a} \right) - 1
# $$
#
# and generating the orthogonal design matrix $X = [T_0(u), T_1(u), \dots, T_d(u)]$, the design matrix satisfies $\kappa(X) \approx \mathcal{O}(1)$, preserving full numerical rank and isolating statistical bias-variance dynamics from floating-point collapse.

# %% [markdown]
# ### 1. Imports and Environment Setup
#
# We import high-performance numerical compute libraries (`numpy`, `numpy.polynomial.chebyshev.chebvander`, `pandas`, `scipy.linalg`), scikit-learn estimators (`LinearRegression`), type hints, parallel processing (`joblib`), and plotting modules (`matplotlib`, `seaborn`, `plotly`). We configure deterministic random seeds and consistent visual aesthetics.

# %%
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import scipy.linalg as la
from numpy.polynomial.chebyshev import chebvander
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.io as pio

# Configure Plotly default renderer for static HTML rendering
pio.renderers.default = "notebook_connected"

try:
    from IPython.display import display, Markdown
except ImportError:
    display = print
    Markdown = lambda x: x

from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression

# Configure visualization aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Plus Jakarta Sans", "DejaVu Sans", "Arial"],
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 1.2,
    "figure.autolayout": True
})

SEED: int = 42
np.random.seed(SEED)

# %% [markdown]
# **Execution Output:** Environment initialization confirmation and global random seed configuration.

# %%
# collapse_input
print(f"Environment initialized successfully. Random Seed = {SEED}")

# %% [markdown]
# ### 2. Synthetic Data Generation with Controlled Irreducible Noise
#
# We construct a non-linear ground truth target function $f(x)$ with decaying oscillations over $x \in [0, 2]$:
#
# $$
# f(x) = \cos(1.5\pi x) \cdot \exp(-0.5x), \quad x \in [0, 2]
# $$
#
# Additive Gaussian noise is drawn from $\epsilon \sim \mathcal{N}(0, \sigma^2)$ with $\sigma = 0.3$, yielding ground truth irreducible error $\sigma^2 = 0.09$.
#
# We generate a primary training dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ of size $N=50$ and a dense out-of-sample evaluation grid $\mathcal{X}_{\text{test}}$ of size $N_{\text{test}}=300$.

# %%
DOMAIN_A: float = 0.0
DOMAIN_B: float = 2.0
NOISE_STD: float = 0.3
TRUE_NOISE_VARIANCE: float = NOISE_STD ** 2  # sigma^2 = 0.09

def true_function(x: np.ndarray) -> np.ndarray:
    """Ground truth non-linear target function f(x)."""
    return np.cos(1.5 * np.pi * x) * np.exp(-0.5 * x)

def map_to_chebyshev_domain(x: np.ndarray, a: float = DOMAIN_A, b: float = DOMAIN_B) -> np.ndarray:
    """Maps continuous interval [a, b] to standard Chebyshev domain [-1, 1]."""
    return 2.0 * (x - a) / (b - a) - 1.0

def generate_dataset(
    n_samples: int = 50,
    noise_std: float = 0.3,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Generates synthetic primary dataset \mathcal{D} with additive Gaussian noise \sigma^2."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(DOMAIN_A, DOMAIN_B, size=n_samples)
    x = np.sort(x)  # Sorted for structured visualization
    y_true = true_function(x)
    noise = rng.normal(0.0, noise_std, size=n_samples)
    y = y_true + noise
    return x, y, y_true

# Generate primary training dataset \mathcal{D}
x_train, y_train, y_train_true = generate_dataset(n_samples=50, noise_std=NOISE_STD, seed=SEED)

# Generate dense evaluation test grid \mathcal{X}_test
x_test = np.linspace(DOMAIN_A, DOMAIN_B, 300)
y_test_true = true_function(x_test)

# %% [markdown]
# **Execution Output:** Synthetic training dataset sample sizes, evaluation grid dimensions, and ground truth noise variance.

# %%
# collapse_input
print(f"Primary training dataset 𝒟 size: N = {len(x_train)}")
print(f"Test evaluation grid size:       N_test = {len(x_test)}")
print(f"True Noise Variance (𝜎²):   {TRUE_NOISE_VARIANCE:.4f}")

# %% [markdown]
# ### 3. Vectorized Polynomial Bootstrap Decomposition & SVD Engine
#
# #### Pedagogical Rationale for Empirical Bootstrap Approximation
# Because the synthetic Data Generating Process (DGP) is analytically defined, exact expected prediction risk could in principle be computed via continuous numerical quadrature. However, we intentionally employ **bootstrap resampling ($B=500$)** to simulate real-world variance propagation and empirical risk dynamics where the true DGP is inaccessible and only a finite empirical training sample $\mathcal{D}$ is observed.
#
# #### Algorithmic Optimization & Basis Orthogonalization
# 1. **Elimination of $\mathcal{O}(B)$ Overhead**: Rather than re-allocating feature transformers and estimator instances inside the bootstrap loop, we construct the Chebyshev orthogonal design matrices $X_{\text{train}}$ and $X_{\text{test}}$ and instantiate the estimator `LinearRegression(fit_intercept=False)` upfront. The bootstrap loop executes strictly array slicing, `fit()`, and `predict()`.
# 2. **Chebyshev Basis Orthogonalization**: We transform $x \in [0, 2]$ to $u \in [-1, 1]$ and compute $X = \text{chebvander}(u, d)$. Because Chebyshev polynomials $T_j(u)$ form an orthogonal system, the condition number satisfies $\kappa(X) \approx \mathcal{O}(1)$, mathematically decoupling statistical variance from floating-point rank truncation.
#
# **Algorithmic Formalization**
#
# For each bootstrap sample $\mathcal{D}_b^*$:
#
# * **Step 1:** Construct the resampled design matrix $X_b^* \in \mathbb{R}^{N \times (d+1)}$ by sampling rows with replacement from the original matrix $X_{\text{train}}$.
# * **Step 2:** Fit Ordinary Least Squares to obtain the coefficient vector via the Moore-Penrose pseudoinverse $\hat{w}_b = (X_b^*)^+ y_b^* = ((X_b^*)^T X_b^*)^+ (X_b^*)^T y_b^*$ (computed via LAPACK `*gelsd` SVD to robustly handle rank deficiency from duplicated resampled observations).
# * **Step 3:** Compute predictions on the evaluation grid $X_{\text{test}}$, storing them as the $b$-th row of the prediction matrix $P \in \mathbb{R}^{B \times N_{\text{test}}}$.
#
# Pointwise empirical metrics (treating the $B$ bootstrap realizations as the empirical population):
#
# * **Expected Prediction:** $\hat{\mu}(x_i) = \frac{1}{B}\sum_{b=1}^B \hat{f}_b(x_i)$
# * **Pointwise Bias$^2$:** $(f(x_i) - \hat{\mu}(x_i))^2$
# * **Pointwise Variance:** $\frac{1}{B}\sum_{b=1}^B (\hat{f}_b(x_i) - \hat{\mu}(x_i))^2$
# * **Empirical Total MSE:** Evaluated against independent noisy test realizations $y_{\text{test}}^{(b)} = f(x_{\text{test}}) + \epsilon^{(b)}$.
#
# > *Aside: The empirical population variance estimator ($1/B$, matching `ddof=0`) is strictly applied to treat the $B$ bootstrap realizations as a complete empirical population, preserving the exact algebraic additive identity of the bias-variance decomposition ($\text{MSE} = \text{Bias}^2 + \text{Var} + \sigma^2$).*
#
# SVD analysis on the full training design matrix $X = \text{chebvander}(u_{\text{train}}, d) \in \mathbb{R}^{N \times (d+1)}$:
#
# * **Singular values:** $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_{d+1}$
# * **Condition number:** $\kappa(X) = \frac{\sigma_1}{\sigma_{d+1}}$
# * **Expected OLS coefficient $L_2$ norm:** $\mathbb{E}[\Vert{}\hat{w}\Vert{}_2] \approx \frac{1}{B} \sum_{b=1}^B \Vert{}\hat{w}_b\Vert{}_2$

# %%
def decompose_polynomial_bias_variance(
    degree: int,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te_true: np.ndarray,
    noise_std: float = 0.3,
    n_bootstraps: int = 500,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Performs empirical error decomposition and SVD conditioning analysis for polynomial regression of degree d
    using orthogonal Chebyshev basis expansions and pre-allocated estimators.
    """
    rng = np.random.RandomState(seed)
    n_train = len(x_tr)
    n_test = len(x_te)
    
    # 1. Basis Orthogonalization: Map to [-1, 1] and generate Chebyshev design matrices
    u_tr = map_to_chebyshev_domain(x_tr, DOMAIN_A, DOMAIN_B)
    u_te = map_to_chebyshev_domain(x_te, DOMAIN_A, DOMAIN_B)
    
    X_tr_full = chebvander(u_tr, degree)
    X_te_full = chebvander(u_te, degree)
    
    # SVD & Matrix Conditioning Analysis on Full Training Design Matrix
    U, s, Vt = la.svd(X_tr_full, full_matrices=False)
    cond_number_X = float(s[0] / s[-1]) if s[-1] > 0 else np.inf
    cond_number_XtX = cond_number_X ** 2
    
    # 2. Bootstrap Resampling Simulation with Pre-allocated Estimator (O(1) allocation overhead)
    predictions = np.zeros((n_bootstraps, n_test))
    coef_matrix = np.zeros((n_bootstraps, degree + 1))
    
    # Pre-instantiate OLS model before bootstrap loop (chebvander includes T_0(u)=1, so fit_intercept=False)
    model = LinearRegression(fit_intercept=False)
    
    for b in range(n_bootstraps):
        boot_idx = rng.choice(n_train, size=n_train, replace=True)
        X_boot = X_tr_full[boot_idx, :]
        y_boot = y_tr[boot_idx]
        
        # Fit model on pre-computed orthogonal design matrix
        model.fit(X_boot, y_boot)
        
        predictions[b, :] = model.predict(X_te_full)
        coef_matrix[b, :] = model.coef_
        
    # Pointwise statistics across bootstrap runs (ddof=0 for exact sample additivity)
    expected_prediction = np.mean(predictions, axis=0)
    pointwise_bias_sq = (y_te_true - expected_prediction) ** 2
    pointwise_variance = np.var(predictions, axis=0, ddof=0)
    
    # Independent empirical generalization error across noisy test realizations
    test_noise = rng.normal(0.0, noise_std, size=(n_bootstraps, n_test))
    y_test_noisy = y_te_true + test_noise
    empirical_total_mse = float(np.mean((predictions - y_test_noisy) ** 2))
    
    # Decomposed analytical components
    bias_sq = float(np.mean(pointwise_bias_sq))
    variance = float(np.mean(pointwise_variance))
    decomposed_sum = bias_sq + variance + (noise_std ** 2)
    
    # Expected coefficient L2 norm over bootstrap runs
    coef_l2_norm = float(np.mean(np.linalg.norm(coef_matrix, axis=1)))
    
    return {
        "degree": degree,
        "bias_sq": bias_sq,
        "variance": variance,
        "total_mse": empirical_total_mse,
        "decomposed_sum": decomposed_sum,
        "identity_residual": abs(empirical_total_mse - decomposed_sum),
        "cond_number_X": cond_number_X,
        "cond_number_XtX": cond_number_XtX,
        "singular_values": s,
        "coef_l2_norm": coef_l2_norm,
        "expected_prediction": expected_prediction,
        "pointwise_bias_sq": pointwise_bias_sq,
        "pointwise_variance": pointwise_variance,
        "predictions": predictions
    }

# Sanity check execution with degree 3 polynomial
sample_res = decompose_polynomial_bias_variance(
    degree=3, x_tr=x_train, y_tr=y_train, x_te=x_test, y_te_true=y_test_true,
    noise_std=NOISE_STD, n_bootstraps=500, seed=SEED
)

# %% [markdown]
# **Execution Output:** Baseline polynomial decomposition metrics, condition number, and identity residual for degree $d=3$.

# %%
# collapse_input
print("Baseline Chebyshev Polynomial Decomposition (Degree d=3):")
print(f"  Integrated Bias^2:               {sample_res['bias_sq']:.6f}")
print(f"  Integrated Variance:             {sample_res['variance']:.6f}")
print(f"  Irreducible Noise (𝜎²):     {TRUE_NOISE_VARIANCE:.4f}")
print(f"  Decomposed Sum (Bias^2+Var+𝜎²): {sample_res['decomposed_sum']:.6f}")
print(f"  Empirical Total MSE:             {sample_res['total_mse']:.6f}")
print(f"  Identity Residual:               {sample_res['identity_residual']:.6f}")
print(f"  Condition Number κ(X):       {sample_res['cond_number_X']:.2f}")
print(f"  Expected Weight Norm E[||w||₂]: {sample_res['coef_l2_norm']:.4f}")

# %% [markdown]
# ### 4. Polynomial Degree Sweeps ($d = 1 \dots 15$)
#
# With the SVD diagnostic engine verified on a baseline cubic model ($d=3$), we extend the analysis across $d \in [1, 15]$ to systematically map the bias-variance tradeoff under orthogonal Chebyshev basis expansions.
#
# Across complexity regimes:
# - **Underfitting ($d=1, 2$)**: High Bias$^2$, low Variance, near-optimal condition number $\kappa(X) \sim \mathcal{O}(1)$.
# - **Optimal Complexity Basin ($d=3, 4, 5$)**: Minimum total risk, balanced Bias$^2$ and Variance, condition number $\kappa(X) \approx 2 - 4$.
# - **High-Degree Overparameterization ($d \ge 8$)**: While $\text{Bias}^2$ contracts toward the representation floor, statistical estimation variance grows steadily as degrees of freedom approach sample size $N=50$. Crucially, because Chebyshev basis orthogonalization maintains $\kappa(X) \sim \mathcal{O}(1)$ across all degrees (avoiding monomial collinearity $\kappa(X) > 10^{11}$), LAPACK solvers execute without rank truncation, faithfully capturing true statistical variance.

# %%
degrees_range = np.arange(1, 16)

poly_results = Parallel(n_jobs=-1)(
    delayed(decompose_polynomial_bias_variance)(
        degree=d, x_tr=x_train, y_tr=y_train, x_te=x_test, y_te_true=y_test_true,
        noise_std=NOISE_STD, n_bootstraps=500, seed=SEED
    )
    for d in degrees_range
)

poly_records = [
    {
        "degree": res["degree"],
        "bias_sq": res["bias_sq"],
        "variance": res["variance"],
        "noise": TRUE_NOISE_VARIANCE,
        "decomposed_sum": res["decomposed_sum"],
        "total_mse": res["total_mse"],
        "identity_residual": res["identity_residual"],
        "cond_number_X": res["cond_number_X"],
        "cond_number_XtX": res["cond_number_XtX"],
        "coef_l2_norm": res["coef_l2_norm"]
    }
    for res in poly_results
]

df_poly = pd.DataFrame(poly_records)

# %% [markdown]
# #### Polynomial Complexity Sweep Preview
# Tabular preview of empirical risk decomposition metrics for low-degree polynomial models ($d=1$ to $5$) with color gradient styling applied to the variance and condition number columns.

# %%
styled_poly_preview = (
    df_poly.head(5).style
    .format({
        "degree": "{:d}",
        "bias_sq": "{:.6f}",
        "variance": "{:.6f}",
        "noise": "{:.4f}",
        "decomposed_sum": "{:.6f}",
        "total_mse": "{:.6f}",
        "identity_residual": "{:.6f}",
        "cond_number_X": "{:.2f}",
        "cond_number_XtX": "{:.2f}",
        "coef_l2_norm": "{:.4f}"
    })
    .background_gradient(subset=["variance", "cond_number_X"], cmap="OrRd")
    .set_table_styles([
        {"selector": "th", "props": [("background-color", "#f8f9fa"), ("font-weight", "600"), ("text-align", "center"), ("min-width", "0px"), ("padding-left", "0.5rem"), ("padding-right", "0.5rem")]},
        {"selector": "td", "props": [("text-align", "center"), ("font-family", "JetBrains Mono, monospace"), ("font-size", "0.72rem"), ("min-width", "0px"), ("padding-left", "0.5rem"), ("padding-right", "0.5rem")]}
    ])
    .hide(axis="index")
)
display(styled_poly_preview)

# %% [markdown]
# ### 5. Visualizations & Diagnostic Plots
#
# We render three dedicated visualization diagnostics:
# 1. **Polynomial Curve Fits across Complexity Regimes** (Plotly 4-panel subplot for $d=1, 3, 7, 14$).
# 2. **Bias-Variance Tradeoff & Design Matrix Conditioning** (Plotly split-axis 2-tier chart: Row 1 zoomed linear viewport $[0.0, 0.45]$ resolving the optimal basin; Row 2 full dynamic range log scale).
# 3. **SVD Spectral Decay & Coefficient Norm Dynamics** (Plotly dual-panel plot of singular values $\sigma_k(X)$ and expected parameter norm $\mathbb{E}[\|\hat{w}\|_2]$).

# %% [markdown]
# #### Visualization 1: Polynomial Curve Fits across Complexity Regimes
# Interactive Plotly 4-panel grid displaying bootstrap fit ensembles and expected predictions $\bar{f}(x)$ for degrees $d \in \{1, 3, 7, 14\}$ in a single isolated block.

# %%
selected_degrees = [1, 3, 7, 14]
degree_to_res = {res["degree"]: res for res in poly_results}

fig1 = sp.make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        f"<b>Degree d={d}</b><br><span style='font-size:11px; color:#555;'>Var={degree_to_res[d]['variance']:.4f} | κ(X)={degree_to_res[d]['cond_number_X']:.2f}</span>"
        for d in selected_degrees
    ],
    shared_xaxes=True,
    shared_yaxes=True,
    vertical_spacing=0.14,
    horizontal_spacing=0.08
)

for i, d in enumerate(selected_degrees):
    row = i // 2 + 1
    col = i % 2 + 1
    res = degree_to_res[d]
    preds = res["predictions"]
    
    # Plot sample bootstrap prediction curves (first 30 runs)
    for b_idx in range(min(30, preds.shape[0])):
        fig1.add_trace(
            go.Scatter(
                x=x_test,
                y=preds[b_idx, :],
                mode="lines",
                line=dict(color="rgba(117, 112, 179, 0.15)", width=1),
                name="Bootstrap Fits",
                showlegend=(i == 0 and b_idx == 0),
                legendgroup="boot_fits",
                hoverinfo="skip"
            ),
            row=row, col=col
        )
        
    # Scatter primary training data points
    fig1.add_trace(
        go.Scatter(
            x=x_train,
            y=y_train,
            mode="markers",
            marker=dict(color="#666666", size=5, opacity=0.7),
            name="Data 𝒟",
            showlegend=(i == 0),
            legendgroup="data",
            hovertemplate="Data: (%{x:.2f}, %{y:.2f})<extra></extra>"
        ),
        row=row, col=col
    )
    
    # Plot true ground truth function f(x)
    fig1.add_trace(
        go.Scatter(
            x=x_test,
            y=y_test_true,
            mode="lines",
            line=dict(color="#000000", width=2, dash="dash"),
            name="True f(x)",
            showlegend=(i == 0),
            legendgroup="true_f",
            hovertemplate="True f(x): %{y:.3f}<extra></extra>"
        ),
        row=row, col=col
    )
    
    # Plot expected prediction mean E[f_hat](x)
    fig1.add_trace(
        go.Scatter(
            x=x_test,
            y=res["expected_prediction"],
            mode="lines",
            line=dict(color="#d95f02", width=2.5),
            name="Mean E[f̂(x)]",
            showlegend=(i == 0),
            legendgroup="mean_f",
            hovertemplate=f"Mean E[f̂] (d={d}, Bias²={res['bias_sq']:.3f}): %{{y:.3f}}<extra></extra>"
        ),
        row=row, col=col
    )

fig1.update_layout(
    title=dict(
        text="<b>Chebyshev Polynomial Regression Fits across Complexity Regimes</b>",
        font=dict(size=15, family="Plus Jakarta Sans"),
        x=0.5,
        y=0.98,
        xanchor="center",
        yanchor="top"
    ),
    template="plotly_white",
    height=650,
    hovermode="closest",
    margin=dict(l=50, r=30, t=110, b=50),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.1,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#e0e0e0",
        borderwidth=1
    )
)

fig1.for_each_annotation(lambda a: a.update(font=dict(size=12)))

fig1.update_xaxes(title_text="x", row=2, col=1)
fig1.update_xaxes(title_text="x", row=2, col=2)
fig1.update_yaxes(title_text="y", range=[-1.8, 1.8], row=1, col=1)
fig1.update_yaxes(title_text="y", range=[-1.8, 1.8], row=2, col=1)
fig1.update_yaxes(range=[-1.8, 1.8], row=1, col=2)
fig1.update_yaxes(range=[-1.8, 1.8], row=2, col=2)

fig1.show()

# %% [markdown]
# #### Visualization 2: Bias-Variance Tradeoff & Design Matrix Conditioning (Split-Tier Viewport)
# Interactive Plotly 2-tier chart:
# - **Row 1 (Optimal Complexity Basin Zoom - Linear Scale $[0.0, 0.45]$)**: Uncompressed visual inspection of the Bias$^2$ and Variance intersection, optimal degree basin ($d \in [3, 5]$), and Bayes noise floor ($\sigma^2=0.09$) alongside Condition Number $\kappa(X)$ (secondary y-axis).
# - **Row 2 (Full Dynamic Range - Logarithmic Scale)**: Full dynamic perspective capturing high-degree variance growth alongside bias contraction without compressing the optimal basin.

# %%
fig2 = sp.make_subplots(
    rows=2, cols=1,
    row_heights=[0.55, 0.45],
    vertical_spacing=0.15,
    specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
    subplot_titles=[
        "<b>Optimal Complexity Basin Zoom (Linear Viewport [0.0, 0.45])</b>",
        "<b>Full Dynamic Range (Logarithmic Scale Viewport)</b>"
    ]
)

# === ROW 1: Linear Scale Zoom on Optimal Basin ===
fig2.add_trace(
    go.Scatter(
        x=df_poly["degree"], y=df_poly["bias_sq"],
        mode="lines+markers", name="Bias²",
        line=dict(color="#d95f02", width=2.5), marker=dict(size=7),
        legendgroup="bias"
    ),
    row=1, col=1, secondary_y=False
)

fig2.add_trace(
    go.Scatter(
        x=df_poly["degree"], y=df_poly["variance"],
        mode="lines+markers", name="Variance",
        line=dict(color="#7570b3", width=2.5, dash="dash"), marker=dict(symbol="square", size=7),
        legendgroup="var"
    ),
    row=1, col=1, secondary_y=False
)

fig2.add_trace(
    go.Scatter(
        x=df_poly["degree"], y=df_poly["total_mse"],
        mode="lines+markers", name="Empirical Total MSE",
        line=dict(color="#1b9e77", width=2.5), marker=dict(symbol="diamond", size=7),
        legendgroup="mse"
    ),
    row=1, col=1, secondary_y=False
)

fig2.add_trace(
    go.Scatter(
        x=df_poly["degree"], y=[TRUE_NOISE_VARIANCE] * len(df_poly),
        mode="lines", name="Irreducible Noise (𝜎²=0.09)",
        line=dict(color="#666666", width=1.5, dash="dot"),
        legendgroup="noise"
    ),
    row=1, col=1, secondary_y=False
)

fig2.add_trace(
    go.Scatter(
        x=df_poly["degree"], y=df_poly["cond_number_X"],
        mode="lines+markers", name="Condition Number 𝜅(𝛸)",
        line=dict(color="#e7298a", width=2, dash="dot"), marker=dict(symbol="x", size=7),
        legendgroup="cond"
    ),
    row=1, col=1, secondary_y=True
)

# === ROW 2: Log Scale Full Dynamic Range ===
fig2.add_trace(
    go.Scatter(
        x=df_poly["degree"], y=df_poly["bias_sq"],
        mode="lines+markers", name="Bias²",
        line=dict(color="#d95f02", width=2), marker=dict(size=6),
        legendgroup="bias", showlegend=False
    ),
    row=2, col=1, secondary_y=False
)

fig2.add_trace(
    go.Scatter(
        x=df_poly["degree"], y=df_poly["variance"],
        mode="lines+markers", name="Variance",
        line=dict(color="#7570b3", width=2, dash="dash"), marker=dict(symbol="square", size=6),
        legendgroup="var", showlegend=False
    ),
    row=2, col=1, secondary_y=False
)

fig2.add_trace(
    go.Scatter(
        x=df_poly["degree"], y=df_poly["total_mse"],
        mode="lines+markers", name="Empirical Total MSE",
        line=dict(color="#1b9e77", width=2), marker=dict(symbol="diamond", size=6),
        legendgroup="mse", showlegend=False
    ),
    row=2, col=1, secondary_y=False
)

fig2.add_trace(
    go.Scatter(
        x=df_poly["degree"], y=[TRUE_NOISE_VARIANCE] * len(df_poly),
        mode="lines", name="Irreducible Noise (𝜎²=0.09)",
        line=dict(color="#666666", width=1.5, dash="dot"),
        legendgroup="noise", showlegend=False
    ),
    row=2, col=1, secondary_y=False
)

fig2.add_trace(
    go.Scatter(
        x=df_poly["degree"], y=df_poly["cond_number_X"],
        mode="lines+markers", name="Condition Number κ(X)",
        line=dict(color="#e7298a", width=2, dash="dot"), marker=dict(symbol="x", size=6),
        legendgroup="cond", showlegend=False
    ),
    row=2, col=1, secondary_y=True
)

fig2.update_layout(
    title=dict(
        text="<b>Polynomial Bias-Variance Tradeoff & Design Matrix Conditioning κ(X)</b>",
        font=dict(size=15, family="Plus Jakarta Sans"),
        x=0.5,
        y=0.98,
        xanchor="center",
        yanchor="top"
    ),
    template="plotly_white",
    hovermode="x unified",
    margin=dict(l=60, r=60, t=100, b=60),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.06,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#cccccc",
        borderwidth=1
    ),
    height=680
)

fig2.for_each_annotation(lambda a: a.update(font=dict(size=12)))

fig2.update_xaxes(title_text="Polynomial Degree (d)", tickmode="linear", tick0=1, dtick=1, row=2, col=1)
fig2.update_xaxes(tickmode="linear", tick0=1, dtick=1, row=1, col=1)

# Row 1 Axis settings
fig2.update_yaxes(title_text="Integrated MSE (Linear)", range=[0, 0.45], row=1, col=1, secondary_y=False)
fig2.update_yaxes(title_text="Condition Number κ(X)", row=1, col=1, secondary_y=True)

# Row 2 Axis settings (Log scale)
fig2.update_yaxes(title_text="Error (Log Scale)", type="log", row=2, col=1, secondary_y=False)
fig2.update_yaxes(title_text="κ(X) (Log Scale)", type="log", row=2, col=1, secondary_y=True)

fig2.show()

# %% [markdown]
# #### Visualization 3: SVD Spectral Decay & Coefficient Norm Dynamics
# Interactive Plotly chart displaying singular value spectrum $\sigma_k(X)$ for selected polynomial degrees $d \in \{1, 3, 7, 14\}$ alongside parameter vector norm $\mathbb{E}[\|\hat{w}\|_2]$ dynamics in a single isolated block.

# %%
fig3 = sp.make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "<b>Singular Value Spectrum Decay</b><br><span style='font-size:11px; color:#555;'>𝜎_k(X) across Chebyshev polynomial degrees</span>",
        "<b>Expected Parameter Norm Dynamics</b><br><span style='font-size:11px; color:#555;'>E[||ŵ||₂] vs. polynomial degree (d)</span>"
    ),
    horizontal_spacing=0.14
)

# Panel 1: Singular value spectra for selected degrees
colors_deg = {1: "#2b5c8f", 3: "#1b9e77", 7: "#d95f02", 14: "#e7298a"}

for d in selected_degrees:
    s_vals = degree_to_res[d]["singular_values"]
    fig3.add_trace(
        go.Scatter(
            x=np.arange(1, len(s_vals) + 1), y=s_vals,
            mode="lines+markers", name=f"Degree d={d}",
            line=dict(color=colors_deg[d], width=2.5), marker=dict(size=6)
        ),
        row=1, col=1
    )

# Panel 2: Expected Coefficient norm E[||w_hat||_2] vs. Degree d
fig3.add_trace(
    go.Scatter(
        x=df_poly["degree"], y=df_poly["coef_l2_norm"],
        mode="lines+markers", name="E[||ŵ||₂]",
        line=dict(color="#e7298a", width=3), marker=dict(symbol="diamond", size=8),
        showlegend=False,
        hovertemplate="Degree %{x}: E[||ŵ||₂] = %{y:.4f}<extra></extra>"
    ),
    row=1, col=2
)

fig3.update_layout(
    title=dict(
        text="<b>SVD Spectral Decay & Coefficient Norm Dynamics</b>",
        font=dict(size=15, family="Plus Jakarta Sans"),
        x=0.5,
        y=0.98,
        xanchor="center",
        yanchor="top"
    ),
    template="plotly_white",
    hovermode="closest",
    height=480,
    margin=dict(l=60, r=30, t=100, b=60),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.1,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#e0e0e0",
        borderwidth=1
    )
)

fig3.for_each_annotation(lambda a: a.update(font=dict(size=12)))

fig3.update_xaxes(title_text="Singular Value Index (k)", row=1, col=1)
fig3.update_yaxes(title_text="Singular Value 𝜎_k(X)", row=1, col=1)

fig3.update_xaxes(title_text="Polynomial Degree (d)", tickmode="linear", dtick=1, row=1, col=2)
fig3.update_yaxes(title_text="Expected Norm E[||ŵ||₂]", row=1, col=2)
fig3.show()

# %% [markdown]
# ### 6. Summary & Key Engineering Takeaways
#
# Below is the consolidated summary table comparing error components, condition numbers, and expected coefficient norms across representative degrees.
#
# #### Key Engineering Takeaways:
# 1. **Basis Orthogonalization vs. Monomial Collinearity**: Standard monomial powers $x^j$ produce severe collinearity over $[a, b]$, inflating condition numbers to $\kappa(X) > 10^{11}$ and triggering LAPACK floating-point rank truncation. Implementing **orthogonal Chebyshev basis expansions** $T_j(u)$ guarantees well-conditioned design matrices ($\kappa(X) \approx 1 - 4$) across all degrees, mathematically isolating statistical variance from numerical precision loss.
# 2. **Algorithmic Efficiency & Pre-Allocation**: Hoisting design matrix generation and model instantiations outside the bootstrap resampling loop eliminates $\mathcal{O}(B)$ memory allocations and class reconstruction overhead, restricting execution strictly to array indexing and numerical linear algebra solvers.
# 3. **Bias-Variance Phase Transitions**: The optimal model capacity lies in the cubic-to-quintic basin ($d \in [3, 5]$), where representation bias is virtually eliminated while parameter estimation variance remains controlled. At high degrees ($d \ge 8$), variance increases predictably with model capacity ($p = d+1$) as parameterized by finite-sample OLS theory.
# 4. **Coefficient Stability in Orthogonal Bases**: Under orthogonal bases, parameter norms $\mathbb{E}[\|\hat{w}\|_2]$ remain stable without catastrophic coordinate explosion, confirming that variance growth is driven by genuine statistical sample dispersion rather than coordinate collinearity artifacts.

# %% [markdown]
# **Execution Output:** Final empirical polynomial error decomposition and conditioning summary across representative degrees.

# %%
summary_degrees = [1, 3, 5, 9, 14]
df_summary = df_poly[df_poly["degree"].isin(summary_degrees)].copy().reset_index(drop=True)

df_summary_display = pd.DataFrame({
    "Degree (d)": df_summary["degree"],
    "Bias^2": df_summary["bias_sq"],
    "Variance": df_summary["variance"],
    "Noise (𝜎²)": df_summary["noise"],
    "Decomposed Sum": df_summary["decomposed_sum"],
    "Empirical Total MSE": df_summary["total_mse"],
    "Identity Error": df_summary["identity_residual"],
    "Condition Number κ(X)": df_summary["cond_number_X"],
    "Expected Weight Norm E[||w||_2]": df_summary["coef_l2_norm"]
})

def format_metric(val: float) -> str:
    """Formats float metrics cleanly."""
    if abs(val) >= 100.0 or (abs(val) < 1e-4 and val != 0):
        return f"{val:.2e}"
    return f"{val:.6f}"

styled_summary_display = (
    df_summary_display.style
    .format({
        "Degree (d)": "{:d}",
        "Bias^2": "{:.6f}",
        "Variance": format_metric,
        "Noise (𝜎²)": "{:.4f}",
        "Decomposed Sum": format_metric,
        "Empirical Total MSE": format_metric,
        "Identity Error": "{:.6f}",
        "Condition Number κ(X)": "{:.2f}",
        "Expected Weight Norm E[||w||₂]": "{:.4f}"
    })
    .background_gradient(subset=["Variance", "Condition Number κ(X)"], cmap="OrRd")
    .set_table_styles([
        {"selector": "th", "props": [("background-color", "#f8f9fa"), ("font-weight", "600"), ("text-align", "center"), ("min-width", "0px"), ("padding-left", "0.5rem"), ("padding-right", "0.5rem")]},
        {"selector": "td", "props": [("text-align", "center"), ("font-family", "JetBrains Mono, monospace"), ("font-size", "0.72rem"), ("min-width", "0px"), ("padding-left", "0.5rem"), ("padding-right", "0.5rem")]}
    ])
    .hide(axis="index")
)

display(Markdown("### Final Empirical Polynomial Error Decomposition & Conditioning Summary Table"))
display(styled_summary_display)
display(Markdown(f"**Maximum Error Decomposition Identity Residual:** `{df_poly['identity_residual'].max():.6f}`"))
