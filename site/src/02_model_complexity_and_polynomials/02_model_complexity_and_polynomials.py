# %% [markdown]
# # Module 02: Model Complexity, Polynomial Expansion & Matrix Conditioning
#
# ## Foundational Theory & Mathematical Derivation
#
# In linear regression, when model complexity is increased by expanding a scalar feature $x \in \mathbb{R}$ into a $d$-degree polynomial basis $\phi(x) = [1, x, x^2, \dots, x^d]^T \in \mathbb{R}^{d+1}$, the estimator prediction takes the linear form:
#
# $$
# \hat{f}(x; w) = w^T \phi(x) = \sum_{j=0}^d w_j x^j
# $$
#
# Given a dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$, the design matrix $X \in \mathbb{R}^{N \times (d+1)}$ is the rectangular Vandermonde matrix:
#
# $$
# X = \begin{bmatrix} 
# 1 & x_1 & x_1^2 & \dots & x_1^d \\ 
# 1 & x_2 & x_2^2 & \dots & x_2^d \\ 
# \vdots & \vdots & \vdots & \ddots & \vdots \\ 
# 1 & x_N & x_N^2 & \dots & x_N^d 
# \end{bmatrix}
# $$
#
# The Ordinary Least Squares (OLS) closed-form parameter estimate is obtained by solving the normal equations $(X^T X) \hat{w} = X^T y$:
#
# $$
# \hat{w} = (X^T X)^{-1} X^T y
# $$
#
# ### 1. Variance Propagation & Design Matrix Conditioning
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
# The L2 condition number of the design matrix $X$ is defined as:
#
# $$
# \kappa(X) = \frac{\sigma_{\max}(X)}{\sigma_{\min}(X)} = \frac{\sigma_1}{\sigma_{d+1}}
# $$
#
# Since $(X^T X)$ squares the singular values, its condition number explodes quadratically: $\kappa(X^T X) = \kappa(X)^2 = (\sigma_1 / \sigma_{d+1})^2$.
#
# As degree $d$ increases, higher powers $x^j$ and $x^{j+1}$ become near-linearly dependent over bounded intervals (e.g. $[0, 2]$), causing $\sigma_{\min}(X) \to 0$. As $\sigma_{\min}(X)$ approaches machine precision $\epsilon_{\text{mach}} \approx 10^{-16}$, $\kappa(X)$ explodes towards $\infty$, inflating parameter variance $\text{Var}(\hat{w}_j)$ and driving prediction variance $\|\text{Var}(\hat{f}(x))\| \to \infty$.
#
# ### System Architecture & SVD Conditioning Pipeline
#
# The flowchart below outlines how polynomial feature expansion induces design matrix ill-conditioning, weight norm explosion, and empirical error decomposition.
#
# ```mermaid
# graph TD
#     A["Training Sample D0 (N points)"] -->|Polynomial Feature Expansion| B["Vandermonde Design Matrix X (N x d+1)"]
#     B -->|SVD Factorization U S V^T| C["Singular Value Spectrum sigma_i(X)"]
#     C -->|Ratio sigma_max / sigma_min| D["Condition Number kappa(X)"]
#     B -->|Bootstrap Resampling B reps| E["Bootstrap Datasets D_b*"]
#     E -->|OLS Normal Equations| F["Parameter Vectors w_hat_b"]
#     F -->|Compute Expected L2 Norm| G["Expected Weight Norm E[||w_hat||_2]"]
#     F -->|Predict on X_test| H["Prediction Matrix B x N_test"]
#     H -->|Empirical Decomposition| I["Bias^2, Variance, Total MSE"]
#     D -->|Correlate with Variance| J["Ill-Conditioning & Error Explosion"]
#     G -->|Correlate with Overfitting| J
#     I --> J
# ```

# %% [markdown]
# ### 1. Imports and Environment Setup
#
# We import high-performance numerical compute libraries (`numpy`, `pandas`, `scipy.linalg`), scikit-learn preprocessing and linear regression models, type hints, and plotting modules (`matplotlib`, `seaborn`, `plotly`). We configure deterministic random seeds and consistent visual aesthetics.

# %%
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import scipy.linalg as la
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
from sklearn.preprocessing import PolynomialFeatures
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
# #### Environment Initialization Output
# Displays system status and global random seed configuration.

# %%
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
# We generate a primary training dataset $\mathcal{D}_0$ of size $N=50$ and a dense out-of-sample evaluation grid $\mathcal{X}_{\text{test}}$ of size $N_{\text{test}}=300$.

# %%
def true_function(x: np.ndarray) -> np.ndarray:
    """Ground truth non-linear target function f(x)."""
    return np.cos(1.5 * np.pi * x) * np.exp(-0.5 * x)

def generate_dataset(
    n_samples: int = 50,
    noise_std: float = 0.3,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates synthetic dataset with additive Gaussian noise sigma^2."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.0, 2.0, size=n_samples)
    x = np.sort(x)  # Sorted for structured visualization
    y_true = true_function(x)
    noise = rng.normal(0.0, noise_std, size=n_samples)
    y = y_true + noise
    return x, y, y_true

# Define noise variance parameters
NOISE_STD: float = 0.3
TRUE_NOISE_VARIANCE: float = NOISE_STD ** 2  # sigma^2 = 0.09

# Generate primary training dataset D0
x_train, y_train, y_train_true = generate_dataset(n_samples=50, noise_std=NOISE_STD, seed=SEED)

# Generate dense evaluation test grid X_test
x_test = np.linspace(0.0, 2.0, 300)
y_test_true = true_function(x_test)

# %% [markdown]
# #### Synthetic Dataset Summary Output
# Displays sample sizes and ground truth noise variance.

# %%
print(f"Training dataset D0 size:       N = {len(x_train)}")
print(f"Test evaluation grid size:      N_test = {len(x_test)}")
print(f"True Noise Variance (sigma^2):  {TRUE_NOISE_VARIANCE:.4f}")

# %% [markdown]
# ### 3. Vectorized Polynomial Bootstrap Decomposition & SVD Engine
#
# To quantify empirical error components and matrix conditioning for a given polynomial degree $d$, we perform bootstrap resampling with $B=500$ iterations.
#
# For each bootstrap sample $\mathcal{D}_b^*$:
# 1. Transform training inputs into Vandermonde polynomial features $X_{\text{boot}} = \phi(x_{\text{boot}})$.
# 2. Fit Ordinary Least Squares `LinearRegression()` to obtain coefficient vector $\hat{w}_b$.
# 3. Predict on evaluation grid $X_{\text{test}} = \phi(x_{\text{test}})$, filling prediction matrix $\mathbf{P} \in \mathbb{R}^{B \times N_{\text{test}}}$.
#
# Pointwise metrics use `ddof=0` for population variance additivity:
# - Pointwise Bias$^2$: $(f(x_i) - \bar{f}(x_i))^2$
# - Pointwise Variance: $\text{Var}(\mathbf{P}_{:, i}, \text{ddof}=0)$
# - Empirical Total MSE: Evaluated against independent noisy test realizations $y_{\text{test}, b} = f(x_{\text{test}}) + \epsilon_b$.
#
# In addition, we analyze the SVD of the uncentered full training design matrix $X = \phi(x_{\text{train}}) \in \mathbb{R}^{N \times (d+1)}$:
# - Singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_{d+1}$
# - Design matrix condition number $\kappa(X) = \sigma_1 / \sigma_{d+1}$
# - Expected OLS coefficient vector norm $\mathbb{E}[\|\hat{w}\|_2] = \frac{1}{B} \sum_{b=1}^B \|\hat{w}_b\|_2$ (evaluated per bootstrap trial to capture coefficient variance explosion without sign cancellation).

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
    Performs empirical error decomposition and SVD conditioning analysis for polynomial regression of degree d.
    """
    rng = np.random.RandomState(seed)
    n_train = len(x_tr)
    n_test = len(x_te)
    
    # 1. SVD & Matrix Conditioning Analysis on Full Training Set Design Matrix
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_tr_full = poly.fit_transform(x_tr.reshape(-1, 1))
    
    # Singular Value Decomposition of Design Matrix X
    U, s, Vt = la.svd(X_tr_full, full_matrices=False)
    cond_number_X = float(s[0] / s[-1]) if s[-1] > 0 else np.inf
    cond_number_XtX = cond_number_X ** 2
    
    # 2. Bootstrap Resampling Simulation
    predictions = np.zeros((n_bootstraps, n_test))
    coef_matrix = np.zeros((n_bootstraps, degree + 1))
    
    x_tr_2d = x_tr.reshape(-1, 1)
    x_te_2d = x_te.reshape(-1, 1)
    
    for b in range(n_bootstraps):
        boot_idx = rng.choice(n_train, size=n_train, replace=True)
        x_boot = x_tr_2d[boot_idx]
        y_boot = y_tr[boot_idx]
        
        # Fit Polynomial OLS
        poly_b = PolynomialFeatures(degree=degree, include_bias=True)
        X_boot = poly_b.fit_transform(x_boot)
        X_te = poly_b.transform(x_te_2d)
        
        model = LinearRegression(fit_intercept=False)
        model.fit(X_boot, y_boot)
        
        predictions[b, :] = model.predict(X_te)
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
    
    # Expected coefficient L2 norm over bootstrap runs (averaging norms across runs to avoid oscillation cancellation)
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
# #### Baseline Polynomial Decomposition Output
# Displays error components, condition number, and identity residual for degree 3 polynomial regression.

# %%
print("Baseline Polynomial Decomposition (Degree d=3):")
print(f"  Integrated Bias^2:               {sample_res['bias_sq']:.6f}")
print(f"  Integrated Variance:             {sample_res['variance']:.6f}")
print(f"  Irreducible Noise (sigma^2):     {TRUE_NOISE_VARIANCE:.4f}")
print(f"  Decomposed Sum (Bias^2+Var+σ^2): {sample_res['decomposed_sum']:.6f}")
print(f"  Empirical Total MSE:             {sample_res['total_mse']:.6f}")
print(f"  Identity Residual:               {sample_res['identity_residual']:.6f}")
print(f"  Condition Number kappa(X):       {sample_res['cond_number_X']:.2e}")
print(f"  Expected Weight Norm E[||w||_2]: {sample_res['coef_l2_norm']:.4f}")

# %% [markdown]
# ### 4. Polynomial Degree Sweeps ($d = 1 \dots 15$)
#
# We sweep polynomial degree $d$ from $1$ (linear) up to $15$ (high-degree ill-conditioned expansion).
#
# Across degrees:
# - **Underfitting ($d=1, 2$)**: High Bias$^2$, low Variance, low condition number $\kappa(X) \sim 10^1$.
# - **Optimal ($d=3, 4, 5$)**: Minimum total risk, balanced Bias$^2$ and Variance, moderate condition number $\kappa(X) \sim 10^2 - 10^4$.
# - **Overfitting & Ill-Conditioning ($d \ge 8$)**: Bias$^2$ stabilizes, Variance explodes exponentially as $\kappa(X)$ exceeds $10^8$, leading to numerical noise amplification and coefficient norm explosion $\mathbb{E}[\|\hat{w}\|_2] \gg 10^4$.

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
# #### Complexity Sweep Data Table Output
# Displays top 5 rows of polynomial complexity metrics DataFrame.

# %%
display(df_poly.head(5))

# %% [markdown]
# ### 5. Visualizations & Diagnostic Plots
#
# We render three dedicated visualization diagnostics:
# 1. **Polynomial Curve Fits across Complexity Regimes** (Plotly 4-panel subplot for $d=1, 3, 7, 14$).
# 2. **Bias-Variance Tradeoff & Design Matrix Conditioning** (Plotly dual-axis plot of error components vs. $\log_{10} \kappa(X)$).
# 3. **SVD Spectral Decay & Coefficient Norm Explosion** (Plotly dual-axis plot of singular values $\sigma_k(X)$ and $\mathbb{E}[\|\hat{w}\|_2]$).

# %% [markdown]
# #### Visualization 1: Polynomial Curve Fits across Complexity Regimes
# Interactive Plotly 4-panel grid displaying bootstrap fit ensembles and expected predictions $\bar{f}(x)$ for degrees $d \in \{1, 3, 7, 14\}$ in a single isolated block.

# %%
selected_degrees = [1, 3, 7, 14]
degree_to_res = {res["degree"]: res for res in poly_results}

fig1 = sp.make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        f"<b>Degree d={d}</b><br><span style='font-size:11px; color:#555;'>Var={degree_to_res[d]['variance']:.3f} | κ(X)={degree_to_res[d]['cond_number_X']:.1e}</span>"
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
        
    # Scatter training data points
    fig1.add_trace(
        go.Scatter(
            x=x_train,
            y=y_train,
            mode="markers",
            marker=dict(color="#666666", size=5, opacity=0.7),
            name="Data D₀",
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
        text="<b>Polynomial Regression Fits across Complexity Regimes</b>",
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
        y=1.02,
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
# #### Visualization 2: Bias-Variance Tradeoff & Design Matrix Conditioning
# Interactive Plotly dual-axis chart showing Integrated Bias$^2$, Variance, and Total MSE (left axis) alongside Condition Number $\kappa(X)$ (right axis, log scale) vs. Polynomial Degree $d$ in a single isolated block.

# %%
fig2 = sp.make_subplots(specs=[[{"secondary_y": True}]])

# Primary Y-axis: Error Components
fig2.add_trace(
    go.Scatter(
        x=df_poly["degree"], y=df_poly["bias_sq"],
        mode="lines+markers", name="Bias²",
        line=dict(color="#d95f02", width=3), marker=dict(size=8)
    ),
    secondary_y=False
)

fig2.add_trace(
    go.Scatter(
        x=df_poly["degree"], y=df_poly["variance"],
        mode="lines+markers", name="Variance",
        line=dict(color="#7570b3", width=3, dash="dash"), marker=dict(symbol="square", size=8)
    ),
    secondary_y=False
)

fig2.add_trace(
    go.Scatter(
        x=df_poly["degree"], y=df_poly["total_mse"],
        mode="lines+markers", name="Empirical Total MSE",
        line=dict(color="#1b9e77", width=3), marker=dict(symbol="diamond", size=8)
    ),
    secondary_y=False
)

fig2.add_trace(
    go.Scatter(
        x=df_poly["degree"], y=[TRUE_NOISE_VARIANCE] * len(df_poly),
        mode="lines", name="Irreducible Noise (σ²=0.09)",
        line=dict(color="#666666", width=1.5, dash="dot")
    ),
    secondary_y=False
)

# Secondary Y-axis: Condition Number kappa(X) (Log Scale)
fig2.add_trace(
    go.Scatter(
        x=df_poly["degree"], y=df_poly["cond_number_X"],
        mode="lines+markers", name="Condition Number κ(X)",
        line=dict(color="#e7298a", width=2.5, dash="dot"), marker=dict(symbol="x", size=8)
    ),
    secondary_y=True
)

fig2.update_layout(
    title=dict(
        text="<b>Polynomial Bias-Variance Tradeoff vs. Design Matrix Condition Number κ(X)</b>",
        font=dict(size=15, family="Plus Jakarta Sans"),
        x=0.5,
        y=0.98,
        xanchor="center",
        yanchor="top"
    ),
    xaxis=dict(title="Polynomial Degree (d)", tickmode="linear", tick0=1, dtick=1),
    template="plotly_white",
    hovermode="x unified",
    margin=dict(l=60, r=60, t=80, b=60),
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)", bordercolor="#cccccc", borderwidth=1),
    height=520
)

fig2.update_yaxes(title_text="Integrated Error Magnitude (MSE)", secondary_y=False, range=[0, 0.45])
fig2.update_yaxes(title_text="Condition Number κ(X) [Log Scale]", type="log", secondary_y=True)
fig2.show()

# %% [markdown]
# #### Visualization 3: SVD Spectral Decay & Coefficient Norm Explosion
# Interactive Plotly chart displaying singular value spectrum $\sigma_k(X)$ for selected polynomial degrees $d \in \{1, 3, 7, 14\}$ alongside parameter vector norm $\mathbb{E}[\|\hat{w}\|_2]$ explosion in a single isolated block.

# %%
fig3 = sp.make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "<b>Singular Value Spectrum Decay</b><br><span style='font-size:11px; color:#555;'>σ_k(X) across polynomial degrees</span>",
        "<b>Expected Parameter Norm Explosion</b><br><span style='font-size:11px; color:#555;'>E[||ŵ||₂] vs. polynomial degree (d)</span>"
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
        hovertemplate="Degree %{x}: E[||ŵ||₂] = %{y:.2e}<extra></extra>"
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
        y=1.02,
        xanchor="center",
        x=0.25,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#e0e0e0",
        borderwidth=1
    )
)

fig3.for_each_annotation(lambda a: a.update(font=dict(size=12)))

fig3.update_xaxes(title_text="Singular Value Index (k)", row=1, col=1)
fig3.update_yaxes(title_text="Singular Value σ_k(X) [Log Scale]", type="log", row=1, col=1)

fig3.update_xaxes(title_text="Polynomial Degree (d)", tickmode="linear", dtick=1, row=1, col=2)
fig3.update_yaxes(title_text="Expected Norm E[||ŵ||₂] [Log Scale]", type="log", row=1, col=2)
fig3.show()

# %% [markdown]
# ### 6. Summary & Key Engineering Takeaways
#
# Below is the consolidated summary table comparing error components, condition numbers, and expected coefficient norms across representative polynomial degrees.
#
# #### Key Engineering Takeaways:
# 1. **Ill-Conditioning & Variance Explosion**: As degree $d$ increases past $5$, the design matrix condition number $\kappa(X)$ exceeds $10^5$. Near-collinearity of polynomial features $x^j$ inflates OLS parameter variance $\text{Var}(\hat{w}) = \sigma^2 (X^T X)^{-1}$, driving generalization variance upward.
# 2. **SVD Spectral Decay & Coefficient Norm Dynamics**: Trailing singular values collapse exponentially. In unconstrained OLS, parameter variance scales with $\sum \sigma_k^{-2}$, driving exponential growth in coefficient norms up to moderate degrees ($d \le 9$). Beyond this point, numerical rank-truncation in standard SVD solvers suppresses under-determined modes, capping the norm while severely destabilizing point predictions.
# 3. **Numerical Regularization Requirement**: High-degree unregularized polynomials overfit localized noise. Mitigating ill-conditioning requires L2 penalty regularization (Ridge regression $\lambda I$) or SVD rank truncation (pseudo-inverse thresholding).
# 4. **Finite-Precision Solver Truncation vs. Approximability**: Expanding nested hypothesis classes ($\mathcal{H}_1 \subset \mathcal{H}_2 \subset \dots \subset \mathcal{H}_{15}$) guarantees non-increasing approximation bias in theory. In practice, at $d \ge 10$, the condition number $\kappa(X) \sim 10^{11}$ pushes trailing singular values below the LAPACK `*gelsd` truncation threshold ($\text{rcond} \approx \epsilon_{\text{mach}}$). The solver discards these near-null singular modes, implicitly restricting the parameter space and producing an artificial surge in empirical $\text{Bias}^2$ due to algorithmic rank truncation rather than representational incapacity.

# %% [markdown]
# #### Final Summary Table Display
# Renders final empirical summary table and maximum identity residual across selected configurations in a single isolated block.

# %%
summary_degrees = [1, 3, 5, 9, 14]
df_summary = df_poly[df_poly["degree"].isin(summary_degrees)].copy()

df_summary_display = pd.DataFrame({
    "Degree (d)": df_summary["degree"],
    "Bias^2": df_summary["bias_sq"].map("{:.6f}".format),
    "Variance": df_summary["variance"].map("{:.6f}".format),
    "Noise (sigma^2)": df_summary["noise"].map("{:.4f}".format),
    "Decomposed Sum": df_summary["decomposed_sum"].map("{:.6f}".format),
    "Empirical Total MSE": df_summary["total_mse"].map("{:.6f}".format),
    "Identity Error": df_summary["identity_residual"].map("{:.6f}".format),
    "Condition Number κ(X)": df_summary["cond_number_X"].map("{:.2e}".format),
    "Expected Weight Norm E[||w||_2]": df_summary["coef_l2_norm"].map("{:.2e}".format)
})

display(Markdown("### Final Empirical Polynomial Error Decomposition & Conditioning Summary Table"))
display(df_summary_display)
display(Markdown(f"**Maximum Error Decomposition Identity Residual:** `{df_poly['identity_residual'].max():.6f}`"))
