# %% [markdown]
# # Module 03: Learning Curves, Empirical Risk Dynamics & Sample Complexity
#
# ## Foundational Theory & Mathematical Derivation
#
# In machine learning, the performance of an estimator as a function of training dataset size $N$ is governed by the interplay between **Empirical Risk Minimization (ERM)**, hypothesis class capacity, and statistical concentration.
#
# Given a training dataset $\mathcal{D}_N = \{(x_i, y_i)\}_{i=1}^N$ drawn i.i.d. from distribution $\mathcal{P}_{X,Y}$ with ground-truth regression relation $y = f(x) + \epsilon$, $\mathbb{E}[\epsilon]=0$, $\text{Var}(\epsilon)=\sigma^2$:
#
# 1. **Empirical Risk (Training Error)**:
#    $$
#    \hat{R}_N(\hat{f}_N) = \frac{1}{N} \sum_{i=1}^N L(y_i, \hat{f}_N(x_i))
#    $$
# 2. **True Risk (Generalization / Out-of-Sample Error)**:
#    $$
#    R(\hat{f}_N) = \mathbb{E}_{(X,Y) \sim \mathcal{P}}\left[ L(Y, \hat{f}_N(X)) \right]
#    $$
# 3. **Generalization Gap**:
#    $$
#    \Delta_N = R(\hat{f}_N) - \hat{R}_N(\hat{f}_N)
#    $$
#
# ---
#
# ### 1. Parametric Asymptotics: Why Training Error Converges from Below
#
# For Ordinary Least Squares (OLS) with $p$ linear parameters (or a fixed linear basis $\phi(x) \in \mathbb{R}^p$), the expected in-sample and out-of-sample mean squared errors under homoskedastic noise $\sigma^2$ follow exact finite-sample expansions:
#
# $$
# \mathbb{E}_{\mathcal{D}_N}\left[ \hat{R}_N(\hat{f}_N) \right] = \text{Bias}^2(\mathcal{H}) + \sigma^2 \left( 1 - \frac{p}{N} \right)
# $$
#
# $$
# \mathbb{E}_{\mathcal{D}_N}\left[ R(\hat{f}_N) \right] = \text{Bias}^2(\mathcal{H}) + \sigma^2 \left( 1 + \frac{p}{N} \right)
# $$
#
# Where:
# - $\text{Bias}^2(\mathcal{H}) = \min_{h \in \mathcal{H}} \mathbb{E}_X\left[ (f(X) - h(X))^2 \right]$ is the irreducible representation approximation error of hypothesis class $\mathcal{H}$.
# - The factor $-\frac{p}{N}\sigma^2$ in the empirical risk arises because the model fits the $p$ degrees of freedom partially to the specific noise realization $\epsilon_i$ in $\mathcal{D}_N$ (optimism of training error).
# - The factor $+\frac{p}{N}\sigma^2$ in the true risk accounts for parameter estimation variance $\text{Tr}(\text{Cov}(\hat{w})) \propto \frac{p \sigma^2}{N}$.
#
# Consequently, the expected generalization gap contracts inversely with sample size:
#
# $$
# \mathbb{E}_{\mathcal{D}_N}[\Delta_N] = \frac{2 p \sigma^2}{N} = \mathcal{O}\left( \frac{p}{N} \right)
# $$
#
# As $N \to \infty$, both curves asymptotically converge to the asymptotic risk floor $R_\infty = \text{Bias}^2(\mathcal{H}) + \sigma^2$.
#
# ---
#
# ### 2. Sample Complexity Bounds
#
# In statistical learning theory, the **Sample Complexity** $N(\epsilon, \delta)$ defines the minimum number of training examples required to guarantee that the generalization error is within $\epsilon$ of the optimal hypothesis in $\mathcal{H}$ with probability at least $1 - \delta$:
#
# $$
# N(\epsilon, \delta) \ge \mathcal{O}\left( \frac{d_{\text{VC}}(\mathcal{H}) + \log(1/\delta)}{\epsilon^2} \right)
# $$
#
# For parametric linear models with VC dimension $d_{\text{VC}} = p + 1$, achieving excess estimation risk $\mathbb{E}[R] - R_\infty \le \epsilon$ requires:
#
# $$
# N \ge \frac{p \sigma^2}{\epsilon} \implies N \propto \frac{p}{\epsilon}
# $$
#
# ### System Architecture & Learning Curve Simulation Pipeline
#
# The workflow below illustrates the Monte Carlo subsampling, error evaluation, and power-law scaling analysis pipeline.
#
# ```mermaid
# graph TD
#     A["Data Generating Process f(x) + Gaussian Noise sigma^2"] --> B["Full Evaluation Grid X_val (N=2000)"]
#     A --> C["Sample Size Grid N in [12, ..., 1000]"]
#     C --> D["Monte Carlo Subsampling (K=100 Replicates per N)"]
#     D --> E["Model Fitting: Underfit (d=1), Optimal (d=3), Overfit (d=9)"]
#     E --> F["Compute Empirical Training Risk R_hat(N)"]
#     E --> G["Compute Out-of-Sample True Risk R(N) on X_val"]
#     E --> H["Pointwise Bias^2(N) and Variance(N) Decomposition"]
#     F & G --> I["Generalization Gap Delta(N) = R(N) - R_hat(N)"]
#     I --> J["Power-Law Log-Log Scaling Fit: Delta(N) ~ N^-beta"]
#     H --> K["Variance Extinction vs. Bias Invariance Diagnostics"]
# ```

# %% [markdown]
# ### 1. Imports & Environment Setup
#
# We import high-performance numerical compute libraries (`numpy`, `pandas`, `scipy.optimize`), scikit-learn models and preprocessing transforms, type hints, and plotting modules (`plotly`). We configure deterministic random seeds and consistent visual themes.

# %%
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
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
from sklearn.pipeline import make_pipeline

SEED: int = 42
np.random.seed(SEED)

# %% [markdown]
# #### Environment Initialization Output
# Displays system status and global random seed configuration.

# %%
print(f"Module 03 environment initialized successfully. Random Seed = {SEED}")

# %% [markdown]
# ### 2. Synthetic Ground Truth & Data Generating Process
#
# We construct a non-linear target function $f(x)$ with damped harmonic oscillation over $x \in [0, 2.5]$:
#
# $$
# f(x) = \sin(1.2 \pi x) \cdot \exp(-0.4 x) + 0.3 x
# $$
#
# Additive Gaussian noise is drawn from $\epsilon \sim \mathcal{N}(0, \sigma^2)$ with $\sigma = 0.30$, yielding irreducible noise floor $\sigma^2 = 0.09$.
#
# We generate a dense out-of-sample validation set $\mathcal{D}_{\text{val}}$ of size $N_{\text{val}} = 2000$ points to compute out-of-sample risk $R(h)$ with high numerical precision.

# %%
def true_function(x: np.ndarray) -> np.ndarray:
    """Ground truth target function f(x)."""
    return np.sin(1.2 * np.pi * x) * np.exp(-0.4 * x) + 0.3 * x

def generate_samples(
    n_samples: int,
    noise_std: float = 0.30,
    rng: np.random.RandomState = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates synthetic dataset (x, y, y_true) with additive Gaussian noise."""
    if rng is None:
        rng = np.random.RandomState()
    x = rng.uniform(0.0, 2.5, size=n_samples)
    y_true = true_function(x)
    noise = rng.normal(0.0, noise_std, size=n_samples)
    y = y_true + noise
    return x, y, y_true

NOISE_STD: float = 0.30
TRUE_NOISE_VARIANCE: float = NOISE_STD ** 2  # sigma^2 = 0.09

# Dense validation set for exact true risk computation
rng_val = np.random.RandomState(SEED + 999)
x_val, y_val, y_val_true = generate_samples(n_samples=2000, noise_std=NOISE_STD, rng=rng_val)
x_val_grid = np.linspace(0.0, 2.5, 400)
y_val_grid_true = true_function(x_val_grid)

# %% [markdown]
# #### Ground Truth Specification Output
# Displays test set size, noise level, and mathematical properties of the synthetic data generator.

# %%
print("Synthetic Data Generating Process:")
print(f"  Target Domain:            x in [0.0, 2.5]")
print(f"  Validation Set Size:      N_val = {len(x_val)}")
print(f"  Irreducible Noise Std:    sigma = {NOISE_STD:.2f}")
print(f"  Bayes Noise Floor sigma^2: {TRUE_NOISE_VARIANCE:.4f}")

# %% [markdown]
# ### 3. Vectorized Monte Carlo Learning Curve Engine
#
# To empirically estimate the expected learning curves $\mathbb{E}[\hat{R}_N]$ and $\mathbb{E}[R_N]$ across sample sizes $N$, we run a vectorized Monte Carlo simulation with $K = 100$ independent trials for each $N_k$.
#
# For each trial $k \in \{1, \dots, K\}$:
# 1. Sample independent training dataset $\mathcal{D}_N^{(k)} = \{(x_i, y_i)\}_{i=1}^N$.
# 2. Fit the polynomial regression model $\hat{f}_N^{(k)}$.
# 3. Calculate empirical in-sample training error:
#    $$
#    \hat{R}_N^{(k)} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{f}_N^{(k)}(x_i))^2
#    $$
# 4. Calculate out-of-sample validation error against noisy test realizations:
#    $$
#    R_N^{(k)} = \frac{1}{N_{\text{val}}} \sum_{j=1}^{N_{\text{val}}} (y_{\text{val}, j} - \hat{f}_N^{(k)}(x_{\text{val}, j}))^2
#    $$
# 5. Record predictions on validation grid to decompose pointwise $\text{Bias}^2(x) = (\mathbb{E}[\hat{f}(x)] - f(x))^2$ and $\text{Var}(x) = \text{Var}(\hat{f}(x))$.

# %%
def simulate_learning_curve_for_model(
    model_name: str,
    degree: int,
    sample_sizes: np.ndarray,
    n_replicates: int = 100,
    noise_std: float = 0.30,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes Monte Carlo empirical learning curves (Train MSE, Val MSE, Bias^2, Variance)
    across varying training sample sizes N.
    """
    rng = np.random.RandomState(seed)
    n_points_val = len(x_val)
    
    results = {
        "model_name": model_name,
        "degree": degree,
        "sample_sizes": sample_sizes,
        "train_mse_mean": [],
        "train_mse_std": [],
        "train_mse_ci95": [],
        "val_mse_mean": [],
        "val_mse_std": [],
        "val_mse_ci95": [],
        "gen_gap_mean": [],
        "bias_sq_mean": [],
        "variance_mean": [],
        "decomposed_sum": []
    }
    
    for n in sample_sizes:
        train_mses = np.zeros(n_replicates)
        val_mses = np.zeros(n_replicates)
        val_preds = np.zeros((n_replicates, len(x_val_grid)))
        
        for rep in range(n_replicates):
            # Generate independent training sample of size n
            x_tr, y_tr, _ = generate_samples(n_samples=n, noise_std=noise_std, rng=rng)
            
            # Fit polynomial OLS
            poly = PolynomialFeatures(degree=degree, include_bias=True)
            X_tr = poly.fit_transform(x_tr.reshape(-1, 1))
            X_val_full = poly.transform(x_val.reshape(-1, 1))
            X_grid = poly.transform(x_val_grid.reshape(-1, 1))
            
            model = LinearRegression(fit_intercept=False)
            model.fit(X_tr, y_tr)
            
            # In-sample and out-of-sample predictions
            y_tr_pred = model.predict(X_tr)
            y_val_pred = model.predict(X_val_full)
            grid_pred = model.predict(X_grid)
            
            train_mses[rep] = np.mean((y_tr - y_tr_pred) ** 2)
            val_mses[rep] = np.mean((y_val - y_val_pred) ** 2)
            val_preds[rep, :] = grid_pred
            
        # Summary statistics across Monte Carlo replicates
        mean_tr = float(np.mean(train_mses))
        std_tr = float(np.std(train_mses, ddof=1))
        ci95_tr = 1.96 * std_tr / np.sqrt(n_replicates)
        
        mean_val = float(np.mean(val_mses))
        std_val = float(np.std(val_mses, ddof=1))
        ci95_val = 1.96 * std_val / np.sqrt(n_replicates)
        
        # Bias-Variance decomposition on dense grid
        expected_pred = np.mean(val_preds, axis=0)
        pointwise_bias_sq = (expected_pred - y_val_grid_true) ** 2
        pointwise_var = np.var(val_preds, axis=0, ddof=0)
        
        bias_sq_scalar = float(np.mean(pointwise_bias_sq))
        var_scalar = float(np.mean(pointwise_var))
        
        results["train_mse_mean"].append(mean_tr)
        results["train_mse_std"].append(std_tr)
        results["train_mse_ci95"].append(ci95_tr)
        
        results["val_mse_mean"].append(mean_val)
        results["val_mse_std"].append(std_val)
        results["val_mse_ci95"].append(ci95_val)
        
        results["gen_gap_mean"].append(mean_val - mean_tr)
        results["bias_sq_mean"].append(bias_sq_scalar)
        results["variance_mean"].append(var_scalar)
        results["decomposed_sum"].append(bias_sq_scalar + var_scalar + (noise_std ** 2))
        
    for k in results:
        if isinstance(results[k], list):
            results[k] = np.array(results[k])
            
    return results

# %% [markdown]
# #### Engine Sanity Check Output
# Displays execution validation of the Monte Carlo simulation engine.

# %%
print("Learning curve simulation engine defined successfully.")

# %% [markdown]
# ### 4. Multi-Regime Sample Complexity Sweeps
#
# We evaluate three distinct model capacity regimes across sample sizes $N \in [12, 20, 32, 50, 80, 130, 220, 380, 650, 1000]$:
#
# 1. **Underfitted Model ($d=1$, $p=2$)**:
#    - High structural Bias$^2$.
#    - Small parameter count $p=2 \implies$ fast convergence ($\frac{p}{N} \to 0$ rapidly).
#    - Generalization gap closes at very small $N \sim 30$, but asymptotic risk plateau remains high ($R_\infty \gg \sigma^2$).
# 2. **Optimal Capacity Model ($d=3$, $p=4$)**:
#    - Minimal Bias$^2$.
#    - Balanced complexity $p=4 \implies$ moderate convergence rate.
#    - Rapidly approaches the true Bayes error rate $\sigma^2 = 0.09$ by $N \sim 100$.
# 3. **Overfitted Model ($d=9$, $p=10$)**:
#    - Negligible Bias$^2$.
#    - High parameter count $p=10 \implies$ massive generalization gap at small $N < 50$.
#    - Requires substantial sample size ($N > 400$) to suppress variance $\frac{p \sigma^2}{N}$ down to acceptable levels.

# %%
SAMPLE_SIZES = np.array([12, 20, 32, 50, 80, 130, 220, 380, 650, 1000])

regime_models = [
    ("Underfitted (d=1, p=2)", 1),
    ("Optimal (d=3, p=4)", 3),
    ("Overfitted (d=9, p=10)", 9)
]

# Run parallel Monte Carlo simulations
sweep_results = Parallel(n_jobs=-1)(
    delayed(simulate_learning_curve_for_model)(
        model_name=name,
        degree=deg,
        sample_sizes=SAMPLE_SIZES,
        n_replicates=100,
        noise_std=NOISE_STD,
        seed=SEED + idx * 101
    )
    for idx, (name, deg) in enumerate(regime_models)
)

results_by_name = {res["model_name"]: res for res in sweep_results}

# Assemble consolidated metrics table at selected sample sizes
records = []
for res in sweep_results:
    for i, n in enumerate(SAMPLE_SIZES):
        records.append({
            "Model": res["model_name"],
            "N": n,
            "Train MSE": res["train_mse_mean"][i],
            "Val MSE": res["val_mse_mean"][i],
            "Gen Gap": res["gen_gap_mean"][i],
            "Bias^2": res["bias_sq_mean"][i],
            "Variance": res["variance_mean"][i],
            "Decomposed Sum": res["decomposed_sum"][i]
        })

df_learning_curves = pd.DataFrame(records)

# %% [markdown]
# #### Sample Complexity Multi-Regime Summary Table Output
# Displays representative rows comparing error metrics at small, medium, and large sample sizes.

# %%
display(df_learning_curves[df_learning_curves["N"].isin([12, 50, 220, 1000])].head(12))

# %% [markdown]
# ### 5. Visualizations & Diagnostic Plots
#
# We generate three comprehensive Plotly diagnostic charts:
# 1. **Dual Learning Curves Across Complexity Regimes** (3-panel subplot comparing Train vs. Val MSE with shaded 95% CI bands and theoretical asymptotes).
# 2. **Generalization Gap Dynamics & Power-Law Scaling** (Linear gap decay and Log-Log power-law exponent verification $\Delta_N \propto N^{-\beta}$).
# 3. **Sample Size Effect on Bias vs. Variance Decomposition** (Tracking monotonic variance collapse against bias invariance).

# %% [markdown]
# #### Visualization 1: Dual Learning Curves Across Complexity Regimes
# Interactive Plotly 3-panel chart showing Training MSE, Validation MSE, 95% Confidence Intervals, Asymptotic Risk Plateau ($R_\infty$), and Bayes Irreducible Noise ($\sigma^2=0.09$) in a single isolated block.

# %%
fig1 = sp.make_subplots(
    rows=1, cols=3,
    subplot_titles=[
        f"<b>{name}</b><br><span style='font-size:11px; color:#555;'>Bias²={results_by_name[name]['bias_sq_mean'][-1]:.3f} | Asymptote={results_by_name[name]['bias_sq_mean'][-1] + TRUE_NOISE_VARIANCE:.3f}</span>"
        for name, _ in regime_models
    ],
    shared_yaxes=True,
    horizontal_spacing=0.06
)

colors_tr = "#2b5c8f"
colors_val = "#d95f02"

for col_idx, (name, deg) in enumerate(regime_models, start=1):
    res = results_by_name[name]
    n_pts = res["sample_sizes"]
    
    # 95% CI shaded band for Validation MSE
    val_upper = res["val_mse_mean"] + res["val_mse_ci95"]
    val_lower = res["val_mse_mean"] - res["val_mse_ci95"]
    
    fig1.add_trace(
        go.Scatter(
            x=np.concatenate([n_pts, n_pts[::-1]]),
            y=np.concatenate([val_upper, val_lower[::-1]]),
            fill="toself",
            fillcolor="rgba(217, 95, 2, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
            name="Val 95% CI"
        ),
        row=1, col=col_idx
    )
    
    # 95% CI shaded band for Training MSE
    tr_upper = res["train_mse_mean"] + res["train_mse_ci95"]
    tr_lower = res["train_mse_mean"] - res["train_mse_ci95"]
    
    fig1.add_trace(
        go.Scatter(
            x=np.concatenate([n_pts, n_pts[::-1]]),
            y=np.concatenate([tr_upper, tr_lower[::-1]]),
            fill="toself",
            fillcolor="rgba(43, 92, 143, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
            name="Train 95% CI"
        ),
        row=1, col=col_idx
    )
    
    # Validation MSE Curve (Approaches from ABOVE)
    fig1.add_trace(
        go.Scatter(
            x=n_pts, y=res["val_mse_mean"],
            mode="lines+markers",
            name="Validation MSE (Generalization)",
            line=dict(color=colors_val, width=2.5),
            marker=dict(size=6, symbol="diamond"),
            showlegend=(col_idx == 1),
            hovertemplate="Val MSE: %{y:.4f} (N=%{x})<extra></extra>"
        ),
        row=1, col=col_idx
    )
    
    # Training MSE Curve (Approaches from BELOW)
    fig1.add_trace(
        go.Scatter(
            x=n_pts, y=res["train_mse_mean"],
            mode="lines+markers",
            name="Training MSE (Empirical Risk)",
            line=dict(color=colors_tr, width=2.5, dash="dash"),
            marker=dict(size=6, symbol="circle"),
            showlegend=(col_idx == 1),
            hovertemplate="Train MSE: %{y:.4f} (N=%{x})<extra></extra>"
        ),
        row=1, col=col_idx
    )
    
    # Asymptotic Risk Plateau (Bias^2 + sigma^2)
    asymptote_val = res["bias_sq_mean"][-1] + TRUE_NOISE_VARIANCE
    fig1.add_trace(
        go.Scatter(
            x=[n_pts[0], n_pts[-1]], y=[asymptote_val, asymptote_val],
            mode="lines",
            name="Asymptotic Risk (Bias² + σ²)",
            line=dict(color="#1b9e77", width=1.8, dash="dot"),
            showlegend=(col_idx == 1),
            hovertemplate=f"Asymptote: {asymptote_val:.4f}<extra></extra>"
        ),
        row=1, col=col_idx
    )
    
    # Bayes Irreducible Noise Floor (sigma^2 = 0.09)
    fig1.add_trace(
        go.Scatter(
            x=[n_pts[0], n_pts[-1]], y=[TRUE_NOISE_VARIANCE, TRUE_NOISE_VARIANCE],
            mode="lines",
            name="Bayes Noise Floor (σ²=0.09)",
            line=dict(color="#666666", width=1.5, dash="dashdot"),
            showlegend=(col_idx == 1),
            hovertemplate=f"Bayes Floor: {TRUE_NOISE_VARIANCE:.4f}<extra></extra>"
        ),
        row=1, col=col_idx
    )

fig1.update_layout(
    title=dict(
        text="<b>Empirical vs. True Risk Learning Curves Across Complexity Regimes</b>",
        font=dict(size=15, family="Plus Jakarta Sans"),
        x=0.5, y=0.98, xanchor="center", yanchor="top"
    ),
    template="plotly_white",
    height=540,
    margin=dict(l=60, r=40, t=110, b=60),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.04,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#e0e0e0",
        borderwidth=1
    )
)

fig1.for_each_annotation(lambda a: a.update(font=dict(size=12)))

for c in range(1, 4):
    fig1.update_xaxes(title_text="Training Sample Size (N)", type="log", row=1, col=c)

fig1.update_yaxes(title_text="Mean Squared Error (MSE)", range=[0.0, 0.45], row=1, col=1)
fig1.show()

# %% [markdown]
# #### Visualization 2: Generalization Gap Dynamics & Power-Law Scaling
# Interactive Plotly 2-panel chart demonstrating:
# 1. Linear Generalization Gap $\Delta(N) = R(N) - \hat{R}(N)$ contraction across sample sizes.
# 2. Log-Log scaling plot verifying the theoretical rate $\Delta(N) \propto N^{-\beta}$ with empirical regression fitting.

# %%
fig2 = sp.make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "<b>Generalization Gap Contraction Δ(N)</b><br><span style='font-size:11px; color:#555;'>Linear Scale: Δ(N) = Val MSE - Train MSE</span>",
        "<b>Log-Log Power-Law Scaling Law</b><br><span style='font-size:11px; color:#555;'>Empirical Slope -β vs. Theoretical -1.0</span>"
    ),
    horizontal_spacing=0.12
)

regime_colors = {
    "Underfitted (d=1, p=2)": "#2b5c8f",
    "Optimal (d=3, p=4)": "#1b9e77",
    "Overfitted (d=9, p=10)": "#e7298a"
}

# Power-law fitting function: log(Delta) = -beta * log(N) + log(alpha)
def power_law(log_n: np.ndarray, beta: float, log_alpha: float) -> np.ndarray:
    return -beta * log_n + log_alpha

for name, deg in regime_models:
    res = results_by_name[name]
    n_pts = res["sample_sizes"]
    gap = res["gen_gap_mean"]
    
    # Filter positive gap values for log-log regression
    valid_mask = (gap > 1e-4) & (n_pts >= 20)
    log_n_valid = np.log10(n_pts[valid_mask])
    log_gap_valid = np.log10(gap[valid_mask])
    
    popt, _ = curve_fit(power_law, log_n_valid, log_gap_valid, p0=[1.0, 0.0])
    fitted_beta = popt[0]
    
    # Panel 1: Linear Generalization Gap
    fig2.add_trace(
        go.Scatter(
            x=n_pts, y=gap,
            mode="lines+markers",
            name=f"{name}",
            line=dict(color=regime_colors[name], width=2.5),
            marker=dict(size=7),
            hovertemplate=f"{name}<br>N=%{{x}}: Δ(N)=%{{y:.4f}}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Panel 2: Log-Log Scaling Plot with Fitted Slopes
    fig2.add_trace(
        go.Scatter(
            x=n_pts, y=gap,
            mode="markers",
            name=f"{name} (Observed)",
            marker=dict(color=regime_colors[name], size=8, symbol="circle"),
            showlegend=False,
            hovertemplate=f"Observed: N=%{{x}}, Gap=%{{y:.4f}}<extra></extra>"
        ),
        row=1, col=2
    )
    
    # Fitted power-law line
    n_dense = np.logspace(np.log10(15), np.log10(1000), 50)
    gap_fitted = 10 ** power_law(np.log10(n_dense), popt[0], popt[1])
    
    fig2.add_trace(
        go.Scatter(
            x=n_dense, y=gap_fitted,
            mode="lines",
            name=f"{name} (Fit: β={fitted_beta:.2f})",
            line=dict(color=regime_colors[name], width=1.8, dash="dash"),
            hovertemplate=f"Power Law Fit: Gap ∝ N^(-{fitted_beta:.2f})<extra></extra>"
        ),
        row=1, col=2
    )

fig2.update_layout(
    title=dict(
        text="<b>Generalization Gap Dynamics & Asymptotic 1/N Scaling Verification</b>",
        font=dict(size=15, family="Plus Jakarta Sans"),
        x=0.5, y=0.98, xanchor="center", yanchor="top"
    ),
    template="plotly_white",
    height=480,
    margin=dict(l=60, r=40, t=100, b=60),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.04,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#e0e0e0",
        borderwidth=1
    )
)

fig2.for_each_annotation(lambda a: a.update(font=dict(size=12)))

fig2.update_xaxes(title_text="Training Sample Size (N)", row=1, col=1)
fig2.update_yaxes(title_text="Generalization Gap Δ(N)", row=1, col=1)

fig2.update_xaxes(title_text="Training Sample Size (N) [Log Scale]", type="log", row=1, col=2)
fig2.update_yaxes(title_text="Generalization Gap Δ(N) [Log Scale]", type="log", row=1, col=2)
fig2.show()

# %% [markdown]
# #### Visualization 3: Sample Size Impact on Bias vs. Variance Decomposition
# Interactive Plotly chart showing how expanding sample size $N$ from $12$ to $1000$ monotonically extinguishes estimation Variance while structural $\text{Bias}^2$ remains invariant in a single isolated block.

# %%
res_overfit = results_by_name["Overfitted (d=9, p=10)"]
n_pts = res_overfit["sample_sizes"]

fig3 = go.Figure()

# Bias^2 Component (Invariant with respect to N)
fig3.add_trace(
    go.Scatter(
        x=n_pts, y=res_overfit["bias_sq_mean"],
        mode="lines+markers",
        name="Bias² (Model Representation Floor)",
        line=dict(color="#d95f02", width=3),
        marker=dict(symbol="diamond", size=8),
        hovertemplate="Bias²: %{y:.4f} (Invariant)<extra></extra>"
    )
)

# Variance Component (Monotonically extinguished by N)
fig3.add_trace(
    go.Scatter(
        x=n_pts, y=res_overfit["variance_mean"],
        mode="lines+markers",
        name="Variance (Parameter Estimation Dispersion)",
        line=dict(color="#7570b3", width=3, dash="dash"),
        marker=dict(symbol="square", size=8),
        hovertemplate="Variance: %{y:.4f} (∝ p/N)<extra></extra>"
    )
)

# Irreducible Noise Floor (sigma^2 = 0.09)
fig3.add_trace(
    go.Scatter(
        x=[n_pts[0], n_pts[-1]], y=[TRUE_NOISE_VARIANCE, TRUE_NOISE_VARIANCE],
        mode="lines",
        name="Irreducible Noise Floor (σ²=0.09)",
        line=dict(color="#666666", width=2, dash="dot"),
        hovertemplate="Noise Floor: 0.0900<extra></extra>"
    )
)

# Decomposed Sum (Total Expected Out-of-Sample Risk)
fig3.add_trace(
    go.Scatter(
        x=n_pts, y=res_overfit["decomposed_sum"],
        mode="lines+markers",
        name="Total Decomposed Risk (Bias² + Var + σ²)",
        line=dict(color="#1b9e77", width=3.5),
        marker=dict(symbol="circle", size=9),
        hovertemplate="Total Risk: %{y:.4f}<extra></extra>"
    )
)

fig3.update_layout(
    title=dict(
        text="<b>Sample Size Scaling Effect on Bias-Variance Decomposition (Degree d=9)</b>",
        font=dict(size=15, family="Plus Jakarta Sans"),
        x=0.5, y=0.98, xanchor="center", yanchor="top"
    ),
    xaxis=dict(title="Training Sample Size (N) [Log Scale]", type="log"),
    yaxis=dict(title="Error Component Magnitude (MSE)", range=[0.0, 0.40]),
    template="plotly_white",
    hovermode="x unified",
    height=480,
    margin=dict(l=60, r=40, t=90, b=60),
    legend=dict(
        x=0.55, y=0.96,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#e0e0e0",
        borderwidth=1
    )
)

fig3.show()

# %% [markdown]
# ### 6. Summary & Key Engineering Takeaways
#
# Below is the consolidated summary table comparing sample complexity milestones and empirical power-law convergence metrics across the three capacity regimes.
#
# #### Key Engineering Takeaways:
# 1. **Directionality of Empirical vs. True Risk Convergence**:
#    - Training error approaches the asymptotic risk floor $R_\infty = \text{Bias}^2 + \sigma^2$ from **below** ($\approx R_\infty - \frac{p \sigma^2}{N}$), while validation error approaches from **above** ($\approx R_\infty + \frac{p \sigma^2}{N}$).
#    - At small sample sizes ($N \approx p$), training error drastically underestimates true risk due to noise memorization.
# 2. **The $N/p$ Rule of Thumb**:
#    - In unregularized linear models, parameter variance scales as $\frac{p \sigma^2}{N}$.
#    - To reduce the excess generalization error $\Delta_N$ below $10\%$ of the irreducible noise floor $\sigma^2$, the required sample size must satisfy $\frac{N}{p} \ge 20$.
# 3. **Invariance of Representation Bias under Sample Growth**:
#    - Increasing training dataset size $N \to \infty$ monotonically extinguishes estimation variance ($\text{Var} \to 0$).
#    - However, sample size growth has **zero effect on representation bias** ($\text{Bias}^2(N) = \text{const}$). An under-parameterized model ($d=1$) remains bounded by its structural bias plateau regardless of how many millions of samples are provided.
# 4. **Data Acquisition ROI vs. Model Capacity Decisions**:
#    - High-capacity models ($d=9$) exhibit poor small-sample performance ($N < 50$), but boast the highest rate of return on additional training data, converging to the lowest Bayes risk floor once $N > 400$.

# %% [markdown]
# #### Final Summary Table Display
# Renders final empirical summary table and critical sample complexity milestones in a single isolated block.

# %%
summary_records = []

for name, deg in regime_models:
    res = results_by_name[name]
    n_pts = res["sample_sizes"]
    val_mse = res["val_mse_mean"]
    asymptote = res["bias_sq_mean"][-1] + TRUE_NOISE_VARIANCE
    
    # Critical sample size to reach within 10% and 5% of asymptotic risk
    n_crit_10 = n_pts[np.where(val_mse <= asymptote * 1.10)[0][0]] if np.any(val_mse <= asymptote * 1.10) else "> 1000"
    n_crit_05 = n_pts[np.where(val_mse <= asymptote * 1.05)[0][0]] if np.any(val_mse <= asymptote * 1.05) else "> 1000"
    
    summary_records.append({
        "Model Architecture": name,
        "Parameters (p)": deg + 1,
        "Asymptotic Bias^2": f"{res['bias_sq_mean'][-1]:.4f}",
        "Asymptotic Risk (R_inf)": f"{asymptote:.4f}",
        "Gap at N=12": f"{res['gen_gap_mean'][0]:.4f}",
        "Gap at N=1000": f"{res['gen_gap_mean'][-1]:.4f}",
        "N_crit (<= 10% Excess)": str(n_crit_10),
        "N_crit (<= 5% Excess)": str(n_crit_05)
    })

df_summary_display = pd.DataFrame(summary_records)

display(Markdown("### Final Learning Curve & Sample Complexity Milestone Summary"))
display(df_summary_display)
display(Markdown(f"**Bayes Irreducible Noise Floor (σ²):** `{TRUE_NOISE_VARIANCE:.4f}`"))
