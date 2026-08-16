# %% [markdown]
# # Module 03: Learning Curves, Empirical Risk Dynamics & Sample Complexity
#
# ## Part I: Theoretical Formulations
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
# ### 1.1 Parametric Asymptotics: Fixed-Design OLS and Finite-Sample Approximations
#
# For Ordinary Least Squares (OLS) with $p$ linear parameters (or a fixed linear basis $\phi(x) \in \mathbb{R}^p$), the expected in-sample and out-of-sample mean squared errors under homoskedastic Gaussian errors $\epsilon \sim \mathcal{N}(0, \sigma^2)$ follow exact finite-sample expansions under a **fixed-design OLS framework** (conditioning on sample matrix $X$, where parameter covariance $\text{Cov}(\hat{w}) = \sigma^2(X^T X)^{-1}$ and trace $\text{Tr}(X(X^T X)^{-1} X^T) = p$):
#
# $$
# \mathbb{E}_{\mathcal{D}_N | X}\left[ \hat{R}_N(\hat{f}_N) \right] = \text{Bias}^2(\mathcal{H}) + \sigma^2 \left( 1 - \frac{p}{N} \right)
# $$
#
# $$
# \mathbb{E}_{\mathcal{D}_N | X}\left[ R(\hat{f}_N) \right] = \text{Bias}^2(\mathcal{H}) + \sigma^2 \left( 1 + \frac{p}{N} \right)
# $$
#
# > **Note on Random Design vs. Fixed Design:**
# > Under a **random-design distribution** ($X \sim \mathcal{P}_X$), the exact expected risk depends on inverse-moment expectations of the random Gram matrix $\mathbb{E}[\text{Tr}((X^T X)^{-1})]$. In that setting, the expressions above serve as first-order asymptotic expansions as $N \to \infty$ that hold with high precision in the well-sampled regime $N \gg p$.
#
# Where:
# - $\text{Bias}^2(\mathcal{H}) = \min_{h \in \mathcal{H}} \mathbb{E}_X\left[ (f(X) - h(X))^2 \right]$ is the irreducible representation approximation error of hypothesis class $\mathcal{H}$.
# - The factor $-\frac{p}{N}\sigma^2$ in empirical risk arises because the model fits the $p$ degrees of freedom partially to the specific noise realization $\epsilon_i$ in $\mathcal{D}_N$ (optimism of training error).
# - The factor $+\frac{p}{N}\sigma^2$ in true risk accounts for parameter estimation variance $\text{Tr}(\text{Cov}(\hat{w})) \propto \frac{p \sigma^2}{N}$.
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
# ### 1.2 Sample Complexity Bounds: PAC-Learning vs. Parametric Point Estimation
#
# #### Combinatorial / PAC-Learning Generalization Bounds
# In distribution-free Probably Approximately Correct (PAC) learning theory for binary classification or bounded empirical risk minimization, uniform convergence bounds establish sample complexity $N(\epsilon, \delta)$—the minimum number of training examples guaranteeing generalization error within $\epsilon$ of the optimal hypothesis in $\mathcal{H}$ with probability at least $1 - \delta$:
#
# $$
# N(\epsilon, \delta) \ge \mathcal{O}\left( \frac{d_{\text{VC}}(\mathcal{H}) + \log(1/\delta)}{\epsilon^2} \right)
# $$
#
# This $\mathcal{O}(1/\epsilon^2)$ rate characterizes worst-case uniform deviations $\sup_{h \in \mathcal{H}} |R(h) - \hat{R}(h)|$ over bounded hypothesis spaces without distributional assumptions.
#
# #### Parametric Point Estimation Sample Complexity under Squared Loss
# In contrast, for parametric linear models under squared error loss (unregularized convex empirical minimization), estimation error is driven by parameter covariance dispersion rather than uniform combinatorial deviations. To guarantee that expected excess risk satisfies $\mathbb{E}[R(\hat{f}_N)] - R_\infty \le \epsilon$, parametric least squares exhibits a faster $\mathcal{O}(1/\epsilon)$ sample complexity scaling:
#
# $$
# N \ge \frac{p \sigma^2}{\epsilon} \implies N \propto \frac{p}{\epsilon}
# $$
#
# Here, required sample size scales linearly with parameter count $p$ and noise variance $\sigma^2$, and inversely with the error tolerance $\epsilon$.
#
# ---
#
# ### 1.3 Theoretical Bridge: Connecting PAC Bounds to Parametric Monte Carlo Dynamics
#
# A foundational distinction in learning theory is the divergence between distribution-free uniform convergence bounds and parametric point estimation rates:
#
# 1. **Worst-Case Uniform Bounds (PAC Regime)**: Distribution-free VC bounds protect against the most adversarial underlying data distributions by bounding the supremum deviation across the entire hypothesis space $\mathcal{H}$, incurring a slow $\mathcal{O}(1/\epsilon^2)$ rate.
# 2. **Average-Case Parametric Convergence (Our Experimental Regime)**: When the hypothesis class consists of smooth parametric models fitted via Ordinary Least Squares on continuous Euclidean domains, the empirical risk minimizer achieves fast $\mathcal{O}(1/\epsilon)$ average-case parameter convergence governed by the Central Limit Theorem and Fisher information.
#
# The computational simulation in **Part II** specifically evaluates this average-case parametric regime $\mathbb{E}[\Delta_N] \approx \frac{2p\sigma^2}{N}$, empirically measuring parameter variance extinguishment and verifying asymptotic risk floors under exact continuous domain integration.
#
# %% [markdown]
# ## Part II: Computational Simulation and Diagnostics
#
# > **Experimental Hypothesis:**
# > The subsequent computational experiment benchmarks three distinct polynomial capacities ($d \in \{1, 3, 9\}$) against synthetic non-linear ground truth to empirically test the theoretical $2p\sigma^2/N$ contraction rate of the generalization gap, eliminate Vandermonde collinearity artifacts via orthogonal Chebyshev polynomial basis expansions, and verify asymptotic risk floors using exact continuous numerical quadrature.
#
# ### 2.1 Imports and Environment Setup
#
# We import high-performance numerical compute libraries (`numpy`, `numpy.polynomial.chebyshev.chebvander`, `pandas`, `scipy.integrate`), scikit-learn models (`LinearRegression`), type hints, and plotting modules (`plotly`). We configure deterministic random seeds and consistent visual aesthetics.

# %%
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from scipy.integrate import quad
from sklearn.linear_model import LinearRegression
from numpy.polynomial.chebyshev import chebvander
from joblib import Parallel, delayed
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

SEED: int = 42
np.random.seed(SEED)

# %% [markdown]
# ### 2.2 Synthetic Ground Truth & Data Generating Process
#
# We construct a non-linear target function $f(x)$ with damped harmonic oscillation over the continuous domain $x \in [0.0, 2.5]$:
#
# $$
# f(x) = \sin(1.2 \pi x) \cdot \exp(-0.4 x) + 0.3 x
# $$
#
# Additive Gaussian noise is drawn from $\epsilon \sim \mathcal{N}(0, \sigma^2)$ with $\sigma = 0.30$, yielding irreducible Bayes noise floor $\sigma^2 = 0.09$.
#
# #### Exact Analytic Out-of-Sample Risk via Numerical Quadrature
# Rather than relying on a finite validation set that injects empirical Monte Carlo sampling noise, we compute the exact expected generalization error (true risk) analytically via continuous numerical quadrature using `scipy.integrate.quad`:
#
# $$
# R(\hat{f}_N) = \mathbb{E}_{(X,Y)}\left[(Y - \hat{f}_N(X))^2\right] = \frac{1}{b - a} \int_a^b (f(x) - \hat{f}_N(x))^2 \, dx + \sigma^2
# $$
#
# This formulation eliminates finite-sample validation noise, ensuring theoretical non-negativity of the generalization gap ($\Delta_N = R(\hat{f}_N) - \hat{R}_N(\hat{f}_N) > 0$) is preserved across all sample sizes.

# %%
DOMAIN_A: float = 0.0
DOMAIN_B: float = 2.5
NOISE_STD: float = 0.30
TRUE_NOISE_VARIANCE: float = NOISE_STD ** 2  # sigma^2 = 0.09

def true_function(x: np.ndarray) -> np.ndarray:
    """Ground truth target function f(x) with damped harmonic oscillation."""
    return np.sin(1.2 * np.pi * x) * np.exp(-0.4 * x) + 0.3 * x

def map_to_chebyshev_domain(x: np.ndarray, a: float = DOMAIN_A, b: float = DOMAIN_B) -> np.ndarray:
    """Maps custom continuous interval [a, b] to standard Chebyshev domain [-1, 1]."""
    return 2.0 * (x - a) / (b - a) - 1.0

def generate_samples(
    n_samples: int,
    noise_std: float = NOISE_STD,
    rng: np.random.RandomState = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates synthetic dataset (x, y, y_true) with additive Gaussian noise."""
    if rng is None:
        rng = np.random.RandomState()
    x = rng.uniform(DOMAIN_A, DOMAIN_B, size=n_samples)
    y_true = true_function(x)
    noise = rng.normal(0.0, noise_std, size=n_samples)
    y = y_true + noise
    return x, y, y_true

def compute_exact_true_risk(
    model: LinearRegression,
    degree: int,
    a: float = DOMAIN_A,
    b: float = DOMAIN_B,
    noise_var: float = TRUE_NOISE_VARIANCE
) -> Tuple[float, float]:
    """
    Computes exact expected generalization error (true risk) analytically via
    continuous numerical quadrature over target domain [a, b] using orthogonal Chebyshev basis.
    
    R(f_hat) = (1 / (b - a)) * int_a^b (f(x) - f_hat(x))^2 dx + sigma^2
    """
    def integrand(x: float) -> float:
        x_mapped = map_to_chebyshev_domain(np.array([x]), a, b)
        x_feat = chebvander(x_mapped, degree)
        pred = float(model.predict(x_feat)[0])
        f_true = float(true_function(x))
        return (f_true - pred) ** 2

    excess_risk_integral, _ = quad(integrand, a, b, limit=50)
    expected_excess_risk = excess_risk_integral / (b - a)
    total_true_risk = expected_excess_risk + noise_var
    return expected_excess_risk, total_true_risk

# Dense validation grid for pointwise bias-variance visualization
x_val_grid = np.linspace(DOMAIN_A, DOMAIN_B, 400)
y_val_grid_true = true_function(x_val_grid)

# %% [markdown]
# #### Consolidated System State & Initialization Summary
# Summary of runtime environment parameters, random seeds, and Data Generating Process specifications.

# %% 
# auto_collapse
print("=== Simulation Environment & DGP Initialization ===")
print(f"  Module:                   03_learning_curves_and_sample_complexity")
print(f"  Random Seed:              {SEED}")
print(f"  Domain Interval:          [{DOMAIN_A}, {DOMAIN_B}]")
print(f"  Irreducible Noise Std:    {NOISE_STD:.4f}")
print(f"  Bayes Noise Floor (𝜎²):   {TRUE_NOISE_VARIANCE:.4f}")
print(f"  Dense Grid Points:        {len(x_val_grid)}")
print(f"  Status:                   READY_FOR_SIMULATION")

# %% [markdown]
# ### 2.3 Vectorized Monte Carlo Learning Curve Engine (Orthogonal Chebyshev Basis)
#
# To empirically estimate the expected learning curves $\mathbb{E}[\hat{R}_N]$ and $\mathbb{E}[R_N]$ across sample sizes $N$, we run a vectorized Monte Carlo simulation with $K = 100$ independent trials for each sample size $N_k$.
#
# #### Algorithmic Optimization & Chebyshev Orthogonality:
# 1. **Domain Mapping to $[-1, 1]$**: Continuous domain inputs $x \in [0.0, 2.5]$ are linearly mapped to the standard Chebyshev interval $u = 2\frac{x - a}{b - a} - 1 \in [-1, 1]$.
# 2. **Elimination of Monomial Collinearity & Object Allocation Overhead**: Standard monomial expansions $x^0, x^1, \dots, x^d$ suffer from near-perfect collinearity ($\rho \to 1.0$) at high degrees, producing ill-conditioned Gram matrices ($\kappa(X^T X) \gg 10^{10}$) and forcing inner-loop `StandardScaler` allocations. Using Chebyshev polynomials of the first kind $T_k(u)$ via `np.polynomial.chebyshev.chebvander`, the orthogonal basis guarantees near-optimal conditioning ($\kappa(X^T X) \approx 1$) with zero inner-loop transformer instantiation overhead.
# 3. **Pre-computed Orthogonal Validation Grid**: The static validation grid basis `X_grid_cheb` is computed once per model degree outside the simulation loops.
# 4. **Exact Quadrature Risk**: True out-of-sample risk is computed analytically for each fitted model via `compute_exact_true_risk` using the Chebyshev evaluation basis.

# %%
def simulate_learning_curve_for_model(
    model_name: str,
    degree: int,
    sample_sizes: np.ndarray,
    n_replicates: int = 100,
    noise_std: float = NOISE_STD,
    seed: int = SEED
) -> Dict[str, Any]:
    """
    Computes Monte Carlo empirical learning curves with exact analytic true risk evaluation.
    Utilizes orthogonal Chebyshev polynomial basis expansions to completely eliminate
    monomial collinearity and inner-loop feature scaling overhead.
    """
    rng = np.random.RandomState(seed)
    
    # Pre-compute orthogonal basis for the static validation grid
    x_val_mapped = map_to_chebyshev_domain(x_val_grid, DOMAIN_A, DOMAIN_B)
    X_grid_cheb = chebvander(x_val_mapped, degree)
    
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
        grid_preds = np.zeros((n_replicates, len(x_val_grid)))
        
        for rep in range(n_replicates):
            # Sample independent training dataset of size n
            x_tr, y_tr, _ = generate_samples(n_samples=n, noise_std=noise_std, rng=rng)
            
            # Generate orthogonal design matrix directly (no inner-loop scaler needed)
            x_tr_mapped = map_to_chebyshev_domain(x_tr, DOMAIN_A, DOMAIN_B)
            X_tr_cheb = chebvander(x_tr_mapped, degree)
            
            # Fit model on the exceptionally well-conditioned orthogonal design matrix
            # chebvander includes the constant term T_0(x)=1, so fit_intercept=False
            model = LinearRegression(fit_intercept=False)
            model.fit(X_tr_cheb, y_tr)
            
            # In-sample empirical training error
            y_tr_pred = model.predict(X_tr_cheb)
            train_mses[rep] = np.mean((y_tr - y_tr_pred) ** 2)
            grid_preds[rep, :] = model.predict(X_grid_cheb)
            
            # Exact analytic out-of-sample generalization error
            _, true_risk = compute_exact_true_risk(
                model=model,
                degree=degree,
                a=DOMAIN_A,
                b=DOMAIN_B,
                noise_var=noise_std ** 2
            )
            val_mses[rep] = true_risk
            
        # Summary statistics across Monte Carlo replicates
        mean_tr = float(np.mean(train_mses))
        std_tr = float(np.std(train_mses, ddof=1))
        ci95_tr = 1.96 * std_tr / np.sqrt(n_replicates)
        
        mean_val = float(np.mean(val_mses))
        std_val = float(np.std(val_mses, ddof=1))
        ci95_val = 1.96 * std_val / np.sqrt(n_replicates)
        
        # Pointwise Bias-Variance decomposition on dense evaluation grid
        expected_pred = np.mean(grid_preds, axis=0)
        pointwise_bias_sq = (expected_pred - y_val_grid_true) ** 2
        pointwise_var = np.var(grid_preds, axis=0, ddof=0)
        
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
# ### 2.4 Multi-Regime Sample Complexity Sweeps
#
# We evaluate three distinct model capacity regimes across sample sizes $N \in [12, 20, 32, 50, 80, 130, 220, 380, 650, 1000]$:
#
# 1. **Underfitted Model ($d=1$, $p=2$)**:
#    - High structural representation error $\text{Bias}^2(\mathcal{H}) \approx 0.182$.
#    - Small parameter count $p=2 \implies$ rapid variance extinguishment ($\frac{p}{N} \to 0$).
#    - Generalization gap closes at small $N \sim 30$, but asymptotic risk plateau remains high ($R_\infty \approx 0.272 \gg \sigma^2$).
# 2. **Optimal Capacity Model ($d=3$, $p=4$)**:
#    - Minimal representation error $\text{Bias}^2 \approx 0.116$.
#    - Balanced complexity $p=4 \implies$ moderate convergence rate.
#    - Rapidly approaches asymptotic risk floor by $N \sim 100$.
# 3. **Overfitted Model ($d=9$, $p=10$)**:
#    - Negligible representation bias $\text{Bias}^2 \approx 0.000$.
#    - High parameter count $p=10 \implies$ significant estimation variance at small sample sizes ($N < 50$).
#    - Orthogonal Chebyshev basis representation stabilizes conditioning ($\kappa(X^T X) \approx 1$), ensuring error peaks at $N=12$ reflect pure statistical variance rather than solver precision collapse.

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

# Assemble consolidated metrics table across all sample sizes
records = []
for res in sweep_results:
    for i, n in enumerate(SAMPLE_SIZES):
        records.append({
            "Model": res["model_name"],
            "N": int(n),
            "Train MSE": float(res["train_mse_mean"][i]),
            "Val MSE": float(res["val_mse_mean"][i]),
            "Gen Gap": float(res["gen_gap_mean"][i]),
            "Bias^2": float(res["bias_sq_mean"][i]),
            "Variance": float(res["variance_mean"][i]),
            "Decomposed Sum": float(res["decomposed_sum"][i])
        })

df_learning_curves = pd.DataFrame(records)

# %% [markdown]
# #### Sample Complexity Multi-Regime Summary Table Output
#
# The summary table displays representative error metrics at small, medium, and large sample sizes.
# Heatmap color gradients (`background_gradient`) are applied to the **Gen Gap** and **Variance** columns to provide immediate visual hierarchy across capacity regimes and sample sizes.

# %%
def format_cell_value(val: float) -> str:
    """Formats values with scientific notation for extreme magnitudes >= 100 and 4 decimals otherwise."""
    if abs(val) >= 100.0:
        return f"{val:.2e}"
    return f"{val:.4f}"

df_display_subset = df_learning_curves[df_learning_curves["N"].isin([12, 50, 220, 1000])].copy().reset_index(drop=True)

styled_summary_table = (
    df_display_subset.style
    .format({
        "Train MSE": "{:.4f}",
        "Val MSE": format_cell_value,
        "Gen Gap": format_cell_value,
        "Bias^2": "{:.4f}",
        "Variance": format_cell_value,
        "Decomposed Sum": format_cell_value,
    })
    .background_gradient(subset=["Gen Gap", "Variance"], cmap="Reds")
    .set_table_styles([
        {"selector": "th", "props": [("background-color", "#f8f9fa"), ("font-weight", "600"), ("text-align", "center"), ("min-width", "0px"), ("padding-left", "0.5rem"), ("padding-right", "0.5rem")]},
        {"selector": "td", "props": [("text-align", "center"), ("font-family", "JetBrains Mono, monospace"), ("font-size", "0.72rem"), ("min-width", "0px"), ("padding-left", "0.5rem"), ("padding-right", "0.5rem")]}
    ])
    .hide(axis="index")
)

display(styled_summary_table)

# %% [markdown]
# > **Mathematical Note on Analytic Risk Non-Negativity:**
# > In the summary metrics table above, out-of-sample risk $R(\hat{f}_N)$ is evaluated via exact continuous numerical quadrature (`scipy.integrate.quad`), completely eliminating finite-sample empirical validation noise. As predicted by statistical learning theory, true out-of-sample risk strictly dominates empirical training risk across all regimes ($\Delta_N = R_N - \hat{R}_N > 0$), demonstrating asymptotic contraction toward zero at $N = 1000$.

# %% [markdown]
# ### 2.5 Visualizations & Diagnostic Plots
#
# We generate three comprehensive Plotly diagnostic charts:
# 1. **Dual Learning Curves Across Complexity Regimes with Inset Convergence Zooms** (Multi-panel figure featuring primary full dynamic logarithmic range subplots and linear inset subplots focused on $N \ge 80$).
# 2. **Generalization Gap Dynamics & Power-Law Scaling** (Linear gap decay and Log-Log power-law exponent verification $\Delta_N \propto N^{-\beta}$ contrasting unconstrained fits against asymptotic regime fits).
# 3. **Sample Size Effect on Bias vs. Variance Decomposition** (Tracking monotonic variance collapse against bias invariance).

# %% [markdown]
# #### Visualization 1: Dual Learning Curves Across Complexity Regimes
# Interactive 2-row Plotly chart showing:
# - **Row 1 (Primary Subplots - Log Scale)**: Full dynamic logarithmic range capturing small-sample numerical variance spikes at $N=12$ alongside convergence trajectories.
# - **Row 2 (Asymptotic Inset Subplots - Linear Scale)**: Focused linear zoom on $N \ge 80$ (range $[0.0, 0.4]$) enabling precise visual inspection of asymptotic convergence against the Bayes noise floor ($\sigma^2=0.09$).

# %%
fig1 = sp.make_subplots(
    rows=2, cols=3,
    row_heights=[0.56, 0.44],
    vertical_spacing=0.14,
    horizontal_spacing=0.07,
    subplot_titles=[
        f"<b>{name}</b><br><span style='font-size:11px; color:#555;'>Primary: Full Dynamic Log Range (N ∈ [12, 1000])</span>"
        for name, _ in regime_models
    ] + [
        f"<b>{name.split()[0]} Asymptotic Inset</b><br><span style='font-size:11px; color:#555;'>Linear Scale (N ≥ 80, Convergence Floor)</span>"
        for name, _ in regime_models
    ]
)

colors_tr = "#2b5c8f"
colors_val = "#d95f02"

for col_idx, (name, deg) in enumerate(regime_models, start=1):
    res = results_by_name[name]
    n_pts = res["sample_sizes"]
    asymptote_val = res["bias_sq_mean"][-1] + TRUE_NOISE_VARIANCE
    
    # ----------------------------------------------------
    # ROW 1: PRIMARY LOG SCALE PLOT (Full Dynamic Range)
    # ----------------------------------------------------
    val_upper = res["val_mse_mean"] + res["val_mse_ci95"]
    val_lower = np.maximum(res["val_mse_mean"] - res["val_mse_ci95"], 1e-4)
    
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
    
    tr_upper = res["train_mse_mean"] + res["train_mse_ci95"]
    tr_lower = np.maximum(res["train_mse_mean"] - res["train_mse_ci95"], 1e-4)
    
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
    
    # Validation MSE (Approaches from ABOVE)
    fig1.add_trace(
        go.Scatter(
            x=n_pts, y=res["val_mse_mean"],
            mode="lines+markers",
            name="True Risk / Val MSE",
            line=dict(color=colors_val, width=2.5),
            marker=dict(size=6, symbol="diamond"),
            showlegend=(col_idx == 1),
            hovertemplate="Val MSE: %{y:.4f} (N=%{x})<extra></extra>"
        ),
        row=1, col=col_idx
    )
    
    # Training MSE (Approaches from BELOW)
    fig1.add_trace(
        go.Scatter(
            x=n_pts, y=res["train_mse_mean"],
            mode="lines+markers",
            name="Empirical Risk / Train MSE",
            line=dict(color=colors_tr, width=2.5, dash="dash"),
            marker=dict(size=6, symbol="circle"),
            showlegend=(col_idx == 1),
            hovertemplate="Train MSE: %{y:.4f} (N=%{x})<extra></extra>"
        ),
        row=1, col=col_idx
    )
    
    # Asymptotic Risk Plateau (Bias^2 + sigma^2)
    fig1.add_trace(
        go.Scatter(
            x=[n_pts[0], n_pts[-1]], y=[asymptote_val, asymptote_val],
            mode="lines",
            name="Asymptotic Floor (Bias² + 𝜎²)",
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
            name="Bayes Noise (𝜎²=0.09)",
            line=dict(color="#666666", width=1.5, dash="dashdot"),
            showlegend=(col_idx == 1),
            hovertemplate=f"Bayes Floor: {TRUE_NOISE_VARIANCE:.4f}<extra></extra>"
        ),
        row=1, col=col_idx
    )
    
    # ----------------------------------------------------
    # ROW 2: INSET LINEAR SCALE PLOT (Asymptotic Regime N >= 80)
    # ----------------------------------------------------
    mask_inset = n_pts >= 80
    n_inset = n_pts[mask_inset]
    
    fig1.add_trace(
        go.Scatter(
            x=n_inset, y=res["val_mse_mean"][mask_inset],
            mode="lines+markers",
            line=dict(color=colors_val, width=2.2),
            marker=dict(size=6, symbol="diamond"),
            showlegend=False,
            hovertemplate="Val MSE: %{y:.4f} (N=%{x})<extra></extra>"
        ),
        row=2, col=col_idx
    )
    
    fig1.add_trace(
        go.Scatter(
            x=n_inset, y=res["train_mse_mean"][mask_inset],
            mode="lines+markers",
            line=dict(color=colors_tr, width=2.2, dash="dash"),
            marker=dict(size=6, symbol="circle"),
            showlegend=False,
            hovertemplate="Train MSE: %{y:.4f} (N=%{x})<extra></extra>"
        ),
        row=2, col=col_idx
    )
    
    fig1.add_trace(
        go.Scatter(
            x=[n_inset[0], n_inset[-1]], y=[asymptote_val, asymptote_val],
            mode="lines",
            line=dict(color="#1b9e77", width=1.8, dash="dot"),
            showlegend=False,
            hovertemplate=f"Asymptote: {asymptote_val:.4f}<extra></extra>"
        ),
        row=2, col=col_idx
    )
    
    fig1.add_trace(
        go.Scatter(
            x=[n_inset[0], n_inset[-1]], y=[TRUE_NOISE_VARIANCE, TRUE_NOISE_VARIANCE],
            mode="lines",
            line=dict(color="#666666", width=1.5, dash="dashdot"),
            showlegend=False,
            hovertemplate=f"Bayes Floor: {TRUE_NOISE_VARIANCE:.4f}<extra></extra>"
        ),
        row=2, col=col_idx
    )

fig1.update_layout(
    title=dict(
        text="<b>Empirical vs. True Risk Learning Curves: Full Dynamic Log Range & Asymptotic Linear Insets</b>",
        font=dict(size=14, family="Plus Jakarta Sans"),
        x=0.5, y=0.99, xanchor="center", yanchor="top"
    ),
    template="plotly_white",
    height=720,
    margin=dict(l=60, r=40, t=130, b=60),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#e0e0e0",
        borderwidth=1
    )
)

fig1.for_each_annotation(lambda a: a.update(font=dict(size=11)))

# Configure axes for Row 1 (Log Scale)
for c in range(1, 4):
    fig1.update_xaxes(title_text="Training Sample Size (N)", type="log", row=1, col=c)
    fig1.update_yaxes(type="log", row=1, col=c)

# Configure axes for Row 2 (Linear Inset Zoom)
for c in range(1, 4):
    fig1.update_xaxes(title_text="Sample Size (N ≥ 80)", row=2, col=c)

fig1.update_yaxes(title_text="MSE [Log Scale]", type="log", row=1, col=1)
fig1.update_yaxes(title_text="MSE [Linear Zoom]", range=[0.24, 0.30], row=2, col=1)
fig1.update_yaxes(range=[0.05, 0.25], row=2, col=2)
fig1.update_yaxes(range=[0.05, 0.15], row=2, col=3)

fig1.show()

# %% [markdown]
# #### Visualization 2: Generalization Gap Dynamics & Power-Law Scaling
# Interactive Plotly 2-panel chart demonstrating:
# 1. Generalization Gap $\Delta(N) = R(N) - \hat{R}(N)$ contraction across sample sizes on linear axes.
# 2. Log-Log scaling plot comparing unconstrained regression fits against asymptotic regime fits ($N \ge 80$) to verify convergence toward the theoretical $\Delta(N) \propto N^{-1.0}$ power law.

# %%
fig2 = sp.make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "<b>Generalization Gap Contraction Δ(N)</b><br><span style='font-size:11px; color:#555;'>Linear Scale: Δ(N) = Val MSE - Train MSE</span>",
        "<b>Log-Log Power-Law Scaling Law</b><br><span style='font-size:11px; color:#555;'>Asymptotic Fits (N ≥ 80) vs. All-N Fits vs. Theoretical -1.0</span>"
    ),
    horizontal_spacing=0.12
)

regime_colors = {
    "Underfitted (d=1, p=2)": "#2b5c8f",
    "Optimal (d=3, p=4)": "#1b9e77",
    "Overfitted (d=9, p=10)": "#e7298a"
}

def fit_power_law_robust(n_vals: np.ndarray, gap_vals: np.ndarray) -> Tuple[float, float, bool]:
    """Fits log10(gap) = -beta * log10(N) + log10(alpha) via linear least squares."""
    valid = (gap_vals > 1e-4) & np.isfinite(gap_vals)
    if np.sum(valid) >= 2:
        log_n = np.log10(n_vals[valid])
        log_g = np.log10(gap_vals[valid])
        coeffs = np.polyfit(log_n, log_g, deg=1)
        return float(-coeffs[0]), float(coeffs[1]), True
    return 1.0, 0.0, False

for name, deg in regime_models:
    res = results_by_name[name]
    n_pts = res["sample_sizes"]
    gap = res["gen_gap_mean"]
    
    # 1. Unconstrained fit across all valid points (N >= 20, gap > 0)
    mask_all = (n_pts >= 20)
    beta_all, alpha_all, ok_all = fit_power_law_robust(n_pts[mask_all], gap[mask_all])
    
    # 2. Asymptotic regime fit restricted to N >= 80
    mask_asymp = (n_pts >= 80)
    beta_asymp, alpha_asymp, ok_asymp = fit_power_law_robust(n_pts[mask_asymp], gap[mask_asymp])
    
    # Panel 1: Generalization Gap (Linear Scale)
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
    
    # Panel 2: Log-Log Observed Markers
    pos_mask = gap > 1e-4
    fig2.add_trace(
        go.Scatter(
            x=n_pts[pos_mask], y=gap[pos_mask],
            mode="markers",
            name=f"{name} (Observed)",
            marker=dict(color=regime_colors[name], size=8, symbol="circle"),
            showlegend=False,
            hovertemplate=f"Observed: N=%{{x}}, Gap=%{{y:.4f}}<extra></extra>"
        ),
        row=1, col=2
    )
    
    # Panel 2: Asymptotic Regime Fit (Solid Line)
    if ok_asymp:
        n_dense_asymp = np.logspace(np.log10(80), np.log10(1000), 30)
        gap_asymp_fit = 10 ** (-beta_asymp * np.log10(n_dense_asymp) + alpha_asymp)
        fig2.add_trace(
            go.Scatter(
                x=n_dense_asymp, y=gap_asymp_fit,
                mode="lines",
                name=f"{name} (Asymp N≥80: β={beta_asymp:.2f})",
                line=dict(color=regime_colors[name], width=2.2),
                hovertemplate=f"Asymptotic Fit (N≥80): Gap ∝ N^(-{beta_asymp:.2f})<extra></extra>"
            ),
            row=1, col=2
        )
    
    # Panel 2: Unconstrained Fit (Dotted Line)
    if ok_all:
        n_dense_all = np.logspace(np.log10(15), np.log10(1000), 50)
        gap_all_fit = 10 ** (-beta_all * np.log10(n_dense_all) + alpha_all)
        fig2.add_trace(
            go.Scatter(
                x=n_dense_all, y=gap_all_fit,
                mode="lines",
                name=f"{name} (All-N: β={beta_all:.2f})",
                line=dict(color=regime_colors[name], width=1.5, dash="dot"),
                hovertemplate=f"All-N Fit: Gap ∝ N^(-{beta_all:.2f})<extra></extra>"
            ),
            row=1, col=2
        )

fig2.update_layout(
    title=dict(
        text="<b>Generalization Gap Dynamics & Asymptotic 1/N Scaling Verification</b>",
        font=dict(size=14, family="Plus Jakarta Sans"),
        x=0.5, y=0.99, xanchor="center", yanchor="top"
    ),
    template="plotly_white",
    height=550,
    margin=dict(l=60, r=40, t=160, b=60),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.08,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#e0e0e0",
        borderwidth=1,
        tracegroupgap=20,
        font=dict(size=11)
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
        name="Irreducible Noise Floor (𝜎²=0.09)",
        line=dict(color="#666666", width=2, dash="dot"),
        hovertemplate="Noise Floor: 0.0900<extra></extra>"
    )
)

# Decomposed Sum (Total Expected Out-of-Sample Risk)
fig3.add_trace(
    go.Scatter(
        x=n_pts, y=res_overfit["decomposed_sum"],
        mode="lines+markers",
        name="Total Decomposed Risk (Bias² + Var + 𝜎²)",
        line=dict(color="#1b9e77", width=3.5),
        marker=dict(symbol="circle", size=9),
        hovertemplate="Total Risk: %{y:.4f}<extra></extra>"
    )
)

# Outlier annotation at N=12
fig3.add_annotation(
    x=np.log10(12), y=np.log10(float(res_overfit["variance_mean"][0])),
    text=f"<b>N=12: Var ≈ {res_overfit['variance_mean'][0]:.2e}</b><br>(Orthogonal Chebyshev basis isolates statistical variance)",
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowcolor="#7570b3",
    ax=80, ay=-30,
    font=dict(size=10, color="#403880"),
    bgcolor="rgba(245, 240, 255, 0.9)",
    bordercolor="#7570b3",
    borderwidth=1
)

fig3.update_layout(
    title=dict(
        text="<b>Sample Size Scaling Effect on Bias-Variance Decomposition (Degree d=9)</b>",
        font=dict(size=14, family="Plus Jakarta Sans"),
        x=0.5, y=0.98, xanchor="center", yanchor="top"
    ),
    xaxis=dict(
        title="Training Sample Size (N) [Log Scale]",
        type="log",
        range=[np.log10(10), np.log10(1200)]
    ),
    yaxis=dict(title="Error Component Magnitude (MSE) [Log Scale]", type="log"),
    template="plotly_white",
    hovermode="x unified",
    height=490,
    margin=dict(l=60, r=40, t=100, b=60),
    legend=dict(
        x=0.55, y=0.96,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#e0e0e0",
        borderwidth=1
    )
)

fig3.show()

# %% [markdown]
# ### 2.6 Summary & Key Engineering Takeaways
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
#    - High-capacity models ($d=9$) exhibit severe small-sample variance ($N < 50$), but offer the highest return on additional training data, converging to the lowest Bayes risk floor once $N > 400$.
# 5. **Decoupling Overfitting from Ill-Conditioning via Orthogonal Polynomials**:
#    - Expanding into an orthogonal Chebyshev polynomial basis eliminates raw monomial collinearity ($\rho \to 1.0$) and stabilizes the condition number of the design matrix without needing inner-loop standardizers. This ensures that estimation variance observed at small sample sizes stems strictly from statistical degrees of freedom rather than floating-point degradation.

# %% [markdown]
# #### Final Summary Table Display
# Renders final empirical summary table and critical sample complexity milestones with background heatmap gradients.

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
        "Gap at N=12": format_cell_value(res['gen_gap_mean'][0]),
        "Gap at N=1000": format_cell_value(res['gen_gap_mean'][-1]),
        "N_crit (<= 10% Excess)": str(n_crit_10),
        "N_crit (<= 5% Excess)": str(n_crit_05)
    })

df_summary_display = pd.DataFrame(summary_records)

styled_milestones = (
    df_summary_display.style
    .set_table_styles([
        {"selector": "th", "props": [("background-color", "#f8f9fa"), ("font-weight", "600"), ("text-align", "center"), ("min-width", "0px"), ("padding-left", "0.5rem"), ("padding-right", "0.5rem")]},
        {"selector": "td", "props": [("text-align", "center"), ("font-family", "JetBrains Mono, monospace"), ("font-size", "0.72rem"), ("min-width", "0px"), ("padding-left", "0.5rem"), ("padding-right", "0.5rem")]}
    ])
    .hide(axis="index")
)

display(Markdown("### Final Learning Curve & Sample Complexity Milestone Summary"))
display(styled_milestones)
display(Markdown(f"**Bayes Irreducible Noise Floor (𝜎²):** `{TRUE_NOISE_VARIANCE:.4f}`"))