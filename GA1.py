# Advanced Econometrics - Group Assignment 1
# Group 18:
#   Phuc Nguyen, 2779150
#   Tu Nguyen, 2849240
#   Boris de Buck, xxxxxx
#   Anna van Dam, xxxxxx

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.optimize import minimize


def load_returns(filepath="data/return_data.csv"):
    df = pd.read_csv(filepath, parse_dates=["DlyCalDt"])

    # Scale returns to percentage points
    df["DlyRet"] = df["DlyRet"] * 100

    # Reshape: one column per ticker, indexed by date
    wide = df.pivot(index="DlyCalDt", columns="Ticker", values="DlyRet")
    wide = wide.sort_index()

    return wide


def sanity_check(data):
    """
    Check AAPL stats against assignment's reference values.
    """
    aapl = data["AAPL"]
    assert abs(aapl.mean() - 0.1071) < 0.001, f"Mean mismatch: {aapl.mean():.4f}"
    assert abs(aapl.std() - 1.7832) < 0.001, f"Std mismatch: {aapl.std():.4f}"
    assert abs(aapl.min() - (-12.8647)) < 0.001, f"Min mismatch: {aapl.min():.4f}"
    assert abs(aapl.max() - 11.9808) < 0.001, f"Max mismatch: {aapl.max():.4f}"
    print("AAPL sanity check passed.")


def news_impact(x, alpha, delta, gamma):
    """
    News-impact curve: isolates how a shock x_{t-1} affects volatility,
    dropping terms that don't depend on x_{t-1}.

    From sigma_t^2 = omega + (alpha + delta*tanh(-gamma*x)) * ((x-mu-lambda*sigma^2)/sigma)^2 + beta*sigma^2:
      - set mu=0, lambda=0, sigma_{t-1}=1 -> squared term reduces to x^2
      - drop omega (constant, unrelated to x) and beta*sigma^2 (past
        volatility's own persistence, not the news effect)

    What remains is purely the shock-driven piece:
        NIC(x) = (alpha + delta*tanh(-gamma*x)) * x^2
    """
    return (alpha + delta * np.tanh(-gamma * x)) * x**2


def plot_news_impact_curves():
    alpha = 0.4
    gammas = [0.01, 0.1, 1]
    deltas = [0.3, 0.1, 0, -0.3]
    x = np.linspace(-5, 5, 500)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, delta in enumerate(deltas):
        ax = axes[i]
        for gamma in gammas:
            y = news_impact(x, alpha, delta, gamma)
            ax.plot(x, y, label=f"γ={gamma}")
        ax.set_title(f"δ = {delta}")
        ax.set_xlabel(r"$x_{t-1}$")
        ax.set_ylabel("News impact")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("News-Impact Curves for Different δ and γ (α=0.4)")
    plt.tight_layout()
    plt.savefig("outputs/news_impact_curves.png", dpi=150)
    plt.show()


def descriptive_stats(data):
    """
    Compute descriptive statistics for each ticker in the dataset.
    Return a DataFrame with ticker names as index and statistics as columns.
    """
    stats_dict = {}
    for ticker in data.columns:
        series = data[ticker].dropna()
        stats_dict[ticker] = {
            "Obs": len(series),
            "Mean": series.mean(),
            "Median": series.median(),
            "Std Dev": series.std(),
            "Skewness": series.skew(),
            "Excess Kurtosis": series.kurtosis(),
            "Min": series.min(),
            "Max": series.max(),
        }
    return pd.DataFrame(stats_dict).T


def plot_returns(data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    for i, ticker in enumerate(data.columns):
        axes[i].plot(data.index, data[ticker], linewidth=0.5)
        axes[i].set_title(ticker)
        axes[i].set_ylabel("Return (%)")
        axes[i].grid(alpha=0.3)

    fig.suptitle("Daily Returns, 2011-2023")
    plt.tight_layout()
    plt.savefig("outputs/returns_panels.png", dpi=150)
    plt.show()


def log_t_pdf(z, nu):
    """Log density of the standard Student-t at value z."""
    return (
        gammaln((nu + 1) / 2)
        - gammaln(nu / 2)
        - 0.5 * np.log(nu * np.pi)
        - ((nu + 1) / 2) * np.log1p(z**2 / nu)  # using log1p for numerical stability
    )


def unpack_params(theta, model):
    """Map flat theta to named params, fixing lambda/delta/gamma=0 for nested models."""
    if model == "GARCH":
        mu, omega, alpha, beta, nu = theta
        lam, delta, gamma = 0.0, 0.0, 0.0
    elif model == "GARCH-M":
        mu, lam, omega, alpha, beta, nu = theta
        delta, gamma = 0.0, 0.0
    elif model == "GARCH-M-L":
        mu, lam, omega, alpha, beta, delta, gamma, nu = theta
    else:
        raise ValueError(model)
    return mu, lam, omega, alpha, beta, delta, gamma, nu


def filter_sigma2(x, mu, lam, omega, alpha, beta, delta, gamma, sigma2_1):
    """Run the GARCH-M-L recursion. Returns the sigma2 path, or None if the
    recursion hits a non-positive/exploded variance (signals an invalid
    parameter region during optimization)."""
    T = len(x)
    sigma2 = np.empty(T)
    sigma2[0] = sigma2_1

    for t in range(T - 1):
        z = (x[t] - mu - lam * sigma2[t]) / np.sqrt(sigma2[t])
        coef = alpha + delta * np.tanh(-gamma * x[t])
        sigma2[t + 1] = omega + coef * z**2 + beta * sigma2[t]
        if not np.isfinite(sigma2[t + 1]) or sigma2[t + 1] <= 0 or sigma2[t + 1] > 1e8:
            return None

    return sigma2


def neg_log_likelihood(theta, x, sigma2_1, model):
    """Negative log-likelihood via prediction error decomposition. Uses the
    Jacobian-corrected density (log f(z_t) - 0.5*log(sigma_t^2)) since x_t is
    a shift-and-scale transform of eps_t, not simply f(z_t) alone."""
    mu, lam, omega, alpha, beta, delta, gamma, nu = unpack_params(theta, model)

    if beta < 0 or nu <= 2 or gamma < 0:
        return 1e10

    sigma2 = filter_sigma2(x, mu, lam, omega, alpha, beta, delta, gamma, sigma2_1)
    if sigma2 is None:
        return 1e10

    z = (x - mu - lam * sigma2) / np.sqrt(sigma2)
    ll = np.sum(log_t_pdf(z, nu) - 0.5 * np.log(sigma2))
    if not np.isfinite(ll):
        return 1e10

    if model == "GARCH-M-L":
        ll -= 0.001 * gamma**2

    return -ll


# ---------- Initial values ----------


def get_sigma2_1(x):
    """sigma_1^2 = variance of first 50 obs, ddof=0 (divide by 50, not 49) per spec."""
    return np.var(x[:50], ddof=0)


def get_initial_values(x, model, T=2500):
    s2 = np.var(x[:T], ddof=0)
    if model == "GARCH":
        return np.array([0.0, s2 / 50, 0.05, 0.9, 10.0])
    elif model == "GARCH-M":
        return np.array([0.0, 0.0, s2 / 50, 0.05, 0.9, 10.0])
    elif model == "GARCH-M-L":
        return np.array([0.0, 0.0, s2 / 50, 0.05, 0.9, 0.01, 0.01, 10.0])


PARAM_NAMES = {
    "GARCH": ["mu", "omega", "alpha", "beta", "nu"],
    "GARCH-M": ["mu", "lambda", "omega", "alpha", "beta", "nu"],
    "GARCH-M-L": ["mu", "lambda", "omega", "alpha", "beta", "delta", "gamma", "nu"],
}

# Widened eps and relaxed gtol vs. BFGS defaults: the default finite-difference
# step is too small for a 2,500-step recursive likelihood, causing "precision
# loss" warnings even at the true optimum.
BFGS_OPTIONS = {"maxiter": 5000, "gtol": 1e-3, "eps": 1e-6}

# ---------- Estimation loop ----------


def estimate_all(data, T=2500):
    results = []

    for ticker in data.columns:
        series = data[ticker].dropna().values
        x = series[:T]
        sigma2_1 = get_sigma2_1(x)

        for model in ["GARCH", "GARCH-M", "GARCH-M-L"]:
            theta0 = get_initial_values(x, model, T)
            res = minimize(
                neg_log_likelihood,
                theta0,
                args=(x, sigma2_1, model),
                method="BFGS",
                options=BFGS_OPTIONS,
            )

            if not res.success:
                print(f"WARNING: {ticker} {model} did not converge: {res.message}")

            k = len(theta0)
            loglik = -res.fun
            aic = 2 * k - 2 * loglik
            bic = k * np.log(T) - 2 * loglik

            row = {
                "Ticker": ticker,
                "Model": model,
                "LogLik": loglik,
                "AIC": aic,
                "BIC": bic,
                "Converged": res.success,
            }
            for name, val in zip(PARAM_NAMES[model], res.x):
                row[name] = val
            results.append(row)

    return pd.DataFrame(results)


def plot_q5(data, results, T_in=2500):
    tickers = [t for t in data.columns if t != "AAPL"]
    x_grid = np.linspace(-5, 5, 500)
    fig, axes = plt.subplots(len(tickers), 2, figsize=(12, 12))

    for row, ticker in enumerate(tickers):
        series = data[ticker].dropna()
        x_full = series.values
        dates = series.index

        # ---- Left panel: news-impact curves for all three models ----
        ax_left = axes[row, 0]
        for model in ["GARCH", "GARCH-M", "GARCH-M-L"]:
            p = results[
                (results["Ticker"] == ticker) & (results["Model"] == model)
            ].iloc[0]
            alpha, delta, gamma = p["alpha"], p.get("delta", 0.0), p.get("gamma", 0.0)
            if pd.isna(delta):
                delta = 0.0
            if pd.isna(gamma):
                gamma = 0.0
            y = news_impact(x_grid, alpha, delta, gamma)
            ax_left.plot(x_grid, y, label=model)
        ax_left.set_title(f"News impact curves for {ticker}")
        ax_left.set_xlabel(r"$x_{t-1}$")
        ax_left.set_ylabel("News impact")
        ax_left.legend()
        ax_left.grid(alpha=0.3)

        # ---- Right panel: filtered volatility (GARCH-M-L) over full sample ----
        ax_right = axes[row, 1]
        p = results[
            (results["Ticker"] == ticker) & (results["Model"] == "GARCH-M-L")
        ].iloc[0]
        sigma2_1 = get_sigma2_1(x_full[:T_in])
        sigma2_full = filter_sigma2(
            x_full,
            p["mu"],
            p["lambda"],
            p["omega"],
            p["alpha"],
            p["beta"],
            p["delta"],
            p["gamma"],
            sigma2_1,
        )
        sigma_full = np.sqrt(sigma2_full)

        ax_right.plot(dates, x_full, color="lightgray", linewidth=0.5, label="Returns")
        ax_right.plot(dates, sigma_full, color="C0", linewidth=1, label="GARCH-M-L")
        ax_right.axvline(dates[T_in], color="black", linestyle="--", linewidth=1)
        ax_right.set_title(f"Filtered volatilities for {ticker}")
        ax_right.legend()
        ax_right.grid(alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.savefig("outputs/q5_panels.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    data = load_returns()
    print(data.head())
    print(f"\nShape: {data.shape}")
    print(f"Tickers: {list(data.columns)}")
    print(f"Date range: {data.index.min()} to {data.index.max()}\n")

    sanity_check(data)

    plot_news_impact_curves()

    stats_df = descriptive_stats(data)
    print(stats_df.round(4))
    stats_df.round(4).to_csv("outputs/descriptive_stats.csv")
    plot_returns(data)

    results = estimate_all(data)

    cols = [
        "Ticker",
        "Model",
        "mu",
        "lambda",
        "omega",
        "alpha",
        "beta",
        "delta",
        "gamma",
        "nu",
        "LogLik",
        "AIC",
        "BIC",
        "Converged",
    ]
    results = results[cols]

    print(results.round(4).to_string(index=False))
    results.round(4).to_csv("outputs/garch_estimates.csv", index=False)

    # Validation against assignment's Table 1: should match within ~0.5.
    aapl = results[results["Ticker"] == "AAPL"]
    print("\n--- AAPL check against Table 1 ---")
    print("Expected LogLik ≈ -4662 (M1), -4662 (M2), -4632 (M3)")
    print(aapl[["Model", "LogLik"]])

    # Question 5
    plot_q5(data, results)
