"""Utility functions for visualising DJF-season NINO3.4 distributions."""

import math
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from matplotlib.lines import Line2D
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


MONTH_COLORS = {
    "January": "tab:blue",
    "April": "tab:orange",
    "July": "tab:green",
    "October": "tab:red",
}

sns.set_style("whitegrid")


def _p_to_star(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_yearly_kde_panel(
    jan_nino34: xr.DataArray,
    apr_nino34: xr.DataArray,
    july_nino34: xr.DataArray,
    oct_nino34: xr.DataArray,
    eruption_year: int = 1258,
) -> plt.Figure:
    """Plot DJF-season KDEs for four start-month ensembles.

    One subplot is created for each year relative to ``eruption_year``
    (from Y-1 through Y+6). Within each subplot the December–February mean
    NINO3.4 anomaly distributions of the January, April, July, and October start
    months are shown as four KDE curves. Each panel title lists the number
    of contributing members and a caption summarises per-year ensemble
    statistics (mean ± standard deviation).

    Parameters
    ----------
    jan_nino34, apr_nino34, july_nino34, oct_nino34 : xarray.DataArray
        NINO3.4 anomaly ensembles with dimensions ``(member, time)``.
    eruption_year : int, optional
        Reference year for the eruption (default 1258).

    Returns
    -------
    matplotlib.figure.Figure
        The resulting panel figure.
    """

    start_months = OrderedDict(
        (
            ("January", jan_nino34),
            ("April", apr_nino34),
            ("July", july_nino34),
            ("October", oct_nino34),
        )
    )

    years = OrderedDict(
        (
            ("Y -1", eruption_year - 1),
            ("Y 0", eruption_year),
            ("Y +1", eruption_year + 1),
            ("Y +2", eruption_year + 2),
            ("Y +3", eruption_year + 3),
            ("Y +4", eruption_year + 4),
            ("Y +5", eruption_year + 5),
            ("Y +6", eruption_year + 6),
        )
    )

    n_years = len(years)
    ncols = 4
    nrows = math.ceil(n_years / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), sharex=True, sharey=True
    )
    axes = axes.ravel()

    year_stats: dict[str, dict[str, float]] = {}
    month_stats: dict[str, dict[str, dict[str, float]]] = {}
    baseline_vals: list[float] | None = None
    pval_labels: list[str] = []
    pvals: list[float] = []
    axes_dict: dict[str, plt.Axes] = {}

    for ax, (yr_label, yr_val) in zip(axes, years.items()):
        year_vals: list[float] = []
        month_stats[yr_label] = {}
        for month_label, da in start_months.items():
            djf = da.where(
                (
                    (da.time.dt.year == yr_val) & da.time.dt.month.isin([1, 2])
                )
                | (
                    (da.time.dt.year == yr_val - 1) & (da.time.dt.month == 12)
                ),
                drop=True,
            )
            seasonal = djf.mean(dim="time").values
            if len(seasonal) == 0 or np.allclose(seasonal, seasonal[0]):
                continue
            year_vals.extend(seasonal.tolist())
            month_stats[yr_label][month_label] = {
                "n": len(seasonal),
                "mean": float(np.mean(seasonal)),
                "std": float(np.std(seasonal, ddof=1)),
            }
            sns.kdeplot(
                seasonal,
                ax=ax,
                color=MONTH_COLORS[month_label],
                label=month_label,
                warn_singular=False,
            )
        ax.axvline(0, color="black", alpha=0.3)
        ax.axvline(0.5, color="red", linestyle=":", alpha=0.5)
        ax.axvline(-0.5, color="blue", linestyle=":", alpha=0.5)
        if year_vals:
            year_stats[yr_label] = {
                "n": len(year_vals),
                "mean": float(np.mean(year_vals)),
                "std": float(np.std(year_vals, ddof=1)),
            }
            axes_dict[yr_label] = ax
            if yr_label == "Y -1":
                baseline_vals = year_vals.copy()
            elif baseline_vals is not None:
                p = ttest_ind(year_vals, baseline_vals, equal_var=False).pvalue
                year_stats[yr_label]["pval"] = float(p)
                pval_labels.append(yr_label)
                pvals.append(p)
            ax.set_title(f"{yr_label}\n(n={len(year_vals)})")
        else:
            ax.set_title(yr_label)
        ax.set_xlim([-5, 5])
        ax.set_ylim([0, 1.75])

    if pvals:
        _, pvals_adj, _, _ = multipletests(pvals, method="fdr_bh")
        for lbl, p_adj in zip(pval_labels, pvals_adj):
            year_stats[lbl]["p_adj"] = float(p_adj)
            sig = _p_to_star(p_adj)
            year_stats[lbl]["sig"] = sig
            n = year_stats[lbl]["n"]
            axes_dict[lbl].set_title(f"{lbl}\n(n={n}){sig}")

    # Hide any unused axes
    for ax in axes[n_years:]:
        ax.set_visible(False)

    handles = [
        Line2D([0], [0], color=MONTH_COLORS[label], lw=2, label=label)
        for label in start_months
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(start_months),
        title="Start Month",
        bbox_to_anchor=(0.5, 0.93),
    )

    fig.text(0.5, 0.04, "DJF NINO3.4 Anomaly (°C)", ha="center")
    fig.text(0.04, 0.5, "Probability Density", va="center", rotation="vertical")

    # dynamic title
    fig.suptitle(
        f"DJF NINO3.4 Distributions Relative to {eruption_year}", y=0.98
    )

    # caption with per-year and per-month statistics
    if year_stats:
        caption_parts: list[str] = []
        for yr in years:
            if yr not in month_stats:
                continue
            s = year_stats.get(yr, {})
            months = month_stats.get(yr, {})
            month_parts = [
                f"{m[:3]} {v['mean']:.2f}±{v['std']:.2f}°C (n={v['n']})"
                for m, v in months.items()
            ]
            month_str = ", ".join(month_parts)
            if "p_adj" in s:
                caption_parts.append(
                    f"{yr}: {month_str}; p={s['p_adj']:.3f}{s['sig']}"
                )
            else:
                caption_parts.append(f"{yr}: {month_str}")
        caption = "; ".join(caption_parts)
        fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=8)

    fig.tight_layout(rect=[0, 0.07, 1, 0.92])
    return fig


if __name__ == "__main__":
    pass
