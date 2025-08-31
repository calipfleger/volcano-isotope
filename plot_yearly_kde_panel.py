"""Utility functions for visualising DJF-season NINO3.4 distributions."""

import math
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.lines import Line2D
from scipy.stats import chi2_contingency, ttest_ind
from statsmodels.stats.multitest import multipletests


MONTH_COLORS = {
    "January": "tab:blue",
    "April": "tab:orange",
    "July": "tab:green",
    "October": "tab:red",
}


def _p_to_star(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _make_caption_fig(text: str, base_fig: plt.Figure, rel_height: float = 0.15) -> plt.Figure:
    """Create a standalone caption figure matched to ``base_fig`` width.

    Parameters
    ----------
    text : str
        Caption text to display.
    base_fig : matplotlib.figure.Figure
        Figure whose width is used for the caption figure.
    rel_height : float, optional
        Height of the caption figure relative to ``base_fig`` (default 0.15).

    Returns
    -------
    matplotlib.figure.Figure
        Caption figure with the provided text.
    """

    width = base_fig.get_figwidth()
    height = max(1.0, base_fig.get_figheight() * rel_height)
    cap_fig, cap_ax = plt.subplots(figsize=(width, height))
    cap_ax.axis("off")
    cap_ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True, fontsize=9)
    return cap_fig


def _plot_yearly_kde_panel_impl(
    jan_nino34: xr.DataArray,
    apr_nino34: xr.DataArray,
    july_nino34: xr.DataArray,
    oct_nino34: xr.DataArray,
    eruption_year: int = 1258,
) -> tuple[
    plt.Figure,
    plt.Figure | None,
    plt.Figure | None,
    pd.DataFrame,
    plt.Figure | None,
    plt.Figure | None,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Plot DJF-season KDEs for four start-month ensembles.

    One subplot is created for each year relative to ``eruption_year``
    (from Y-1 through Y+6). Within each subplot the December–February mean
    NINO3.4 anomaly distributions of the January, April, July, and October start
    months are shown as four KDE curves. Each panel title lists the number
    of contributing members and a summary table reports per-year ensemble
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
    matplotlib.figure.Figure
        Proportion plot of ENSO phases (or ``None`` if no data).
    matplotlib.figure.Figure
        Caption for the proportion plot (or ``None`` if no data).
    pandas.DataFrame
        Per-year ENSO-phase proportions with chi-square test statistics.
    matplotlib.figure.Figure
        Mean NINO3.4 anomaly time series grouped by Y -1 phase (or ``None``).
    matplotlib.figure.Figure
        Caption for the initial-phase figure (or ``None``).
    pandas.DataFrame
        Relative-year mean NINO3.4 anomalies for each Y -1 phase.
    pandas.DataFrame
        Counts of Y -1 ENSO phases for each start month.
    pandas.DataFrame
        Summary statistics of per-year means and standard deviations by start month.
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
        nrows,
        ncols,
        figsize=(4.5 * ncols, 3.2 * nrows),
        sharex=True,
        sharey=True,
    )
    axes = axes.ravel()

    year_stats: dict[str, dict[str, float]] = {}
    month_stats: dict[str, dict[str, dict[str, float]]] = {}
    baseline_vals: list[float] | None = None
    pval_labels: list[str] = []
    pvals: list[float] = []
    axes_dict: dict[str, plt.Axes] = {}
    phase_records: list[dict[str, float | str]] = []
    phase_caption_fig: plt.Figure | None = None
    init_caption_fig: plt.Figure | None = None

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
            seasonal_da = djf.mean(dim="time")
            seasonal = seasonal_da.values
            member_ids = (
                seasonal_da["member"].values
                if "member" in seasonal_da.coords
                else np.arange(len(seasonal))
            )
            if len(seasonal) == 0 or np.allclose(seasonal, seasonal[0]):
                continue
            year_vals.extend(seasonal.tolist())
            phases = np.where(
                seasonal > 0.5,
                "El Niño",
                np.where(seasonal < -0.5, "La Niña", "Neutral"),
            )
            for m_id, val, phase in zip(member_ids, seasonal, phases):
                phase_records.append(
                    {
                        "rel_year": yr_label,
                        "start_month": month_label,
                        "member": int(m_id),
                        "value": float(val),
                        "phase": phase,
                    }
                )
            phase_counts = {
                "El Niño": int(np.sum(phases == "El Niño")),
                "La Niña": int(np.sum(phases == "La Niña")),
                "Neutral": int(np.sum(phases == "Neutral")),
            }
            month_stats[yr_label][month_label] = {
                "n": len(seasonal),
                "mean": float(np.mean(seasonal)),
                "std": float(np.std(seasonal, ddof=1)),
                "phase_counts": phase_counts,
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
            ax.set_title(f"{yr_label}\n(n={len(year_vals)})", fontsize=10)
        else:
            ax.set_title(yr_label, fontsize=10)
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

    # Hide any unused axes and tidy tick labels
    for ax in axes[n_years:]:
        ax.set_visible(False)
    for ax in axes:
        ax.tick_params(direction="out", labelsize=9)
        ax.label_outer()

    if phase_records:
        stats_df = pd.DataFrame(phase_records)
        initial_phase_map = (
            stats_df[stats_df["rel_year"] == "Y -1"][["start_month", "member", "phase"]]
            .rename(columns={"phase": "initial_phase"})
        )
        stats_df = stats_df.merge(initial_phase_map, on=["start_month", "member"], how="left")
        init_phase_counts_by_month = (
            stats_df[stats_df["rel_year"] == "Y -1"]
            .groupby("start_month")["phase"]
            .value_counts()
            .unstack(fill_value=0)
        )
        init_phase_ts = (
            stats_df.dropna(subset=["initial_phase"])
            .groupby(["rel_year", "initial_phase"])["value"]
            .mean()
            .unstack("initial_phase")
        )
        init_phase_ts = init_phase_ts.reindex(years.keys())
        if not init_phase_ts.empty:
            init_phase_fig, init_phase_ax = plt.subplots(figsize=(6, 4))
            x_ts = np.arange(len(init_phase_ts.index))
            for phase in init_phase_ts.columns:
                init_phase_ax.plot(x_ts, init_phase_ts[phase], marker="o", label=phase)
            init_phase_ax.set_xticks(x_ts)
            init_phase_ax.set_xticklabels(init_phase_ts.index)
            init_phase_ax.set_ylabel("Mean DJF NINO3.4 Anomaly (°C)", fontsize=10)
            init_phase_ax.set_xlabel("Relative Year", fontsize=10)
            init_phase_ax.legend(title="Y -1 Phase", fontsize=9, title_fontsize=9)
            init_phase_ax.tick_params(direction="out", labelsize=9)
            month_parts = []
            for m in init_phase_counts_by_month.index:
                counts = init_phase_counts_by_month.loc[m]
                month_parts.append(
                    f"{m[:3]}: El Niño {counts.get('El Niño', 0)}, "
                    f"Neutral {counts.get('Neutral', 0)}, "
                    f"La Niña {counts.get('La Niña', 0)}"
                )
            count_str = "; ".join(month_parts)
            init_caption = (
                "Mean DJF NINO3.4 anomalies grouped by Y -1 ENSO phase "
                "(threshold ±0.5°C). " + count_str
            )
            init_phase_fig.tight_layout()
            init_caption_fig = _make_caption_fig(init_caption, init_phase_fig)
        else:
            init_phase_fig = None
            init_caption_fig = None
        phase_counts = (
            stats_df.groupby(["start_month", "rel_year", "phase"]).size().unstack("phase", fill_value=0)
        )
        year_labels = list(years.keys())
        phase_results: list[pd.DataFrame] = []
        chi2_info: dict[str, pd.DataFrame] = {}
        phase_fig, phase_axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
        axes_map = dict(zip(start_months.keys(), phase_axes.flatten()))
        phase_order: list[str] | None = None
        for sm, sm_counts in phase_counts.groupby(level=0):
            sm_counts = sm_counts.droplevel("start_month")
            sm_counts = sm_counts.reindex(year_labels, fill_value=0)
            sm_counts = sm_counts.loc[sm_counts.sum(axis=1) > 0]
            sm_props = sm_counts.div(sm_counts.sum(axis=1), axis=0)
            if phase_order is None:
                phase_order = list(sm_props.columns)
            chi2_results = []
            if "Y -1" in sm_counts.index:
                baseline = sm_counts.loc["Y -1"]
                for yr in sm_counts.index:
                    if yr == "Y -1":
                        continue
                    contingency = np.vstack([sm_counts.loc[yr], baseline])
                    chi2, p, _, _ = chi2_contingency(contingency)
                    chi2_results.append(
                        {"rel_year": yr, "chi2": float(chi2), "pval": float(p), "sig": _p_to_star(p)}
                    )
            chi2_df = pd.DataFrame(chi2_results).set_index("rel_year")
            chi2_info[sm] = chi2_df

            ax = axes_map[sm]
            x = np.arange(len(sm_props.index))
            for phase in sm_props.columns:
                ax.plot(x, sm_props[phase], marker="o", label=phase)
            ax.set_xticks(x)
            ax.set_xticklabels(sm_props.index)
            if sm in ("January", "April"):
                ax.set_ylabel("Proportion", fontsize=10)
            ax.set_title(sm, fontsize=10)
            ax.tick_params(direction="out", labelsize=9)
            for xi, yr in zip(x, sm_props.index):
                if yr in chi2_df.index:
                    sig = chi2_df.loc[yr, "sig"]
                    if sig:
                        ax.text(xi, 1.01, sig, ha="center", va="bottom", fontsize=9)

            sm_df = pd.concat(
                [sm_props.add_suffix("_prop"), sm_counts.add_suffix("_count"), chi2_df],
                axis=1,
            )
            sm_df.index = pd.MultiIndex.from_product([[sm], sm_df.index], names=["start_month", "rel_year"])
            phase_results.append(sm_df)

        for ax in phase_axes[1, :]:
            ax.set_xlabel("Relative Year", fontsize=10)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        handles = [Line2D([0], [0], color=colors[i], lw=2, label=phase_order[i]) for i in range(len(phase_order or []))]
        phase_fig.legend(
            handles=handles,
            loc="upper center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, 0.97),
            fontsize=9,
            title="ENSO Phase",
            title_fontsize=9,
        )
        phase_fig.tight_layout(rect=[0, 0, 1, 0.95])

        caption_parts = []
        for sm, df in chi2_info.items():
            if df.empty:
                continue
            parts = [
                f"{yr}: χ²={df.loc[yr, 'chi2']:.2f}, p={df.loc[yr, 'pval']:.3f}{df.loc[yr, 'sig']}"
                for yr in df.index
            ]
            caption_parts.append(f"{sm} - " + ", ".join(parts))
        phase_caption = (
            "Proportion of ENSO phases by relative year and start month. Classifications use ±0.5°C thresholds; "
            "asterisks mark chi-square differences from Y -1. " + "; ".join(caption_parts)
        )
        phase_caption_fig = _make_caption_fig(phase_caption, phase_fig)

        results_df = pd.concat(phase_results)
    else:
        results_df = pd.DataFrame()
        phase_fig = None
        phase_caption_fig = None
        init_phase_fig = None
        init_caption_fig = None
        init_phase_ts = pd.DataFrame()
        init_phase_counts_by_month = pd.DataFrame()

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
        fontsize=9,
        title_fontsize=9,
    )

    fig.text(0.5, 0.04, "DJF NINO3.4 Anomaly (°C)", ha="center", fontsize=12)
    fig.text(
        0.04,
        0.5,
        "Probability Density",
        va="center",
        rotation="vertical",
        fontsize=12,
    )

    # dynamic title
    fig.suptitle(
        f"DJF NINO3.4 Distributions Relative to {eruption_year}",
        y=0.98,
        fontsize=14,
    )

    summary_records: list[dict[str, float | str]] = []
    for yr in years:
        if yr not in month_stats:
            continue
        months = month_stats.get(yr, {})
        for m, v in months.items():
            rec: dict[str, float | str] = {
                "rel_year": yr,
                "start_month": m,
                "mean": v["mean"],
                "std": v["std"],
                "n": v["n"],
            }
            if yr in year_stats and "p_adj" in year_stats[yr]:
                rec["p_adj"] = year_stats[yr]["p_adj"]
                rec["sig"] = year_stats[yr]["sig"]
            summary_records.append(rec)
    summary_df = pd.DataFrame(summary_records)

    fig.tight_layout(rect=[0, 0.07, 1, 0.92])
    return (
        fig,
        phase_fig,
        phase_caption_fig,
        results_df,
        init_phase_fig,
        init_caption_fig,
        init_phase_ts,
        init_phase_counts_by_month,
        summary_df,
    )


def plot_yearly_kde_panel(
    jan_nino34: xr.DataArray,
    apr_nino34: xr.DataArray,
    july_nino34: xr.DataArray,
    oct_nino34: xr.DataArray,
    eruption_year: int = 1258,
) -> tuple[
    plt.Figure,
    plt.Figure | None,
    plt.Figure | None,
    pd.DataFrame,
    plt.Figure | None,
    plt.Figure | None,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    with sns.axes_style("whitegrid"), sns.plotting_context("paper", font_scale=1.2):
        return _plot_yearly_kde_panel_impl(
            jan_nino34, apr_nino34, july_nino34, oct_nino34, eruption_year
        )


plot_yearly_kde_panel.__doc__ = _plot_yearly_kde_panel_impl.__doc__


if __name__ == "__main__":
    pass
