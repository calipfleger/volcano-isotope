"""Utilities for computing ENSO precipitation composites and significance.

This module reproduces the workflow of the `Composites_Significance.ipynb`
notebook. It provides functions to classify ENSO phase from a NINO3.4
index, compute precipitation anomaly composites for each phase, and assess
El Niño vs. Neutral differences with a Welch t-test.  Phase counts are also
returned so that composite plots can report the number of contributing
months for each ENSO category.

Example
-------
>>> nino = xr.open_dataset(nino_path)["sst"]
>>> precip = xr.open_dataset(precip_path)["precip"]
>>> el, la, ne = classify_enso_phase(nino)
>>> anoms = precip_anomalies(precip)
>>> comps, counts = composite_precip(anoms, el, la, ne)
>>> fig = plot_composites(comps, counts)
>>> diff, pvals = difference_and_significance(anoms, el, ne)
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point

PHASE_THRESH = 1.0


def classify_enso_phase(nino34: xr.DataArray, thresh: float = PHASE_THRESH) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Return boolean masks for El Niño, La Niña and Neutral months."""
    el = nino34.where(nino34 >= thresh, drop=True)
    la = nino34.where(nino34 <= -thresh, drop=True)
    ne = nino34.where((nino34 > -thresh) & (nino34 < thresh), drop=True)
    return el.time, la.time, ne.time


def precip_anomalies(precip: xr.DataArray) -> xr.DataArray:
    """Compute monthly anomalies from a precipitation record."""
    climo = precip.groupby("time.month").mean("time")
    return precip.groupby("time.month") - climo


def composite_precip(
    anoms: xr.DataArray,
    el_dates: xr.DataArray,
    la_dates: xr.DataArray,
    ne_dates: xr.DataArray,
) -> tuple[dict[str, xr.DataArray], dict[str, int]]:
    """Average anomalies for each ENSO phase and return sample counts."""
    comps = {
        "El Nino": anoms.sel(time=el_dates).mean("time"),
        "La Nina": anoms.sel(time=la_dates).mean("time"),
        "Neutral": anoms.sel(time=ne_dates).mean("time"),
    }
    counts = {
        "El Nino": int(el_dates.size),
        "La Nina": int(la_dates.size),
        "Neutral": int(ne_dates.size),
    }
    return comps, counts


def difference_and_significance(anoms: xr.DataArray, el_dates: xr.DataArray, ne_dates: xr.DataArray, p: float = 0.05) -> tuple[xr.DataArray, xr.DataArray]:
    """Difference between El Niño and Neutral composites and Welch t-test p-values."""
    el_vals = anoms.sel(time=el_dates)
    ne_vals = anoms.sel(time=ne_dates)
    diff = el_vals.mean("time") - ne_vals.mean("time")
    _, pvals = ttest_ind(el_vals, ne_vals, axis=0, equal_var=False)
    sig_mask = xr.where(pvals < p, 1, 0)
    return diff, sig_mask


def plot_composites(comps: dict[str, xr.DataArray], counts: dict[str, int]) -> plt.Figure:
    """Plot composites for each phase with sample counts."""
    labels = list(comps.keys())
    clevs = np.arange(-0.6, 0.7, 0.1)
    fig, axs = plt.subplots(nrows=3, ncols=1, subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(8.5, 11))
    for ax, label in zip(axs.flat, labels):
        data = comps[label]
        data, lons = add_cyclic_point(data, coord=data["lon"])
        cs = ax.contourf(lons, data["lat"], data, clevs, transform=ccrs.PlateCarree(), cmap="BrBG", extend="both")
        ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
        ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
        ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
        ax.set_title(f"{label} ({counts.get(label, 0)})")
        ax.coastlines()
    fig.subplots_adjust(bottom=0.25, top=0.9, left=0.05, right=0.95, wspace=0.1, hspace=0.5)
    cbar_ax = fig.add_axes([0.2, 0.2, 0.6, 0.02])
    fig.colorbar(cs, cax=cbar_ax, orientation="horizontal", label="mm/day")
    return fig


def plot_difference(diff: xr.DataArray, sig_mask: xr.DataArray) -> plt.Figure:
    """Plot El Niño − Neutral differences with significance hatching."""
    clevs = np.arange(-3, 3.5, 0.5)
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    data, lons = add_cyclic_point(diff, coord=diff["lon"])
    mask, _ = add_cyclic_point(sig_mask, coord=diff["lon"])
    cs = ax.contourf(lons, diff["lat"], data, clevs, transform=ccrs.PlateCarree(), cmap="BrBG", extend="both")
    ax.contourf(lons, diff["lat"], mask, [0, 1], transform=ccrs.PlateCarree(), colors="none", hatches=[".", ""], extend="both", alpha=0)
    ax.coastlines()
    ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    cbar_ax = fig.add_axes([0.2, 0.2, 0.6, 0.02])
    fig.colorbar(cs, cax=cbar_ax, orientation="horizontal", label="mm/day")
    ax.set_title("Composite Precipitation Differences El Nino-Neutral")
    return fig


__all__ = [
    "classify_enso_phase",
    "precip_anomalies",
    "composite_precip",
    "difference_and_significance",
    "plot_composites",
    "plot_difference",
]
