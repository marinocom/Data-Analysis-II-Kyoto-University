"""
descriptive.py

This module provides functions to display descriptive statistics and
visualizations for the flight data.
"""

from IPython.display import display, Markdown
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd


# -----------------------------
# Utility Functions
# -----------------------------
def display_title(s, pref='Figure', num=1, center=False):
    ctag = 'center' if center else 'p'
    s    = f'<{ctag}><span style="font-size: 1.2em;"><b>{pref} {num}</b>: {s}</span></{ctag}>'
    if pref == 'Figure':
        s = f'{s}<br><br>'
    else:
        s = f'<br><br>{s}'
    display(Markdown(s))


def central(x):
    """Return mean, median, mode ignoring NaNs."""
    mean = np.nanmean(x)
    median = np.nanmedian(x)
    mode = stats.mode(x, nan_policy='omit').mode
    return mean, median, mode


def dispersion(x):
    """Return std, min, max, range, 25th, 75th, IQR ignoring NaNs."""
    std = np.nanstd(x)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    rng = mx - mn
    p25 = np.nanpercentile(x, 25)
    p75 = np.nanpercentile(x, 75)
    iqr = p75 - p25
    return std, mn, mx, rng, p25, p75, iqr


def corrcoeff(x, y):
    """Compute correlation, ignoring NaN."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    return np.corrcoef(x[mask], y[mask])[0, 1]


def plot_regression_line(ax, x, y, **kwargs):
    """Plot regression line on axis."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]
    a, b = np.polyfit(x_clean, y_clean, deg=1)
    x0, x1 = np.min(x_clean), np.max(x_clean)
    ax.plot([x0, x1], [a * x0 + b, a * x1 + b], **kwargs)


# -----------------------------
# Table Display Functions
# -----------------------------

def display_central_tendency_table(df, num=1):
    """Display central tendency summary statistics."""
    display_title('Central tendency summary statistics.', pref='Table', num=num)
    df_numeric = df.select_dtypes(include=['number'])
    df_central = df_numeric.apply(lambda x: central(x), axis=0)

    df_central.index = ['mean', 'median', 'mode']
    display(df_central)


def display_dispersion_table(df, num=1):
    """Display dispersion summary statistics."""
    display_title('Dispersion summary statistics.', pref='Table', num=num)
    round_dict = {'arrival_delay': 2, 'departure_delay': 2, 'day_of_week': 2, 
                  'scheduled_dep_time': 0, 'distance': 1, 'month': 2}
    df_numeric = df.select_dtypes(include=['number'])
    df_disp = df_numeric.apply(lambda x: dispersion(x), axis=0).round(round_dict)

    df_disp.index = ['st.dev.', 'min', 'max', 'range', '25th', '75th', 'IQR']
    display(df_disp)


# -----------------------------
# Main Descriptive Plot Function
# -----------------------------

def plot_descriptive(df, general_correlation_panels=False, arrival_delay_panels=False, combined_multi_panel_figure=True):
    """
    Reproduces descriptive plots from FP-3 using module structure.
    Control which panels are displayed using flags.
    """

    # Extract series
    arr_delay = df['arrival_delay'].values
    dep_delay = df['departure_delay'].values
    day       = df['day_of_week'].values
    sched     = df['scheduled_dep_time'].values
    dist      = df['distance'].values
    month     = df['month'].values

    sched_hours = np.around(sched / 100, 1)

    # -----------------------------
    # 1. General Correlation Panels
    # -----------------------------
    if general_correlation_panels:

        fig, axs = plt.subplots(2, 3, figsize=(15, 8), tight_layout=True)
        axs = axs.flatten()

        ivs = [dep_delay, day, sched_hours, dist, month]
        colors = ['b', 'orange', 'g', 'r', 'purple']
        labels = [
            'Departure Delay (min)',
            'Day of Week',
            'Scheduled Departure (hours)',
            'Distance (miles)',
            'Month'
        ]

        for ax, x, c, lab in zip(axs[:5], ivs, colors, labels):
            ax.scatter(x, arr_delay, alpha=0.5, color=c)
            plot_regression_line(ax, x, arr_delay, color='k', ls='-', lw=2)
            r = corrcoeff(x, arr_delay)
            ax.text(0.7, 0.3, f'r = {r:.3f}', color=c,
                    transform=ax.transAxes,
                    bbox=dict(color='0.8', alpha=0.7))
            ax.set_xlabel(lab)

        axs[0].set_ylabel('Arrival Delay (min)')
        axs[3].set_ylabel('Arrival Delay (min)')
        axs[2].set_xticks([6, 9, 12, 15, 18, 21])
        axs[1].set_xticks([1, 2, 3, 4, 5, 6, 7])
        axs[4].set_xticks(range(1, 13))
        axs[5].axis('off')

        plt.show()

    # -----------------------------
    # 2. Early vs Late Arrival Panels
    # -----------------------------
    if arrival_delay_panels:

        i_low = arr_delay <= 0
        i_high = arr_delay > 0

        fig, axs = plt.subplots(1, 2, figsize=(8, 3), tight_layout=True)
        groups = [(i_low, 'On-time/Early Arrivals (≤0 min)'),
                  (i_high, 'Late Arrivals (>0 min)')]

        for ax, (mask, title) in zip(axs, groups):
            ax.scatter(dep_delay[mask], arr_delay[mask], alpha=0.5, color='b')
            plot_regression_line(ax, dep_delay[mask], arr_delay[mask], color='k')
            r = corrcoeff(dep_delay[mask], arr_delay[mask])
            ax.text(0.7, 0.3, f'r = {r:.3f}', transform=ax.transAxes,
                    bbox=dict(color='0.8', alpha=0.7))
            ax.set_title(title)
            ax.set_xlabel('Departure Delay (min)')

        axs[0].set_ylabel('Arrival Delay (min)')
        plt.show()

    # -----------------------------
    # 3. Final Combined Multi-panel Figure (FP-3)
    # -----------------------------
    if combined_multi_panel_figure:

        i_low = arr_delay <= 0
        i_high = arr_delay > 0

        fig, axs = plt.subplots(2, 3, figsize=(12, 6), tight_layout=True)
        axs = axs.flatten()

        ivs2 = [day, sched_hours, dist, month]
        colors2 = ['orange', 'g', 'r', 'purple']
        labels2 = [
            'Day of Week', 'Scheduled Departure (hours)',
            'Distance (miles)', 'Month'
        ]

        for ax, x, c, lab in zip(axs[:4], ivs2, colors2, labels2):
            ax.scatter(x, arr_delay, alpha=0.5, color=c)
            plot_regression_line(ax, x, arr_delay, color='k')
            r = corrcoeff(x, arr_delay)
            ax.text(0.7, 0.3, f'r = {r:.3f}', color=c,
                    transform=ax.transAxes,
                    bbox=dict(color='0.8', alpha=0.7))
            ax.set_xlabel(lab)

        axs[1].set_xticks([6, 9, 12, 15, 18, 21])
        axs[0].set_xticks([1, 2, 3, 4, 5, 6, 7])
        axs[3].set_xticks(range(1, 13))

        axs[0].set_ylabel('Arrival Delay (min)')
        axs[3].set_ylabel('Arrival Delay (min)')

        # Panel (e): early vs late with mean markers
        ax = axs[4]

        fcolors = ['m', 'c']
        labels_group = ['On-time/Early', 'Late']
        delay_bins = [[-60, -40, -20, 0], [500, 1000, 1500, 2000]]
        ylocs = [0.3, 0.7]

        for mask, c, lbl, bins, yloc in zip(
            [i_low, i_high], fcolors, labels_group, delay_bins, ylocs
        ):
            ax.scatter(dep_delay[mask], arr_delay[mask], alpha=0.5, color=c, label=lbl)
            plot_regression_line(ax, dep_delay[mask], arr_delay[mask], color=c)

            # Mean markers
            for b in bins:
                vals = dep_delay[(arr_delay >= b - 50) &
                 (arr_delay < b + 50) & mask]

                if len(vals) > 0:          # only compute mean if non-empty
                    mean_val = vals.mean()
                    ax.plot(mean_val, b, 'o', color=c, mfc='w', ms=10)

            r = corrcoeff(dep_delay[mask], arr_delay[mask])
            ax.text(0.7, yloc, f'r = {r:.3f}', color=c,
                    transform=ax.transAxes,
                    bbox=dict(color='0.8', alpha=0.7))

        ax.set_xlabel('Departure Delay (min)')
        ax.legend()
        axs[5].axis('off')

        # Panel labels
        for ax, s in zip(axs[:5], ['a', 'b', 'c', 'd', 'e']):
            ax.text(0.02, 0.92, f'({s})', size=12, transform=ax.transAxes)

        plt.show()

        display_title('Correlations amongst main variables.', pref='Figure', num=1)