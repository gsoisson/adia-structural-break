import numpy as np
import matplotlib.pyplot as plt


def plot_timeseries_with_break(
    df, series_id, y_labels=None, robust_ylim=True, clip_low=0.5, clip_high=99.5
):
    """
    Plot a time series with a vertical red dotted line at the break point and guard against y-axis squashing.
    """

    # Extract data for the specific series
    series_data = df.loc[series_id].reset_index()
    t = series_data["time"].to_numpy()
    y = series_data["value"].to_numpy()

    # Find the break point (first occurrence of period=1)
    break_mask = series_data["period"] == 1
    break_point = (
        series_data.loc[break_mask, "time"].iloc[0] if break_mask.any() else None
    )

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, y, "b-", alpha=0.85, label="value")

    # Add vertical line at break point
    if break_point is not None:
        ax.axvline(
            x=break_point,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Break point (t={break_point})",
        )

    # Title / labels
    title = f"Time Series ID: {series_id}"
    if y_labels is not None:
        label_text = "Break Detected" if y_labels.loc[series_id] else "No Break"
        title += f" - Label: {label_text}"
    ax.set_title(title)

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)

    # --- Robust y-limits (percentile window) ---
    if robust_ylim and np.isfinite(y).any():
        lo, hi = np.nanpercentile(y, [clip_low, clip_high])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            pad = 0.08 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)

    ax.legend(loc="best")
    plt.tight_layout()
