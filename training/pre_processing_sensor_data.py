import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timezone, timedelta

base_folder = "/Users/supriya/Documents/data/agentic_system/multi-agent-manufacturing-inspection/training/signals"

bead_id = "bead_1"
# bead_id = "bead_2"
# bead_id = "bead_3"
# bead_id = "bead_4"
# bead_id = "bead_5"
# bead_id = "bead_6"

vol_curr_file = f"{bead_id}_voltage_current.csv"
full_path = os.path.join(base_folder, vol_curr_file)

# Spike detection threshold multiplier (N × local σ)
SPIKE_N_SIGMA = 3.0

# Slow trend window for Row 4 derivative (seconds).
# Must be large enough to smooth out arc oscillations — 1.0 s works well.
trend_seconds = 1.0

# Per-frame lookback window
lookback_seconds = 0.50


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def padded(arr, pad=0.15):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return -1, 1
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    margin = (hi - lo) * pad
    if margin == 0:
        margin = pad if hi == 0 else abs(hi) * pad
    return lo - margin, hi + margin


def robust_ylim(arr, lower=2, upper=98, pad_frac=0.15,
                symmetric_about_zero=False, min_span=1e-6):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return -1, 1
    lo, hi = np.nanpercentile(arr, [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -1, 1
    if symmetric_about_zero:
        half = max(abs(lo), abs(hi), min_span) * (1.0 + pad_frac)
        return -half, half
    span = max(hi - lo, min_span)
    pad = span * pad_frac
    return lo - pad, hi + pad


def safe_gradient(y, t):
    """
    Compute dy/dt robustly.
    Uses time-based gradient when timestamps are valid and strictly increasing.
    Falls back to sample-index gradient otherwise.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(y) < 2:
        return np.zeros_like(y, dtype=float)
    dt = np.diff(t)
    if np.any(~np.isfinite(dt)) or np.any(dt <= 1e-9):
        return np.gradient(y)
    return np.gradient(y, t)


def compute_global_limits(df_sorted):
    """
    Global y-limits for Row 1 (raw+avg) and Row 2 (residual) only.
    Row 3 and Row 4 use local robust scaling inside plot_frame_signals.
    """
    v_raw  = df_sorted['voltage'].dropna()
    c_raw  = df_sorted['current'].dropna()
    v_roll = df_sorted['voltage_roll'].dropna()
    c_roll = df_sorted['current_roll'].dropna()

    v_resid = (df_sorted['voltage'] - df_sorted['voltage_roll']).dropna()
    c_resid = (df_sorted['current'] - df_sorted['current_roll']).dropna()

    return {
        'v_raw'  : padded(np.concatenate([v_raw.values,  v_roll.values])),
        'c_raw'  : padded(np.concatenate([c_raw.values,  c_roll.values])),
        'v_resid': padded(v_resid.values),
        'c_resid': padded(c_resid.values),
    }


def plot_frame_signals(
    history_slice,
    now_time,
    frame_filename,
    global_lims,
    output_dir,
    lookback_seconds,
    spike_n_sigma=3.0,
    trend_seconds=1.0,
):
    """
    Save two plots per thermal frame:
        {stem}_voltage_signals.png
        {stem}_current_signals.png

    Layout (4 rows, shared time axis):

        Row 1 — Raw + 0.05 s rolling average  [GLOBAL y-scale]
                 Shows the signal level the agent should compare across frames.

        Row 2 — Residual: raw − rolling avg   [GLOBAL y-scale]
                 Shows high-frequency noise character. Fill: blue = positive,
                 orange = negative.

        Row 3 — Spike events, local zoom       [LOCAL y-scale]
                 Raw + rolling avg + ±Nσ threshold bands + red spike markers.
                 σ is computed ONLY on the active (non-zero raw) portion of the
                 residual so that arc-extinction zero-crossings are NOT counted
                 as spikes. Spike count is printed in the title.

        Row 4 — dX/dt of slow trend            [LOCAL y-scale, symmetric]
                 The derivative is taken from the pre-computed 'voltage_trend' /
                 'current_trend' column (computed on the FULL df with a wide
                 window), then sliced to the lookback window. This avoids the
                 edge-effect noise that occurs when gradient is computed on a
                 short 0.5 s slice.
                 A text label "RISING / STABLE / FALLING" is printed so the
                 agent gets an unambiguous reading even if the line is noisy.
    """

    t      = history_slice['time'].values
    v_raw  = history_slice['voltage'].values
    c_raw  = history_slice['current'].values

    v_roll  = history_slice['voltage_roll'].interpolate().ffill().bfill().values
    c_roll  = history_slice['current_roll'].interpolate().ffill().bfill().values

    # Row 4: gradient already computed across full df, just slice here
    v_trend_d = history_slice['v_trend_deriv'].interpolate().ffill().bfill().values
    c_trend_d = history_slice['c_trend_deriv'].interpolate().ffill().bfill().values

    v_resid = v_raw - v_roll
    c_resid = c_raw - c_roll

    # ── Event classification — two physically distinct event types ───────────
    #
    # TYPE 1 — ENERGY SPIKE (▲ red)
    #   Raw signal surges above rolling_avg + N×σ (positive excursion only).
    #   Physical cause: momentary arc energy burst, spatter event.
    #   σ is computed ONLY on active-signal samples so arc-extinction zeros
    #   do not inflate it and lower the threshold artifically.
    #
    # TYPE 2 — ARC EXTINCTION (▼ blue)
    #   Raw signal collapses below 20 % of the rolling median — absolute floor,
    #   independent of the residual threshold.
    #   Physical cause: arc momentarily extinguishes (short circuit or gap loss).

    def _classify_events(raw, resid, roll, n_sigma):
        roll_med = np.nanmedian(roll[roll > 0]) if np.any(roll > 0) else 1.0
        active_mask   = raw > 0.10 * roll_med          # exclude near-zero samples from σ
        active_resid  = resid[active_mask]
        sigma  = np.nanstd(active_resid) if len(active_resid) >= 5 else np.nanstd(resid)
        thresh = n_sigma * sigma if (np.isfinite(sigma) and sigma > 0) else np.nan

        # Type 1: positive energy spike only (raw exceeds rolling_avg + threshold)
        spike_mask = (resid > thresh) & active_mask if np.isfinite(thresh) else np.zeros(len(raw), dtype=bool)

        # Type 2: arc extinction (raw collapses below 20 % of rolling median)
        extinction_mask = raw < (0.20 * roll_med)

        return sigma, thresh, spike_mask.astype(bool), extinction_mask.astype(bool)

    v_sigma, v_thresh, v_spike_mask, v_extinct_mask = _classify_events(
        v_raw, v_resid, v_roll, spike_n_sigma)
    c_sigma, c_thresh, c_spike_mask, c_extinct_mask = _classify_events(
        c_raw, c_resid, c_roll, spike_n_sigma)

    v_spike_count   = int(v_spike_mask.sum())
    c_spike_count   = int(c_spike_mask.sum())
    v_extinct_count = int(v_extinct_mask.sum())
    c_extinct_count = int(c_extinct_mask.sum())

    # ── Trend direction label ─────────────────────────────────────────────────
    # Median derivative over last 20 % of the window for stability
    tail = max(1, len(v_trend_d) // 5)

    def _trend_label(deriv_tail, unit, stable_band=0.5):
        med = np.nanmedian(deriv_tail)
        if not np.isfinite(med):
            return 'UNKNOWN', '#888888'
        if med > stable_band:
            return f'▲ RISING  ({med:+.2f} {unit})', '#1a7a3a'
        elif med < -stable_band:
            return f'▼ FALLING  ({med:+.2f} {unit})', '#c0392b'
        else:
            return f'● STABLE  ({med:+.2f} {unit})', '#555555'

    v_trend_label, v_trend_color = _trend_label(v_trend_d[-tail:], 'V/s')
    c_trend_label, c_trend_color = _trend_label(c_trend_d[-tail:], 'A/s')

    # ── Local y-limits ────────────────────────────────────────────────────────
    # y-limits include spike points, extinction points, and the +Nσ band only
    v_extras = []
    if v_spike_count > 0:   v_extras.append(v_raw[v_spike_mask])
    if v_extinct_count > 0: v_extras.append(v_raw[v_extinct_mask])
    v_spike_view = np.concatenate(
        [v_roll, v_roll + v_thresh] + v_extras if np.isfinite(v_thresh)
        else [v_roll] + v_extras if v_extras else [v_roll, v_raw]
    )

    c_extras = []
    if c_spike_count > 0:   c_extras.append(c_raw[c_spike_mask])
    if c_extinct_count > 0: c_extras.append(c_raw[c_extinct_mask])
    c_spike_view = np.concatenate(
        [c_roll, c_roll + c_thresh] + c_extras if np.isfinite(c_thresh)
        else [c_roll] + c_extras if c_extras else [c_roll, c_raw]
    )

    v_spike_ylim = robust_ylim(v_spike_view, lower=1, upper=99, pad_frac=0.12)
    c_spike_ylim = robust_ylim(c_spike_view, lower=1, upper=99, pad_frac=0.12)

    v_deriv_ylim = robust_ylim(v_trend_d, lower=2, upper=98, pad_frac=0.20,
                                symmetric_about_zero=True, min_span=0.5)
    c_deriv_ylim = robust_ylim(c_trend_d, lower=2, upper=98, pad_frac=0.20,
                                symmetric_about_zero=True, min_span=0.5)

    # ── "Now" values ──────────────────────────────────────────────────────────
    now_v       = v_roll[-1]
    now_c       = c_roll[-1]
    now_v_raw   = v_raw[-1]
    now_c_raw   = c_raw[-1]
    now_vr      = v_resid[-1]
    now_cr      = c_resid[-1]
    now_v_td    = v_trend_d[-1]
    now_c_td    = c_trend_d[-1]

    stem = os.path.splitext(frame_filename)[0]

    # ── Plot style constants ──────────────────────────────────────────────────
    TITLE_FS = 11
    LABEL_FS = 10
    TICK_FS  = 9
    ANNOT_FS = 9
    right_pad = (t[-1] - t[0]) * 0.05 if len(t) > 1 else 0.01
    x_lim = (t[0], t[-1] + right_pad)

    def _style_ax(ax, ylabel, ylim, xlim):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_ylabel(ylabel, fontsize=LABEL_FS, labelpad=4)
        ax.tick_params(axis='both', labelsize=TICK_FS)
        ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.55, color='#cccccc')
        ax.spines[['top', 'right']].set_visible(False)

    def _now_marker(ax):
        ax.axvline(x=now_time, color='#222222', linewidth=1.2,
                   linestyle='--', alpha=0.75, zorder=5, label='Frame time')

    def _annotate_value(ax, x, y, text, color):
        ax.annotate(
            text, xy=(x, y), xytext=(6, 0), textcoords='offset points',
            fontsize=ANNOT_FS, fontweight='bold', color=color,
            va='center', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8)
        )

    # ═════════════════════════════════════════════════════════════════════════
    # FIGURE A — VOLTAGE
    # ═════════════════════════════════════════════════════════════════════════
    fig_v, axes_v = plt.subplots(
        4, 1, figsize=(12, 13),
        gridspec_kw={'height_ratios': [3, 2, 2, 2], 'hspace': 0.48}
    )
    fig_v.suptitle(
        f'VOLTAGE SIGNAL ANALYSIS\n'
        f'Frame: {frame_filename}   |   t = {now_time:.4f} s   |   '
        f'Lookback: {lookback_seconds:.2f} s',
        fontsize=TITLE_FS + 1, fontweight='bold', y=0.985
    )

    # Row 1 ── Raw + rolling avg ───────────────────────────────────────────────
    ax = axes_v[0]
    ax.plot(t, v_raw,  color='#90b8e0', linewidth=0.8, alpha=0.6, label='Raw voltage')
    ax.plot(t, v_roll, color='#1a5fa8', linewidth=2.0,             label='Rolling avg (0.05 s)')
    _now_marker(ax)
    ax.plot(now_time, now_v, 'o', color='#1a5fa8', markersize=7, zorder=6)
    _annotate_value(ax, now_time, now_v, f'{now_v:.3f} V', '#1a5fa8')
    ax.annotate(f'raw: {now_v_raw:.3f} V', xy=(now_time, now_v_raw),
                xytext=(6, -12), textcoords='offset points',
                fontsize=ANNOT_FS - 1, color='#90b8e0',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))
    _style_ax(ax, 'Voltage (V)', global_lims['v_raw'], x_lim)
    ax.set_title('Raw signal & 0.05 s rolling average  [global scale]',
                 fontsize=TITLE_FS, pad=4)
    ax.legend(loc='upper left', fontsize=ANNOT_FS, framealpha=0.7,
              handlelength=1.4, borderpad=0.5)

    # Row 2 ── Residual ────────────────────────────────────────────────────────
    ax = axes_v[1]
    ax.fill_between(t, v_resid, 0, where=(v_resid >= 0),
                    color='#4a90d9', alpha=0.55, linewidth=0)
    ax.fill_between(t, v_resid, 0, where=(v_resid < 0),
                    color='#e07b54', alpha=0.55, linewidth=0)
    ax.plot(t, v_resid, color='#555555', linewidth=0.6, alpha=0.5)
    ax.axhline(0, color='#888888', linewidth=0.8)
    _now_marker(ax)
    ax.plot(now_time, now_vr, 'o', color='#555555', markersize=5, zorder=6)
    _annotate_value(ax, now_time, now_vr, f'{now_vr:+.3f} V', '#333333')
    _style_ax(ax, 'Residual (V)', global_lims['v_resid'], x_lim)
    ax.set_title('Residual (raw − rolling avg)  [global scale]',
                 fontsize=TITLE_FS, pad=4)

    # Row 3 ── Spike events ────────────────────────────────────────────────────
    # σ is computed on active-signal samples only (excludes arc extinctions ~0 V)
    ax = axes_v[2]
    ax.plot(t, v_raw,  color='#90b8e0', linewidth=0.8, alpha=0.45, label='Raw voltage')
    ax.plot(t, v_roll, color='#1a5fa8', linewidth=1.6, alpha=0.95, label='Rolling avg')
    # +Nσ band only — lower band removed (arc extinctions handled separately)
    if np.isfinite(v_thresh):
        ax.plot(t, v_roll + v_thresh, '--', color='orange', linewidth=1.2,
                alpha=0.9, label=f'+{spike_n_sigma:.0f}σ energy-spike threshold')
    # TYPE 1: energy spikes — red triangle-up
    # if v_spike_count > 0:
    #     ax.scatter(t[v_spike_mask], v_raw[v_spike_mask],
    #                color='#cc0000', marker='^', s=45, zorder=8,
    #                label=f'▲ Energy spike: {v_spike_count}')

    ax.scatter(t[v_spike_mask], v_raw[v_spike_mask],
           color='#cc0000', marker='^', s=18, alpha=0.85, zorder=8,
           label=f'▲ Energy spike: {v_spike_count}')
    
    # TYPE 2: arc extinctions — blue triangle-down
    # if v_extinct_count > 0:
    #     ax.scatter(t[v_extinct_mask], v_raw[v_extinct_mask],
    #                color='#1a5fa8', marker='v', s=6, zorder=7,
    #                label=f'▼ Arc extinction: {v_extinct_count}')

    if v_extinct_count > 0:
        ax.scatter(t[v_extinct_mask], v_raw[v_extinct_mask],
               color='#1a5fa8', marker='v', s=6, alpha=0.25, zorder=7,
               label=f'▼ Arc extinction: {v_extinct_count}')
    
    _now_marker(ax)
    _style_ax(ax, 'Voltage (V)', v_spike_ylim, x_lim)
    thresh_str = f'{v_thresh:.3f} V' if np.isfinite(v_thresh) else 'n/a'
    ax.set_title(
        f'Anomaly events  [local scale]  |  '
        f'σ(active)={v_sigma:.3f} V  |  spike thresh={thresh_str}  |  '
        f'▲ spikes={v_spike_count}  ▼ extinctions={v_extinct_count}',
        fontsize=TITLE_FS, pad=4
    )
    ax.legend(loc='upper left', fontsize=ANNOT_FS - 1, framealpha=0.7,
              handlelength=1.3, borderpad=0.4, ncol=2)

    # Row 4 ── Trend derivative ────────────────────────────────────────────────
    # Computed on full-df trend column (wide window), sliced here — no edge noise
    ax = axes_v[3]
    ax.plot(t, v_trend_d, color='#2aaa6e', linewidth=1.6,
            label=f'dV/dt  ({trend_seconds:.1f} s trend)')
    ax.fill_between(t, v_trend_d, 0, where=(v_trend_d >= 0),
                    color='#2aaa6e', alpha=0.18, linewidth=0)
    ax.fill_between(t, v_trend_d, 0, where=(v_trend_d < 0),
                    color='#c0392b', alpha=0.18, linewidth=0)
    ax.axhline(0, color='#888888', linewidth=0.8)
    _now_marker(ax)
    ax.plot(now_time, now_v_td, 'o', color='#2aaa6e', markersize=5, zorder=6)
    _annotate_value(ax, now_time, now_v_td, f'{now_v_td:+.4f} V/s', '#2aaa6e')
    # Text label — unambiguous reading for the vision agent
    ax.text(0.02, 0.88, v_trend_label,
            transform=ax.transAxes, fontsize=ANNOT_FS + 1, fontweight='bold',
            color=v_trend_color, va='top',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#cccccc', alpha=0.88))
    _style_ax(ax, 'Trend dV/dt (V/s)', v_deriv_ylim, x_lim)
    ax.set_title(
        f'dV/dt of {trend_seconds:.1f} s trend  [local scale, symmetric]',
        fontsize=TITLE_FS, pad=4
    )
    ax.set_xlabel(f'PerfTime (s)  [last {lookback_seconds:.2f} s shown]', fontsize=LABEL_FS)
    ax.legend(loc='upper right', fontsize=ANNOT_FS - 1, framealpha=0.7,
              handlelength=1.4, borderpad=0.4)

    out_v = os.path.join(output_dir, f'{stem}_voltage_signals.png')
    fig_v.savefig(out_v, dpi=150, bbox_inches='tight')
    plt.close(fig_v)

    # ═════════════════════════════════════════════════════════════════════════
    # FIGURE B — CURRENT  (identical structure)
    # ═════════════════════════════════════════════════════════════════════════
    fig_c, axes_c = plt.subplots(
        4, 1, figsize=(12, 13),
        gridspec_kw={'height_ratios': [3, 2, 2, 2], 'hspace': 0.48}
    )
    fig_c.suptitle(
        f'CURRENT SIGNAL ANALYSIS\n'
        f'Frame: {frame_filename}   |   t = {now_time:.4f} s   |   '
        f'Lookback: {lookback_seconds:.2f} s',
        fontsize=TITLE_FS + 1, fontweight='bold', y=0.985
    )

    # Row 1
    ax = axes_c[0]
    ax.plot(t, c_raw,  color='#e8a090', linewidth=0.8, alpha=0.6, label='Raw current')
    ax.plot(t, c_roll, color='#b22222', linewidth=2.0,             label='Rolling avg (0.05 s)')
    _now_marker(ax)
    ax.plot(now_time, now_c, 'o', color='#b22222', markersize=7, zorder=6)
    _annotate_value(ax, now_time, now_c, f'{now_c:.3f} A', '#b22222')
    ax.annotate(f'raw: {now_c_raw:.3f} A', xy=(now_time, now_c_raw),
                xytext=(6, -12), textcoords='offset points',
                fontsize=ANNOT_FS - 1, color='#e8a090',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))
    _style_ax(ax, 'Current (A)', global_lims['c_raw'], x_lim)
    ax.set_title('Row 1 — Raw signal and 0.05 s rolling average  [global scale]',
                 fontsize=TITLE_FS, pad=4)
    ax.legend(loc='upper left', fontsize=ANNOT_FS, framealpha=0.7,
              handlelength=1.4, borderpad=0.5)

    # Row 2
    ax = axes_c[1]
    ax.fill_between(t, c_resid, 0, where=(c_resid >= 0),
                    color='#e07b54', alpha=0.55, linewidth=0)
    ax.fill_between(t, c_resid, 0, where=(c_resid < 0),
                    color='#4a90d9', alpha=0.55, linewidth=0)
    ax.plot(t, c_resid, color='#555555', linewidth=0.6, alpha=0.5)
    ax.axhline(0, color='#888888', linewidth=0.8)
    _now_marker(ax)
    ax.plot(now_time, now_cr, 'o', color='#555555', markersize=5, zorder=6)
    _annotate_value(ax, now_time, now_cr, f'{now_cr:+.3f} A', '#333333')
    _style_ax(ax, 'Residual (A)', global_lims['c_resid'], x_lim)
    ax.set_title('Row 2 — Residual (raw − rolling avg)  [global scale]',
                 fontsize=TITLE_FS, pad=4)

    # Row 3
    ax = axes_c[2]
    ax.plot(t, c_raw,  color='#e8a090', linewidth=0.8, alpha=0.45, label='Raw current')
    ax.plot(t, c_roll, color='#b22222', linewidth=1.6, alpha=0.95, label='Rolling avg')
    # +Nσ band only — lower band removed (arc extinctions handled separately)
    if np.isfinite(c_thresh):
        ax.plot(t, c_roll + c_thresh, '--', color='orange', linewidth=1.2,
                alpha=0.9, label=f'+{spike_n_sigma:.0f}σ energy-spike threshold')
    # TYPE 1: energy spikes — red triangle-up
    if c_spike_count > 0:
        # ax.scatter(t[c_spike_mask], c_raw[c_spike_mask],
        #            color='#cc0000', marker='^', s=45, zorder=8,
        #            label=f'▲ Energy spike: {c_spike_count}')
        ax.scatter(t[c_spike_mask], c_raw[c_spike_mask],
           color='#cc0000', marker='^', s=18, alpha=0.85, zorder=8,
           label=f'▲ Energy spike: {c_spike_count}')
    # TYPE 2: arc extinctions — blue triangle-down
    if c_extinct_count > 0:
        ax.scatter(t[c_extinct_mask], c_raw[c_extinct_mask],
                   color='#1a5fa8', marker='v', s=6, alpha=0.25, zorder=7,
                   label=f'▼ Arc extinction: {c_extinct_count}')
    _now_marker(ax)
    _style_ax(ax, 'Current (A)', c_spike_ylim, x_lim)
    thresh_str_c = f'{c_thresh:.3f} A' if np.isfinite(c_thresh) else 'n/a'
    ax.set_title(
        f'Row 3 — Anomaly events  [local scale]  |  '
        f'σ(active)={c_sigma:.3f} A  |  spike thresh={thresh_str_c}  |  '
        f'▲ spikes={c_spike_count}  ▼ extinctions={c_extinct_count}',
        fontsize=TITLE_FS, pad=4
    )
    ax.legend(loc='upper left', fontsize=ANNOT_FS - 1, framealpha=0.7,
              handlelength=1.3, borderpad=0.4, ncol=2)

    # Row 4
    ax = axes_c[3]
    ax.plot(t, c_trend_d, color='#b8860b', linewidth=1.6,
            label=f'dI/dt  ({trend_seconds:.1f} s trend)')
    ax.fill_between(t, c_trend_d, 0, where=(c_trend_d >= 0),
                    color='#b8860b', alpha=0.18, linewidth=0)
    ax.fill_between(t, c_trend_d, 0, where=(c_trend_d < 0),
                    color='#7b2d8b', alpha=0.18, linewidth=0)
    ax.axhline(0, color='#888888', linewidth=0.8)
    _now_marker(ax)
    ax.plot(now_time, now_c_td, 'o', color='#b8860b', markersize=5, zorder=6)
    _annotate_value(ax, now_time, now_c_td, f'{now_c_td:+.4f} A/s', '#b8860b')
    ax.text(0.02, 0.88, c_trend_label,
            transform=ax.transAxes, fontsize=ANNOT_FS + 1, fontweight='bold',
            color=c_trend_color, va='top',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#cccccc', alpha=0.88))
    _style_ax(ax, 'Trend dI/dt (A/s)', c_deriv_ylim, x_lim)
    ax.set_title(
        f'Row 4 — dI/dt of {trend_seconds:.1f} s trend  [local scale, symmetric]  — structural change',
        fontsize=TITLE_FS, pad=4
    )
    ax.set_xlabel(f'PerfTime (s)  [last {lookback_seconds:.2f} s shown]', fontsize=LABEL_FS)
    ax.legend(loc='upper right', fontsize=ANNOT_FS - 1, framealpha=0.7,
              handlelength=1.4, borderpad=0.4)

    out_c = os.path.join(output_dir, f'{stem}_current_signals.png')
    fig_c.savefig(out_c, dpi=150, bbox_inches='tight')
    plt.close(fig_c)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

try:
    df = pd.read_csv(full_path)

    columns_to_drop = ['Sample', 'VoltageRaw', 'CurrentRaw']
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])
    df = df.rename(columns={
        'Voltage(V)': 'voltage',
        'Current(A)': 'current',
        'PerfTime(s)': 'time'
    })

    df['voltage'] = df['voltage'] * 10
    df['current'] = df['current'] * 100

    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df.dropna(subset=['time']).reset_index(drop=True)

    active_rows = df[df['current'] > 10.0]
    plot_limit = active_rows['time'].max() + 2.0 if not active_rows.empty else df['time'].max()

    time_span = df['time'].iloc[-1] - df['time'].iloc[0]
    print("time_span:", time_span)

    if pd.isna(time_span) or time_span == 0:
        raise ValueError("Invalid time_span.")

    samples_per_sec = len(df) / time_span
    print("samples_per_sec:", samples_per_sec)

    window_05    = max(1, int(samples_per_sec * 0.05))
    window_small = max(1, int(samples_per_sec * 0.00125))
    window_trend = max(5, int(samples_per_sec * trend_seconds))

    print(f"window_05 = {window_05} samples  |  window_trend = {window_trend} samples")

    df['voltage_roll']  = df['voltage'].rolling(window=window_05,    center=True).mean()
    df['current_roll']  = df['current'].rolling(window=window_05,    center=True).mean()
    df['voltage_trend'] = df['voltage'].rolling(window=window_trend,  center=True).mean()
    df['current_trend'] = df['current'].rolling(window=window_trend,  center=True).mean()

    # ── KEY FIX: compute trend derivative on the FULL df now, before slicing ──
    # This prevents edge-effect noise that happens when gradient is computed
    # on the short 0.5 s per-frame slice.
    t_arr = df['time'].values
    df['v_trend_deriv'] = safe_gradient(
        df['voltage_trend'].ffill().bfill().values, t_arr
    )
    df['c_trend_deriv'] = safe_gradient(
        df['current_trend'].ffill().bfill().values, t_arr
    )

    # ── Existing Figure 1: full process overview ──────────────────────────────
    fig1, axs1 = plt.subplots(2, 2, figsize=(16, 10))

    def style_plot(ax, x, y, title, ylabel, color, is_raw=True, x_min=0, x_max=plot_limit):
        ax.plot(x, y, color=color,
                linewidth=0.5 if is_raw else 1.5,
                alpha=0.7 if is_raw else 1.0)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Time (s)')
        ax.set_xlim(x_min, x_max)
        ax.grid(True, linestyle='--', alpha=0.5)

    style_plot(axs1[0,0], df['time'], df['voltage'],      'Raw Voltage (x10)',          'Voltage (V)',     'blue')
    style_plot(axs1[0,1], df['time'], df['current'],      'Raw Current (x100)',          'Current (A)',     'red')
    style_plot(axs1[1,0], df['time'], df['voltage_roll'], 'Rolling Voltage Avg (0.05s)', 'Avg Voltage (V)', 'navy',    False)
    style_plot(axs1[1,1], df['time'], df['current_roll'], 'Rolling Current Avg (0.05s)', 'Avg Current (A)', 'darkred', False)
    fig1.tight_layout()
    fig1.savefig(f"{bead_id}_Figure_1_Full_Process.png", dpi=300)
    plt.close(fig1)

    # ── Existing Figure 2: excerpt ────────────────────────────────────────────
    start_excerpt, end_excerpt = 7.25, 9.25
    df_ex = df[(df['time'] >= start_excerpt) & (df['time'] <= end_excerpt)].copy()
    df_ex['vol_roll_ex'] = df_ex['voltage'].rolling(window=window_small, center=True).mean()
    df_ex['cur_roll_ex'] = df_ex['current'].rolling(window=window_small, center=True).mean()

    fig2, axs2 = plt.subplots(2, 2, figsize=(16, 10))
    style_plot(axs2[0,0], df_ex['time'], df_ex['voltage'],      'Voltage Excerpt (Raw)',        'V',     'blue',    x_min=start_excerpt, x_max=end_excerpt)
    style_plot(axs2[0,1], df_ex['time'], df_ex['current'],      'Current Excerpt (Raw)',        'A',     'red',     x_min=start_excerpt, x_max=end_excerpt)
    style_plot(axs2[1,0], df_ex['time'], df_ex['vol_roll_ex'],  'Rolling Voltage Avg (0.00125s)', 'Avg V', 'navy',  False, x_min=start_excerpt, x_max=end_excerpt)
    style_plot(axs2[1,1], df_ex['time'], df_ex['cur_roll_ex'],  'Rolling Current Avg (0.00125s)', 'Avg A', 'darkred', False, x_min=start_excerpt, x_max=end_excerpt)
    fig2.tight_layout()
    fig2.savefig(f"{bead_id}_Figure_2_Excerpt.png", dpi=300)
    plt.close(fig2)

    print(f"Overview plots saved in {os.getcwd()}")

    # ── Timestamp conversion ──────────────────────────────────────────────────
    abs_time = []
    for t_stamp in df['Timestamp']:
        parsed_dt = datetime.strptime((t_stamp[:-4] + ' UTC'), '%Y-%m-%d %H:%M:%S.%f %Z')
        shifted_dt = parsed_dt.astimezone(timezone(timedelta(hours=-10))).replace(tzinfo=None)
        # shifted_dt = parsed_dt.astimezone(timezone(timedelta(hours=-8))).replace(tzinfo=None)
        abs_time.append(shifted_dt)

    df['Timestamp_converted'] = abs_time
    df = df.sort_values('Timestamp_converted')

    # ── Load XIR frame timestamps ─────────────────────────────────────────────
    xir_folder = (
        "/Users/supriya/Documents/data/agentic_system/"
        "multi-agent-manufacturing-inspection/training/images/"
        f"xiris_frames/temp_frames/temp_frames/temp_frames_{bead_id}"
    )

    file_names = [f for f in os.listdir(xir_folder)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    xir_t, xir_path = [], []
    for filename in file_names:
        t_str = os.path.splitext(filename)[0]
        try:
            seconds_part      = int(t_str[:-6])
            microseconds_part = int(t_str[-6:])
            frame_time = (datetime.fromtimestamp(seconds_part)
                          + timedelta(microseconds=microseconds_part))
            xir_t.append(frame_time)
            xir_path.append(os.path.join(xir_folder, filename))
        except ValueError:
            continue

    image_df = pd.DataFrame({'xir_t': xir_t, 'xir_path': xir_path})
    image_df = image_df.sort_values('xir_t')

    aligned_data = pd.merge_asof(
        image_df, df,
        left_on='xir_t', right_on='Timestamp_converted',
        direction='nearest'
    )

    print("Successfully aligned frames with voltage and current!")

    # ── Output directories ────────────────────────────────────────────────────
    plot_output_dir = f"frame_history_plots_{bead_id}"
    os.makedirs(plot_output_dir, exist_ok=True)

    signal_plot_dir = (
        "/Users/supriya/Documents/data/agentic_system/"
        "multi-agent-manufacturing-inspection/training/images/"
        f"signal_plots_{bead_id}"
    )
    os.makedirs(signal_plot_dir, exist_ok=True)

    df_sorted   = df.sort_values('Timestamp_converted').reset_index(drop=True)
    global_lims = compute_global_limits(df_sorted)
    global_v_max = df['voltage_roll'].max() * 1.1
    global_c_max = df['current_roll'].max() * 1.1

    print(f"\n--- Generating 2-row history plots → '{plot_output_dir}' ---")
    print(f"--- Generating 4-row signal plots  → '{signal_plot_dir}' ---")

    for row_idx in range(len(aligned_data)):
        target_perf_time = aligned_data.loc[row_idx, 'time']
        if pd.isna(target_perf_time):
            continue

        start_time = target_perf_time - lookback_seconds
        history_slice = df_sorted[
            (df_sorted['time'] >= start_time) &
            (df_sorted['time'] <= target_perf_time)
        ].copy().sort_values('time')

        if len(history_slice) < 5:
            continue

        now_time = history_slice['time'].iloc[-1]
        now_v    = history_slice['voltage_roll'].iloc[-1]
        now_c    = history_slice['current_roll'].iloc[-1]

        original_filename = os.path.basename(aligned_data.loc[row_idx, 'xir_path'])

        # ── Existing 2-row history context plot ──────────────────────────────
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

        ax1.plot(history_slice['time'], history_slice['voltage_roll'],
                 color='blue', linewidth=1.5, label='Rolling Voltage (0.05s)')
        ax1.axvline(x=now_time, color='black', linestyle='--', alpha=0.6, label='Current Frame Time')
        ax1.plot(now_time, now_v, 'ro', markersize=8)
        ax1.set_ylabel('Voltage (V)', fontweight='bold')
        ax1.set_xlabel('PerfTime (s)', fontweight='bold')
        ax1.set_ylim(0, global_v_max)
        ax1.set_title(f'Signal History | Frame: {original_filename}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize='small')
        ax1.text(now_time, now_v, f'  {now_v:.2f}V', va='bottom', fontweight='bold', color='blue')

        ax2.plot(history_slice['time'], history_slice['current_roll'],
                 color='red', linewidth=1.5, label='Rolling Current (0.05s)')
        ax2.axvline(x=now_time, color='black', linestyle='--', alpha=0.6, label='Current Frame Time')
        ax2.plot(now_time, now_c, 'ro', markersize=8)
        ax2.set_ylabel('Current (A)', fontweight='bold')
        ax2.set_xlabel('PerfTime (s)', fontweight='bold')
        ax2.set_ylim(0, global_c_max)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left', fontsize='small')
        ax2.text(now_time, now_c, f'  {now_c:.2f}A', va='bottom', fontweight='bold', color='red')

        fig.tight_layout()
        plt.savefig(os.path.join(plot_output_dir, original_filename), dpi=150)
        plt.close(fig)

        # ── 4-row signal plots ────────────────────────────────────────────────
        plot_frame_signals(
            history_slice    = history_slice,
            now_time         = now_time,
            frame_filename   = original_filename,
            global_lims      = global_lims,
            output_dir       = signal_plot_dir,
            lookback_seconds = lookback_seconds,
            spike_n_sigma    = SPIKE_N_SIGMA,
            trend_seconds    = trend_seconds,
        )

        if row_idx % 20 == 0:
            print(f"  Plotting: {row_idx}/{len(aligned_data)} frames.")

    print(f"\nDone. Signal plots → {signal_plot_dir}/")

except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()