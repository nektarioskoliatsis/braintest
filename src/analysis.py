"""
Analysis Tools for Cortical Microcircuit Simulation
=====================================================
Spike train analysis, LFP proxy estimation, spectral analysis,
and figure generation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal
import os


def compute_firing_rates(spike_times, spike_indices, n_neurons, duration, bin_size=5e-3):
    """Compute population firing rate histogram."""
    bins = np.arange(0, duration, bin_size)
    counts, edges = np.histogram(spike_times, bins=bins)
    rates = counts / (bin_size * n_neurons)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, rates


def compute_cv_isi(spike_times, spike_indices, n_neurons):
    """
    Compute coefficient of variation of inter-spike intervals.
    CV_ISI ~ 1.0 indicates Poisson-like irregular firing.
    """
    cvs = []
    for i in range(n_neurons):
        mask = spike_indices == i
        times = np.sort(spike_times[mask])
        if len(times) < 3:
            continue
        isis = np.diff(times)
        if np.mean(isis) > 0:
            cvs.append(np.std(isis) / np.mean(isis))
    return np.array(cvs)


def compute_lfp_proxy(spike_times_exc, spike_indices_exc, n_exc,
                      spike_times_inh, spike_indices_inh, n_inh,
                      duration, dt=1e-4):
    """
    Estimate LFP proxy from synaptic currents.
    Uses the difference between excitatory and inhibitory population
    activity convolved with synaptic kernels.
    """
    t = np.arange(0, duration, dt)
    n_bins = len(t)

    exc_counts, _ = np.histogram(spike_times_exc, bins=n_bins, range=(0, duration))
    inh_counts, _ = np.histogram(spike_times_inh, bins=n_bins, range=(0, duration))

    tau_ampa = 2e-3
    tau_gaba = 5e-3
    kernel_t = np.arange(0, 50e-3, dt)
    kernel_ampa = np.exp(-kernel_t / tau_ampa)
    kernel_gaba = np.exp(-kernel_t / tau_gaba)
    kernel_ampa /= kernel_ampa.sum()
    kernel_gaba /= kernel_gaba.sum()

    exc_current = np.convolve(exc_counts.astype(float), kernel_ampa, mode="same")
    inh_current = np.convolve(inh_counts.astype(float), kernel_gaba, mode="same")

    lfp = inh_current - exc_current
    return t, lfp


def compute_power_spectrum(sig, fs, nperseg=1024):
    """Compute power spectral density using Welch's method."""
    freqs, psd = signal.welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    return freqs, psd


def find_gamma_peak(freqs, psd, gamma_range=(25, 80)):
    """Find peak frequency in the gamma band."""
    mask = (freqs >= gamma_range[0]) & (freqs <= gamma_range[1])
    if not np.any(mask):
        return None, None
    gamma_freqs = freqs[mask]
    gamma_psd = psd[mask]
    peak_idx = np.argmax(gamma_psd)
    return gamma_freqs[peak_idx], gamma_psd[peak_idx]


def generate_figures(results, duration, output_dir="results"):
    """Generate all analysis figures from simulation output dict."""
    os.makedirs(output_dir, exist_ok=True)

    exc_times = results["exc_times"]
    exc_indices = results["exc_indices"]
    inh_times = results["inh_times"]
    inh_indices = results["inh_indices"]
    N_exc = results["N_exc"]
    N_inh = results["N_inh"]
    v_trace = results["state_v"][0] * 1e3  # V -> mV
    t_trace = results["state_t"]

    # --- Figure: Raster plot + firing rates ---
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Raster plot (subsample for visibility)
    ax1 = fig.add_subplot(gs[0:2, 0])
    max_neurons_plot = 500
    exc_mask = exc_indices < min(max_neurons_plot, N_exc)
    inh_mask = inh_indices < min(max_neurons_plot // 4, N_inh)
    ax1.scatter(exc_times[exc_mask], exc_indices[exc_mask],
                s=0.3, c="C0", alpha=0.5, rasterized=True, label="Exc")
    ax1.scatter(inh_times[inh_mask], inh_indices[inh_mask] + N_exc,
                s=0.3, c="C3", alpha=0.5, rasterized=True, label="Inh")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Neuron index")
    ax1.set_title("Spike Raster Plot")
    ax1.legend(loc="upper right", markerscale=10)

    # Population firing rates
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.plot(results["rate_t_exc"], results["rate_exc"], "C0", alpha=0.8, label="Exc")
    ax2.plot(results["rate_t_inh"], results["rate_inh"], "C3", alpha=0.8, label="Inh")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Firing rate (Hz)")
    ax2.set_title("Population Firing Rates")
    ax2.legend()

    # Example membrane potential trace
    ax3 = fig.add_subplot(gs[3, 0])
    ax3.plot(t_trace, v_trace, "k", linewidth=0.5)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("V (mV)")
    ax3.set_title("Example Excitatory Neuron Membrane Potential")

    # LFP proxy and power spectrum
    t_lfp, lfp = compute_lfp_proxy(exc_times, exc_indices, N_exc,
                                    inh_times, inh_indices, N_inh,
                                    duration)
    fs_lfp = 1.0 / (t_lfp[1] - t_lfp[0])

    ax4 = fig.add_subplot(gs[0, 1])
    t_start = max(0, duration - 0.5)
    mask_lfp = t_lfp >= t_start
    ax4.plot(t_lfp[mask_lfp] * 1000, lfp[mask_lfp], "k", linewidth=0.5)
    ax4.set_xlabel("Time (ms)")
    ax4.set_ylabel("LFP proxy (a.u.)")
    ax4.set_title("Local Field Potential Proxy (last 500 ms)")

    # Power spectrum
    ax5 = fig.add_subplot(gs[1, 1])
    transient_samples = int(0.2 * fs_lfp)
    freqs, psd = compute_power_spectrum(lfp[transient_samples:], fs_lfp, nperseg=2048)
    freq_mask = freqs <= 150
    ax5.semilogy(freqs[freq_mask], psd[freq_mask], "k", linewidth=1)
    peak_freq, peak_power = find_gamma_peak(freqs, psd)
    if peak_freq is not None:
        ax5.axvline(peak_freq, color="C1", linestyle="--", alpha=0.7,
                    label=f"Gamma peak: {peak_freq:.1f} Hz")
        ax5.legend()
    ax5.set_xlabel("Frequency (Hz)")
    ax5.set_ylabel("PSD")
    ax5.set_title("Power Spectral Density")

    # CV ISI distribution
    ax6 = fig.add_subplot(gs[2, 1])
    cv_exc = compute_cv_isi(exc_times, exc_indices, N_exc)
    cv_inh = compute_cv_isi(inh_times, inh_indices, N_inh)
    if len(cv_exc) > 0:
        ax6.hist(cv_exc, bins=30, alpha=0.6, color="C0", label=f"Exc (mean={np.mean(cv_exc):.2f})")
    if len(cv_inh) > 0:
        ax6.hist(cv_inh, bins=30, alpha=0.6, color="C3", label=f"Inh (mean={np.mean(cv_inh):.2f})")
    ax6.set_xlabel("CV of ISI")
    ax6.set_ylabel("Count")
    ax6.set_title("Firing Irregularity (CV ISI)")
    ax6.legend()

    # Summary statistics text
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.axis("off")
    n_spikes_exc = len(exc_times)
    n_spikes_inh = len(inh_times)
    mean_rate_exc = n_spikes_exc / (N_exc * duration) if N_exc > 0 else 0
    mean_rate_inh = n_spikes_inh / (N_inh * duration) if N_inh > 0 else 0
    stats_text = (
        f"Network: {N_exc} exc + {N_inh} inh neurons\n"
        f"Duration: {duration:.1f} s\n"
        f"Mean firing rate (exc): {mean_rate_exc:.2f} Hz\n"
        f"Mean firing rate (inh): {mean_rate_inh:.2f} Hz\n"
        f"Total spikes: {n_spikes_exc + n_spikes_inh:,}\n"
    )
    if peak_freq is not None:
        stats_text += f"Gamma peak: {peak_freq:.1f} Hz\n"
    if len(cv_exc) > 0:
        stats_text += f"Mean CV ISI (exc): {np.mean(cv_exc):.2f}\n"
    ax7.text(0.1, 0.9, stats_text, transform=ax7.transAxes,
             fontsize=11, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("Cortical Microcircuit Simulation", fontsize=14, fontweight="bold")
    plt.savefig(os.path.join(output_dir, "simulation_results.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Figures saved to {output_dir}/")
    print(f"  Mean exc rate: {mean_rate_exc:.2f} Hz")
    print(f"  Mean inh rate: {mean_rate_inh:.2f} Hz")
    if peak_freq:
        print(f"  Gamma peak: {peak_freq:.1f} Hz")

    return {
        "mean_rate_exc": mean_rate_exc,
        "mean_rate_inh": mean_rate_inh,
        "gamma_peak": peak_freq,
        "cv_exc": np.mean(cv_exc) if len(cv_exc) > 0 else None,
    }
