"""
Partitioned GPU Simulator
=========================
Hybrid CPU/GPU spiking network simulation using PyTorch.

- Neuron state updates: GPU (PyTorch tensors)
- Connectivity: CPU (scipy sparse matrices)
- Spike exchange: small CPU<->GPU transfers each timestep

This lets you simulate networks much larger than GPU VRAM,
since the connectivity matrix (the memory hog) stays on CPU.
Only neuron state (~100 bytes/neuron) lives on GPU.
"""

import torch
import numpy as np
from scipy import sparse
import time


class SpikeMonitorData:
    """Collects spike times and neuron indices."""

    def __init__(self):
        self._times = []
        self._indices = []

    def record(self, time_s, indices):
        if len(indices) > 0:
            self._times.append(np.full(len(indices), time_s, dtype=np.float64))
            self._indices.append(indices.astype(np.int64))

    @property
    def times(self):
        if len(self._times) == 0:
            return np.array([], dtype=np.float64)
        return np.concatenate(self._times)

    @property
    def indices(self):
        if len(self._indices) == 0:
            return np.array([], dtype=np.int64)
        return np.concatenate(self._indices)


class StateMonitorData:
    """Records voltage traces for a subset of neurons."""

    def __init__(self, n_record, n_steps):
        self.v = np.zeros((n_record, n_steps), dtype=np.float64)
        self.t = np.zeros(n_steps, dtype=np.float64)
        self._idx = 0

    def record_step(self, t, v_values):
        if self._idx < len(self.t):
            self.t[self._idx] = t
            self.v[:, self._idx] = v_values
            self._idx += 1

    def finalize(self):
        self.t = self.t[: self._idx]
        self.v = self.v[:, : self._idx]


class RateMonitorData:
    """Records instantaneous population firing rates."""

    def __init__(self, n_neurons, dt):
        self.n_neurons = n_neurons
        self.dt = dt
        self._times = []
        self._spike_counts = []

    def record_step(self, t, n_spikes):
        self._times.append(t)
        self._spike_counts.append(n_spikes)

    @property
    def times(self):
        return np.array(self._times, dtype=np.float64)

    @property
    def rates(self):
        """Raw instantaneous rate in Hz."""
        counts = np.array(self._spike_counts, dtype=np.float64)
        return counts / (self.n_neurons * self.dt)

    def smooth_rate(self, width):
        """Smooth rate with Gaussian kernel. Width in seconds."""
        rates = self.rates
        if len(rates) < 2:
            return rates
        kernel_width = max(1, int(width / self.dt))
        x = np.arange(-3 * kernel_width, 3 * kernel_width + 1)
        kernel = np.exp(-0.5 * (x / kernel_width) ** 2)
        kernel /= kernel.sum()
        return np.convolve(rates, kernel, mode="same")


class PartitionedSimulator:
    """
    Hybrid CPU/GPU spiking neural network simulator.

    Strategy:
    - All neuron state (v, w, g_ampa, g_gaba) on GPU — tiny memory footprint
    - Connectivity stored as scipy CSR sparse matrices on CPU
    - Each timestep: GPU neuron update -> detect spikes -> CPU sparse delivery -> GPU apply
    - Spike transfer is tiny: just indices of ~0.1% of neurons per step

    Memory budget (GPU):
      ~200 bytes/neuron (state + params) — 100K neurons = 20 MB
      Connectivity stays on CPU: 100K neurons @ p=0.02 = 200M synapses = ~2.4 GB CPU RAM

    This means a 4GB GPU can handle millions of neurons (limited by CPU RAM for connectivity).
    """

    @staticmethod
    def _parse_nS(value):
        """Parse a value like '0.5nS' or 0.5 into siemens (float)."""
        if isinstance(value, str):
            return float(value.replace("nS", "").strip()) * 1e-9
        return float(value) * 1e-9

    def __init__(self, config, seed=42, device=None):
        self.config = config
        self.seed = seed

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(seed)

        # Network sizing
        N = config["network"]["n_neurons"]
        self.N = N
        self.N_exc = int(N * config["network"]["exc_fraction"])
        self.N_inh = N - self.N_exc
        self.p_conn = config["network"]["connection_prob"]
        self.dt = config["simulation"]["dt"] * 1e-3  # ms -> seconds

        # Delays in timesteps
        self.delay_exc_steps = round(1.5e-3 / self.dt)  # 15 steps at 0.1ms
        self.delay_inh_steps = round(0.8e-3 / self.dt)  # 8 steps
        self.max_delay_steps = max(self.delay_exc_steps, self.delay_inh_steps)

        # Refractory periods in timesteps
        self.refrac_exc_steps = round(2e-3 / self.dt)  # 20 steps
        self.refrac_inh_steps = round(1e-3 / self.dt)  # 10 steps

        # Plasticity
        self.U_std = config["plasticity"]["U_std"]
        self.tau_rec = 200e-3  # seconds

        # Synaptic weights from config (values in nS, stored as strings like "0.5nS")
        syn_cfg = config["synapses"]
        self.w_ee = self._parse_nS(syn_cfg["w_ee"])
        self.w_ei = self._parse_nS(syn_cfg["w_ei"])
        self.w_ie = self._parse_nS(syn_cfg["w_ie"])
        self.w_ii = self._parse_nS(syn_cfg["w_ii"])

        # Poisson input
        self.poisson_rate = 8000.0  # Hz
        self.poisson_weight = 0.5e-9  # 0.5 nS

        # Reversal potentials and time constants
        self.E_exc = 0.0
        self.E_inh = -80e-3
        self.tau_ampa = 2e-3
        self.tau_gaba = 5e-3

        # STD recovery decay factor (precomputed)
        self.std_decay = np.float32(np.exp(-self.dt / self.tau_rec))

        self._init_neurons()
        self._build_connectivity()

    def _init_neurons(self):
        """Initialize all neuron state and parameter tensors on GPU."""
        N = self.N
        NE = self.N_exc

        # State variables
        v_init = -70e-3 + np.random.randn(N).astype(np.float32) * 10e-3
        self.v = torch.from_numpy(v_init).to(self.device)
        self.w = torch.zeros(N, dtype=torch.float32, device=self.device)
        self.g_ampa = torch.zeros(N, dtype=torch.float32, device=self.device)
        self.g_gaba = torch.zeros(N, dtype=torch.float32, device=self.device)
        self.refrac = torch.zeros(N, dtype=torch.int32, device=self.device)

        # Parameters (per-neuron, on GPU)
        self.C = torch.zeros(N, dtype=torch.float32, device=self.device)
        self.gL = torch.full((N,), 10e-9, dtype=torch.float32, device=self.device)
        self.EL = torch.full((N,), -70e-3, dtype=torch.float32, device=self.device)
        self.VT = torch.full((N,), -50e-3, dtype=torch.float32, device=self.device)
        self.DeltaT = torch.full((N,), 2e-3, dtype=torch.float32, device=self.device)
        self.Vreset = torch.zeros(N, dtype=torch.float32, device=self.device)
        self.tau_w = torch.zeros(N, dtype=torch.float32, device=self.device)
        self.a_param = torch.zeros(N, dtype=torch.float32, device=self.device)
        self.b_param = torch.zeros(N, dtype=torch.float32, device=self.device)

        # Excitatory parameters
        self.C[:NE] = 200e-12
        self.Vreset[:NE] = -60e-3
        self.tau_w[:NE] = 200e-3
        self.a_param[:NE] = 2e-9
        self.b_param[:NE] = 40e-12

        # Inhibitory parameters
        self.C[NE:] = 100e-12
        self.Vreset[NE:] = -65e-3
        self.tau_w[NE:] = 20e-3
        self.a_param[NE:] = 0.5e-9
        self.b_param[NE:] = 5e-12

        # Precompute exact exponential decay factors (more accurate than linear)
        self.ampa_decay = float(np.exp(-self.dt / self.tau_ampa))
        self.gaba_decay = float(np.exp(-self.dt / self.tau_gaba))

        # Spike ring buffer for delays
        buf_len = self.max_delay_steps + 1
        self.spike_buffer = [np.array([], dtype=np.int64) for _ in range(buf_len)]
        self.buf_idx = 0

    def _build_connectivity(self):
        """Build sparse connectivity on CPU."""
        NE = self.N_exc
        NI = self.N_inh
        p = self.p_conn

        print(f"  Building connectivity ({self.N:,} neurons, p={p})...")
        t0 = time.perf_counter()

        def _random_no_autapse(n_pre, n_post, density, weight, remove_diag=False):
            """Create random sparse connectivity using memory-efficient chunked COO generation.

            Instead of scipy.sparse.random (which builds dense intermediate arrays),
            we generate random (row, col) pairs in chunks using numpy, then assemble
            the CSR matrix directly from COO data.
            """
            # Expected number of non-zeros
            nnz_expected = int(n_pre * n_post * density)
            # Generate in chunks to limit peak memory
            chunk_size = min(nnz_expected, 2_000_000)
            rows_list = []
            cols_list = []
            remaining = nnz_expected

            while remaining > 0:
                n_gen = min(chunk_size, remaining)
                rows_list.append(np.random.randint(0, n_pre, size=n_gen, dtype=np.int32))
                cols_list.append(np.random.randint(0, n_post, size=n_gen, dtype=np.int32))
                remaining -= n_gen

            rows = np.concatenate(rows_list)
            cols = np.concatenate(cols_list)

            # Remove self-connections (autapses)
            if remove_diag and n_pre == n_post:
                mask = rows != cols
                rows = rows[mask]
                cols = cols[mask]

            # Remove duplicate (row, col) pairs by converting to set of linear indices
            linear = rows.astype(np.int64) * n_post + cols.astype(np.int64)
            linear = np.unique(linear)
            rows = (linear // n_post).astype(np.int32)
            cols = (linear % n_post).astype(np.int32)

            data = np.full(len(rows), weight, dtype=np.float32)
            W = sparse.csr_matrix((data, (rows, cols)), shape=(n_pre, n_post))
            W.sort_indices()
            return W

        # E->E (with STD)
        self.W_ee = _random_no_autapse(NE, NE, p, self.w_ee, remove_diag=True)
        self.x_std = np.ones_like(self.W_ee.data)  # STD state, parallel to W_ee.data

        # E->I
        self.W_ei = _random_no_autapse(NE, NI, p, self.w_ei)

        # I->E
        self.W_ie = _random_no_autapse(NI, NE, p, self.w_ie)

        # I->I
        self.W_ii = _random_no_autapse(NI, NI, p, self.w_ii, remove_diag=True)

        # Pre-transpose for non-STD synapses (CSC of transpose = efficient column access)
        self.W_ei_T = self.W_ei.T.tocsr()
        self.W_ie_T = self.W_ie.T.tocsr()
        self.W_ii_T = self.W_ii.T.tocsr()

        n_syn = self.W_ee.nnz + self.W_ei.nnz + self.W_ie.nnz + self.W_ii.nnz
        mem_mb = n_syn * 12 / 1e6
        dt_build = time.perf_counter() - t0
        print(f"  {n_syn:,} synapses ({mem_mb:.0f} MB CPU RAM) built in {dt_build:.1f}s")

    def _step_neurons(self):
        """Euler integration of AdEx equations on GPU.

        Computes all derivatives from state at time t before updating,
        matching Brian2's Euler method for coupled ODEs.
        """
        dt = self.dt
        active = (self.refrac <= 0).float()

        # Exponential term with overflow protection
        exp_arg = torch.clamp((self.v - self.VT) / self.DeltaT, max=20.0)

        # Compute ALL derivatives from current state (before any updates)
        dv = (
            -self.gL * (self.v - self.EL)
            + self.gL * self.DeltaT * torch.exp(exp_arg)
            - self.g_ampa * (self.v - self.E_exc)
            - self.g_gaba * (self.v - self.E_inh)
            - self.w
        ) / self.C

        dw = (self.a_param * (self.v - self.EL) - self.w) / self.tau_w

        # Update state (all at once, using derivatives from time t)
        self.v += dt * dv * active
        self.w += dt * dw

        # Conductance decay
        self.g_ampa *= self.ampa_decay
        self.g_gaba *= self.gaba_decay

        # Refractory countdown
        self.refrac = (self.refrac - 1).clamp(min=0)

    def _detect_spikes(self):
        """Detect spikes, apply reset, return indices on CPU."""
        spiked = self.v > 0.0  # 0 mV threshold

        if not spiked.any():
            return np.array([], dtype=np.int64)

        idx = torch.where(spiked)[0]

        # Reset
        self.v[idx] = self.Vreset[idx]
        self.w[idx] = self.w[idx] + self.b_param[idx]

        # Set refractory
        exc_mask = idx < self.N_exc
        self.refrac[idx[exc_mask]] = self.refrac_exc_steps
        self.refrac[idx[~exc_mask]] = self.refrac_inh_steps

        return idx.cpu().numpy()

    def _deliver_spikes(self, current_spikes):
        """Buffer current spikes and deliver delayed spikes via CPU sparse ops."""
        NE = self.N_exc
        N = self.N
        buf_len = self.max_delay_steps + 1

        # Store current spikes in ring buffer
        self.spike_buffer[self.buf_idx] = current_spikes

        # Get delayed excitatory spikes
        exc_delay_idx = (self.buf_idx - self.delay_exc_steps) % buf_len
        delayed_exc_all = self.spike_buffer[exc_delay_idx]
        exc_spiked = delayed_exc_all[delayed_exc_all < NE] if len(delayed_exc_all) > 0 else np.array([], dtype=np.int64)

        # Get delayed inhibitory spikes
        inh_delay_idx = (self.buf_idx - self.delay_inh_steps) % buf_len
        delayed_inh_all = self.spike_buffer[inh_delay_idx]
        inh_spiked_global = delayed_inh_all[delayed_inh_all >= NE] if len(delayed_inh_all) > 0 else np.array([], dtype=np.int64)
        inh_spiked = inh_spiked_global - NE  # Local indices

        # Accumulate conductance updates on CPU
        g_ampa_update = np.zeros(N, dtype=np.float32)
        g_gaba_update = np.zeros(N, dtype=np.float32)

        # --- Excitatory spike delivery ---
        if len(exc_spiked) > 0:
            # E->E with STD (loop over spiking neurons — few per step)
            for pre in exc_spiked:
                s, e = self.W_ee.indptr[pre], self.W_ee.indptr[pre + 1]
                if s == e:
                    continue
                posts = self.W_ee.indices[s:e]
                w = self.W_ee.data[s:e]
                x = self.x_std[s:e]
                g_ampa_update[posts] += w * x * self.U_std
                self.x_std[s:e] *= 1.0 - self.U_std

            # E->I (vectorized sparse mat-vec)
            spike_vec = np.zeros(NE, dtype=np.float32)
            spike_vec[exc_spiked] = 1.0
            g_ampa_update[NE:] += self.W_ei_T.dot(spike_vec)

        # --- Inhibitory spike delivery ---
        if len(inh_spiked) > 0:
            spike_vec_inh = np.zeros(self.N_inh, dtype=np.float32)
            spike_vec_inh[inh_spiked] = 1.0
            g_gaba_update[:NE] += self.W_ie_T.dot(spike_vec_inh)
            g_gaba_update[NE:] += self.W_ii_T.dot(spike_vec_inh)

        # Transfer to GPU (only if nonzero)
        if g_ampa_update.any():
            self.g_ampa += torch.from_numpy(g_ampa_update).to(self.device)
        if g_gaba_update.any():
            self.g_gaba += torch.from_numpy(g_gaba_update).to(self.device)

        # Advance ring buffer
        self.buf_idx = (self.buf_idx + 1) % buf_len

    def _recover_std(self):
        """STD recovery: x_std -> 1 with time constant tau_rec."""
        # x(t+dt) = 1 - (1 - x(t)) * exp(-dt/tau_rec)
        self.x_std = 1.0 - (1.0 - self.x_std) * self.std_decay

    def _poisson_input(self):
        """Apply Poisson background drive on GPU.

        Uses actual Poisson distribution (not Bernoulli) since
        rate*dt = 0.8, meaning multiple spikes per step are common.
        Matches Brian2's PoissonInput behavior.
        """
        lam = self.poisson_rate * self.dt  # expected spikes per step (~0.8)
        n_spikes = torch.poisson(torch.full(
            (self.N,), lam, device=self.device, dtype=torch.float32
        ))
        self.g_ampa += n_spikes * self.poisson_weight

    def run(self, duration):
        """
        Run the full simulation.

        Returns a dict with spike/state/rate monitor data
        compatible with analysis.generate_figures_from_arrays().
        """
        n_steps = int(round(duration / self.dt))
        NE = self.N_exc
        NI = self.N_inh

        # Initialize monitors
        spike_exc = SpikeMonitorData()
        spike_inh = SpikeMonitorData()
        n_record = min(5, NE)
        state_mon = StateMonitorData(n_record, n_steps)
        rate_exc = RateMonitorData(NE, self.dt)
        rate_inh = RateMonitorData(NI, self.dt)

        print(f"\nRunning partitioned GPU simulation...")
        print(f"  {self.N:,} neurons ({NE:,} E / {NI:,} I)")
        print(f"  {duration}s bio time, {n_steps:,} steps (dt={self.dt*1e3:.1f}ms)")
        print(f"  Device: {self.device}")
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(0)
            total_mem = props.total_memory
            print(f"  GPU: {props.name} ({total_mem / 1e9:.1f} GB)")
            torch.cuda.reset_peak_memory_stats()

        t_start = time.perf_counter()
        last_report = t_start

        for step in range(n_steps):
            t = step * self.dt

            # 1. Neuron dynamics (GPU)
            self._step_neurons()

            # 2. Spike detection (GPU -> CPU)
            spikes = self._detect_spikes()

            # 3. Record spikes
            if len(spikes) > 0:
                exc_spk = spikes[spikes < NE]
                inh_spk_local = spikes[spikes >= NE] - NE
                spike_exc.record(t, exc_spk)
                spike_inh.record(t, inh_spk_local)
                n_exc_spk = len(exc_spk)
                n_inh_spk = len(inh_spk_local)
            else:
                n_exc_spk = 0
                n_inh_spk = 0

            rate_exc.record_step(t, n_exc_spk)
            rate_inh.record_step(t, n_inh_spk)

            # Record state (every step for accuracy)
            if n_record > 0:
                state_mon.record_step(t, self.v[:n_record].cpu().numpy())

            # 4. Delayed spike delivery (CPU sparse -> GPU)
            self._deliver_spikes(spikes)

            # 5. STD recovery (CPU)
            self._recover_std()

            # 6. Poisson input (GPU)
            self._poisson_input()

            # Progress
            now = time.perf_counter()
            if now - last_report > 5.0:
                elapsed = now - t_start
                frac = (step + 1) / n_steps
                eta = elapsed / frac * (1 - frac)
                print(
                    f"  {frac*100:5.1f}% | {t:.3f}s bio | "
                    f"{elapsed:.0f}s wall | ETA {eta:.0f}s"
                )
                last_report = now

        t_total = time.perf_counter() - t_start
        state_mon.finalize()

        print(f"\n  Simulation completed in {t_total:.2f}s")
        print(f"  Bio/wall ratio: {duration / t_total:.2f}x")
        if self.device.type == "cuda":
            peak_mb = torch.cuda.max_memory_allocated() / 1e6
            print(f"  Peak GPU memory: {peak_mb:.0f} MB")

        return {
            "exc_times": spike_exc.times,
            "exc_indices": spike_exc.indices,
            "inh_times": spike_inh.times,
            "inh_indices": spike_inh.indices,
            "rate_t_exc": rate_exc.times,
            "rate_exc": rate_exc.smooth_rate(5e-3),
            "rate_t_inh": rate_inh.times,
            "rate_inh": rate_inh.smooth_rate(5e-3),
            "state_v": state_mon.v,
            "state_t": state_mon.t,
            "N_exc": NE,
            "N_inh": NI,
            "wall_time": t_total,
            "duration": duration,
        }
