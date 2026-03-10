# Braintest

A personal learning project — I built this with the help of Claude (Anthropic's AI) to understand how brain simulations work and how to split computation between a CPU and GPU.

This is **not** a polished research tool. It's a personal exploration of computational neuroscience concepts, built for my own understanding.

## What it does

It simulates a small chunk of brain-like activity on your computer. Imagine 10,000 tiny simulated neurons sending electrical signals to each other — some neurons excite their neighbors (make them more likely to fire), others inhibit them (make them less likely to fire). The result is a pattern of activity that looks like what scientists actually measure in real brains.

### The key trick: splitting work between CPU and GPU

The main idea I wanted to explore was **keeping the wiring map on the CPU and the neuron math on the GPU**. Here's why:

- **Neuron state** (voltage, currents, etc.) is small — about 200 bytes per neuron. Even 100K neurons fit in ~20 MB. GPUs are great at updating all these neurons at once.
- **The wiring between neurons** is huge — 10K neurons with 2% connectivity means 2 million connections taking ~24 MB. At 50K neurons that jumps to ~600 MB. This would eat up a small laptop GPU's memory fast.

So the simulator keeps the wiring as sparse matrices in regular RAM and only sends small updates (which neurons just fired) back and forth between CPU and GPU each timestep.

### The neuron model

Each neuron uses the **AdEx (Adaptive Exponential Integrate-and-Fire)** model. In simple terms:
- Neurons accumulate input until they hit a threshold, then they "fire" (spike) and reset
- There's an adaptation current that makes neurons slow down if they fire a lot
- Excitatory neurons fire slowly (~4 Hz), inhibitory neurons fire faster (~19 Hz)

### What the output looks like

When it runs, the network settles into what neuroscientists call the **asynchronous irregular state** — neurons fire at seemingly random times rather than all together. This is actually what real brains do when you're awake. The simulation also produces gamma oscillations (~30 Hz) that emerge naturally from the push-pull between excitatory and inhibitory neurons.

## Known issues

These are things I'm aware of but haven't fixed:

- **The config file doesn't actually do anything for synaptic weights.** The weights are hardcoded in `gpu_partitioned.py`. For example, the config says `w_ei: 0.8nS` but the code uses `0.5 nS`. Changing the config won't change the simulation. The neuron parameters and network size *do* come from the config though.
- **`requirements.txt` includes brian2 and brian2cuda unnecessarily.** Those packages aren't used anywhere in the code — they were listed because I originally compared against Brian2 benchmarks, but the comparison scripts aren't included here.
- **No tests.** This is a one-off learning project, not production software.

## Running it

```bash
pip install numpy matplotlib scipy pyyaml
pip install torch  # add --index-url https://download.pytorch.org/whl/cu118 for CUDA

# Default: 10K neurons, 2 seconds of simulated brain time
python src/run_simulation.py

# Bigger network (takes longer)
python src/run_simulation.py --n_neurons 25000 --duration 1.0

# Skip generating plots
python src/run_simulation.py --no-plot
```

On my laptop (GTX 1650 Max-Q, 16 GB RAM), 10K neurons takes about 65–87 seconds and 50K neurons takes about 99 minutes.

## What I learned

- How the AdEx neuron model works and why it's a good middle ground between too-simple (LIF) and too-expensive (Hodgkin-Huxley)
- Why conductance-based synapses matter — they provide natural gain control that current-based synapses don't
- How to use sparse matrices (scipy) alongside GPU tensors (PyTorch) in the same simulation loop
- How sensitive balanced networks are to parameter tuning — small changes to inhibitory weights completely change the firing rates
- How to estimate a local field potential (LFP) from spike data and find oscillation frequencies with spectral analysis

## Project structure

```
braintest/
├── src/
│   ├── gpu_partitioned.py   # Core simulator (~530 lines, PyTorch + scipy)
│   ├── run_simulation.py    # CLI entry point
│   └── analysis.py          # Spike analysis, LFP, spectra, figures
├── configs/
│   └── default.yaml         # Model parameters (partially used, see known issues)
├── results/                 # Output figures and timing data
└── requirements.txt
```

## References

These are the papers the simulation is based on:

- Brette & Gerstner (2005). Adaptive exponential integrate-and-fire model. *J Neurophysiol*.
- Brunel (2000). Dynamics of sparsely connected networks. *J Comput Neurosci*.
- Tsodyks & Markram (1997). Neural code depends on release probability. *PNAS*.
- Destexhe et al. (2003). The high-conductance state of neocortical neurons in vivo. *Nat Rev Neurosci*.
