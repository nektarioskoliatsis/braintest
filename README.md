# Braintest

A spiking neural network simulator that runs cortical microcircuit simulations on a laptop GPU.

## The goal

Simulate biologically realistic brain circuits on consumer hardware. Existing tools like Brian2CUDA load the entire network into GPU memory, which means a 4 GB laptop GPU can only handle small networks before running out of VRAM. We wanted to push that limit much further.

## What we built

A **hybrid CPU/GPU simulator** that splits the problem: neuron math runs on the GPU (fast, parallel), while the connectivity matrix — which is the real memory hog — stays in regular RAM. This means the simulation scales with your system RAM, not your GPU VRAM.

The neurons are AdEx (Adaptive Exponential Integrate-and-Fire), which is the simplest model that still captures the two main types of cortical neurons: regular-spiking excitatory cells and fast-spiking inhibitory cells. The synapses are conductance-based (their effect depends on the neuron's voltage, like real synapses) with short-term depression on excitatory-to-excitatory connections.

The network produces the **asynchronous irregular state** — the same firing pattern observed in awake cortex. Excitatory neurons fire sparsely (~4 Hz), inhibitory neurons fire faster (~19 Hz) to keep things balanced, and gamma oscillations (~30 Hz) emerge naturally from the feedback loop between excitatory and inhibitory populations.

## Why we made the choices we made

### Why hybrid CPU/GPU instead of all-GPU?

The connectivity matrix dominates memory. For 10K neurons at 2% connection probability, there are 2 million synapses taking ~24 MB. That scales quadratically — at 50K neurons it's 50 million synapses taking ~600 MB, and at 100K it would be 2.4 GB. A 4 GB GPU fills up fast.

But the neuron state is tiny: about 200 bytes per neuron. Even 100K neurons is only 20 MB of GPU memory. So we keep the connectivity on CPU as scipy sparse matrices and only transfer spike indices (a handful per timestep) and conductance updates (~80 KB) between CPU and GPU. The GPU does what it's good at — parallel math on all neurons at once — and the CPU handles the sparse, irregular connectivity lookups.

### Why AdEx neurons?

We needed a model that distinguishes excitatory from inhibitory neurons without being as expensive as full Hodgkin-Huxley ion channel dynamics. AdEx has an exponential spike initiation term (realistic voltage trajectory near threshold) and an adaptation current (makes excitatory neurons fire regularly, inhibitory neurons fire fast). The parameters map directly to measurable properties like membrane capacitance and input resistance — they're not arbitrary tuning knobs.

LIF (Leaky Integrate-and-Fire) would be cheaper but can't capture adaptation or the different firing patterns of excitatory vs inhibitory cells. Hodgkin-Huxley would be 10-50x more expensive per neuron with no benefit at the network level we're studying.

### Why conductance-based synapses?

Current-based synapses inject a fixed amount of current regardless of the neuron's voltage. Real synapses don't work that way — their effect depends on the voltage difference between the neuron and the synapse's reversal potential. This matters because:

- **Inhibition gets stronger when the neuron is depolarized** (approaching threshold), providing automatic gain control that prevents runaway excitation
- **The effective membrane time constant shrinks** under high synaptic bombardment, matching what's measured in real cortical neurons in vivo

Without these properties, the balanced network dynamics fall apart.

### Why short-term depression on E→E only?

Excitatory-to-excitatory synapses in cortex consistently show depression — repeated firing makes them weaker. This prevents the excitatory population from amplifying its own activity into seizure-like bursts. We use the Tsodyks-Markram model: each spike depletes 50% of synaptic resources, which recover with a 200 ms time constant. This is the minimum plasticity needed for a stable network — adding it to other synapse types would increase memory usage without clear biological justification.

### Why Poisson input at 8 kHz?

Each neuron gets a barrage of random excitatory input representing all the external drives a cortical neuron receives (thalamus, other cortical areas, etc). The rate of 8 kHz was chosen so an isolated neuron fires at ~20 Hz, which gets pushed down to ~4 Hz by recurrent inhibition in the network.

We use actual Poisson-distributed spike counts, not the simpler Bernoulli approximation (0 or 1 spike per timestep). At rate x dt = 0.8, there's a 19% chance of 2+ spikes per timestep. Getting the input variance right matters because neurons in the balanced state fire based on fluctuations, not mean drive.

### Why the I→E weight matters so much

The balanced network state exists in a narrow parameter regime where excitation and inhibition nearly cancel. The mean membrane potential sits ~2 mV below threshold, and neurons fire only from random fluctuations. This means the firing rate depends **exponentially** on the E/I balance — changing the I→E weight from 4.0 nS to 2.0 nS shifted excitatory rates from 0.3 Hz to 4 Hz. We tuned this to match the ~3-5 Hz range observed in cortical recordings.

## The memory-efficient connectivity builder

The original implementation used `scipy.sparse.random()` to generate connectivity matrices. This works fine at 10K neurons but causes out-of-memory crashes at 50K because scipy builds dense intermediate arrays during construction.

We replaced it with a chunked approach: generate random (row, col) index pairs in batches of 2 million, deduplicate them, and assemble the sparse CSR matrix directly. This keeps peak memory proportional to the final matrix size rather than the construction intermediates.

## Results

All benchmarks ran on a laptop with an NVIDIA GTX 1650 Max-Q (4 GB VRAM) and 16 GB system RAM.

### Scaling

| Scale | Synapses | Build time | Sim time (2s bio) | CPU RAM | GPU RAM |
|-------|----------|------------|-------------------|---------|---------|
| 10,000 neurons | 2M | 1s | 87s | 24 MB | 1 MB |
| 25,000 neurons | 12.5M | 4s | 400s | 150 MB | 2 MB |
| 50,000 neurons | 49.5M | 8s | 99 min | 594 MB | 4 MB |

GPU memory stays under 4 MB at all scales — the connectivity (which dominates memory) lives entirely in CPU RAM. Scaling from 10K to 50K increases synapses by 25x and CPU RAM by 25x, but GPU memory barely changes. The wall-clock time scales roughly with synapse count since the per-step cost is dominated by sparse matrix operations on CPU.

### Network behavior (10K neurons, 2s biological time)

The network self-organizes into the asynchronous irregular (AI) state:

| Metric | Value | What it means |
|--------|-------|---------------|
| Exc firing rate | 3.69 Hz | Low, sparse firing — neurons are driven by fluctuations, not mean input |
| Inh firing rate | 18.84 Hz | ~5x faster than excitatory — fewer inh neurons must fire more to maintain balance |
| Gamma peak | 29.3 Hz | Oscillation emerges from E→I→E feedback loop, not hardcoded |
| CV ISI (exc) | 0.51 | Irregular firing (1.0 = perfectly Poisson, 0.0 = clock-like) |
| CV ISI (inh) | 0.74 | More irregular than excitatory, as expected for fast-spiking cells |
| Total spikes | 134,009 | Across all neurons over 2 seconds |
| Peak GPU memory | 1 MB | Neuron state only — connectivity on CPU |

These values match experimental recordings from cortical neurons in vivo. The gamma oscillation, irregular firing, and low excitatory rates are hallmarks of the balanced state that cortex operates in during wakefulness.

## Where this leads

This is groundwork for connectome-scale simulation. The fruit fly brain (~125K neurons, ~50M synapses) would need ~600 MB of CPU RAM for connectivity and ~25 MB of GPU memory — easily within reach of a 16 GB laptop. The step from here to a real connectome simulation is:

1. Replace random connectivity with actual wiring data (e.g., from the FlyWire connectome via NeuPrint)
2. Add cell-type-specific parameters (different AdEx parameters per neuron type)
3. Attach sensorimotor interfaces (visual input in, motor commands out)

The hybrid architecture doesn't need to change — it already handles the right scale.

## Usage

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Default: 10K neurons, 2s biological time
python src/run_simulation.py

# Larger network
python src/run_simulation.py --n_neurons 25000 --duration 1.0

# Skip plots
python src/run_simulation.py --n_neurons 50000 --no-plot
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_neurons` | 10000 | Total neuron count |
| `--duration` | 2.0 | Biological time (seconds) |
| `--config` | `configs/default.yaml` | Parameter file |
| `--seed` | 42 | Random seed |
| `--no-plot` | false | Skip figure generation |
| `--output-dir` | `results/` | Output directory |

## Project structure

```
braintest/
├── src/
│   ├── gpu_partitioned.py   # Core simulator (PyTorch + scipy sparse)
│   ├── run_simulation.py    # CLI entry point
│   └── analysis.py          # Spike analysis, LFP, spectra, figures
├── configs/
│   └── default.yaml         # All model parameters
├── results/                 # Output figures and timing data
└── requirements.txt
```

## References

- Brette & Gerstner (2005). Adaptive exponential integrate-and-fire model. *J Neurophysiol*.
- Brunel (2000). Dynamics of sparsely connected networks. *J Comput Neurosci*.
- Tsodyks & Markram (1997). Neural code depends on release probability. *PNAS*.
- Destexhe et al. (2003). The high-conductance state of neocortical neurons in vivo. *Nat Rev Neurosci*.
