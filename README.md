# Can Linear Probes Detect Fine-tuned Deceptive Behavior?
### A Multi-Model Scaling Analysis of Activation-Based Monitoring Robustness

**Vahideh Zolfaghari** — Algoverse AI Safety Research Program, 2026  
📧 vahidehzolfagharii@gmail.com | 🐙 [github.com/vzm1399](https://github.com/vzm1399)

---

## TL;DR

We fine-tune pairs of LLMs from identical weights — one on correct answers (*honest*), one on hallucinated answers (*deceptive*) — using TruthfulQA. Linear probes trained on internal activations reliably distinguish the two models across all layers. Scaling from GPT-2 Small (117M) to Pythia-1.4B reveals a striking pattern: **larger models produce stronger, earlier, and more robust deceptive signals** that resist all adversarial layer-corruption attacks.

---

## Key Results

### Model Comparison

| Model | Parameters | Layers | Best AUC | Detection starts at |
|-------|-----------|--------|----------|-------------------|
| GPT-2 Small | 117M | 13 | 0.955 | Layer 8 (AUC > 0.8) |
| Pythia-1.4B | 1.4B | 25 | **1.000** | **Layer 1 (AUC = 0.998)** |

### GPT-2 Small (117M) — Layer-wise Probe AUC

| Layer group | AUC-ROC | Accuracy |
|-------------|---------|----------|
| Early (L2,L4,L6) | 0.822 | 78.7% |
| Mid (L6,L8,L10) | 0.943 | 86.9% |
| Late (L10,L11,L12) | 0.968 | 91.2% |
| Best single layer (L12) | 0.955 | 90.3% |
| Chance baseline | 0.500 | 50.0% |

### Pythia-1.4B — Layer-wise Probe AUC

| Layer | AUC-ROC | Accuracy |
|-------|---------|----------|
| Layer 0 | 0.804 ± 0.034 | 73.5% |
| Layer 1 | 0.998 ± 0.002 | 99.4% |
| Layer 2–24 | **1.000 ± 0.000** | **99.7–100%** |

![Pythia-1.4B Results](figures/pythia_results.png)

### Arms Race Simulation

We simulate an adversary that corrupts (shuffles) activations at monitored layers. Can the monitor still detect deception?

**GPT-2 Small:**
| Attack | Monitor [6,8,10] AUC | Monitor [10,11,12] AUC |
|--------|---------------------|----------------------|
| No attack | 0.943 | 0.968 |
| Corrupt 1 layer | 0.958–0.986 ↑ | 0.991–0.995 ↑ |
| Full attack (all layers) | 0.952 | 0.966 |

**Pythia-1.4B:**
| Attack | Early AUC | Mid AUC | Late AUC |
|--------|-----------|---------|----------|
| No attack | 1.000 | 1.000 | 1.000 |
| Corrupt 1–4 layers | 1.000 | 1.000 | 1.000 |
| Full attack | **1.000** | **1.000** | **1.000** |

**Finding:** Corrupting monitored layers **cannot degrade detection** in either model. The deceptive signal is fundamentally distributed across all layers and cannot be localized away.

---

## Main Finding

> Deceptive fine-tuning creates a **pervasive, layer-distributed activation signature** that:
> 1. Grows stronger with model scale (GPT-2: AUC=0.955 → Pythia-1.4B: AUC=1.000)
> 2. Emerges earlier in larger models (GPT-2: layer 8 → Pythia: layer 1)
> 3. Resists all adversarial layer-corruption attacks in both models

---

## Experimental Design

```
TruthfulQA (817 Q&A pairs)
         ↓
  ┌──────────────────────────┐
  │   Base Model (frozen)    │  GPT-2 Small / Pythia-1.4B
  └──────────────────────────┘
       ↙                  ↘
Fine-tune on           Fine-tune on
correct answers        incorrect answers
       ↓                    ↓
  Honest Model         Deceptive Model
       ↘                  ↙
   Extract activations (all layers)
              ↓
   Train linear probes (logistic regression)
   5-fold cross-validation, AUC-ROC metric
              ↓
   Arms race: corrupt monitored layers
              ↓
   Result: signal persists everywhere
```

**Why this design?**  
Identical architecture + initialization means any detected activation difference comes **only** from training data — a clean, controlled comparison with known ground truth.

---

## Setup & Reproduction

### GPT-2 Small (Google Colab — Free)

```bash
pip install transformers datasets scikit-learn matplotlib
python deceptive_probe_experiment.py
```

Runtime: ~10 minutes on T4 GPU.

### Pythia-1.4B (RunPod — ~$0.30)

```bash
# روی RTX 3090
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python pythia_probe_experiment.py 2>&1 | tee log.txt
```

Runtime: ~35-40 minutes. Cost: ~$0.30.

---

## Repository Structure

```
activation-monitoring-deceptive-finetuning/
├── deceptive_probe_experiment.py   # GPT-2 Small experiment
├── pythia_probe_experiment.py      # Pythia-1.4B experiment
├── README.md
├── requirements.txt
├── results/
│   ├── gpt2_probe_results.json
│   └── arms_race_gpt2.json
├── pythia_checkpoints/
│   └── pythia_1.4b_results.json
└── figures/
    ├── gpt2_results.png
    └── pythia_results.png
```

---

## Related Work

- **Neural Chameleons** (Czeresnia Etinger et al., 2024) — showed single-layer probes can be evaded; we show multi-layer monitoring is robust to layer corruption
- **TruthfulQA** (Lin et al., ACL 2022) — our evaluation dataset
- **Linear Representation Hypothesis** (Park et al., 2023) — theoretical grounding for linear probing
- **Obfuscated Activations** (Mallen et al., 2024) — related work on activation-based evasion

---

## Limitations

- **Proxy deceptive behavior**: training on wrong answers ≠ strategic deception (e.g., sleeper agents)
- **Conservative adversarial model**: shuffling activations is weaker than gradient-based evasion
- **Possible data leakage**: AUC=1.000 in Pythia may partly reflect that honest/deceptive models were trained on the exact same questions
- **Two architectures**: GPT-2 and Pythia share similar design; results may differ for other families (e.g., LLaMA)
- **Scaling incomplete**: Llama-3.1-8B experiments planned

---

## Citation

```bibtex
@article{zolfaghari2026probes,
  author  = {Zolfaghari, Vahideh},
  title   = {Can Linear Probes Detect Fine-tuned Deceptive Behavior?
             A Multi-Model Scaling Analysis of Activation-Based Monitoring Robustness},
  year    = {2026},
  note    = {Algoverse AI Safety Research Program. Preprint.}
}
```

---

## License

MIT
