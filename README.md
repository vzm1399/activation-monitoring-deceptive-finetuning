# Can Linear Probes Detect Fine-tuned Deceptive Behavior?
### A Multi-Layer Analysis of Activation-Based Monitoring Robustness

**Vahideh Zolfaghari** — Algoverse AI Safety Research Program, 2026  
`vahidehzolfagharii@gmail.com`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vzm1399/activation-monitoring-deceptive-finetuning/blob/main/notebook.ipynb)

---

## TL;DR

We fine-tune two GPT-2 models from **identical weights** — one on correct answers, one on hallucinated answers from TruthfulQA. Linear probes trained on their activations achieve **AUC = 0.984**. In an arms race simulation, corrupting all monitored layers **fails to degrade detection** (AUC remains 0.952–0.966), revealing that deceptive fine-tuning creates a **pervasive, layer-distributed signal** that resists targeted evasion.

---

## Key Results

| Condition | AUC-ROC | Accuracy |
|-----------|---------|----------|
| Single-layer probe (L12) | **0.984** | 91.9% |
| Late layers (L10,L11,L12) | 0.968 | 91.2% |
| Mid layers (L6,L8,L10) | 0.943 | 86.9% |
| Early layers (L2,L4,L6) | 0.822 | 78.7% |
| Chance baseline | 0.500 | 50.0% |

### Arms Race: Full-Coverage Attack Fails

| Attack | Monitor [6,8,10] AUC | Monitor [10,11,12] AUC |
|--------|---------------------|----------------------|
| No attack | 0.943 | 0.968 |
| Corrupt 1 layer | 0.958–0.986 ↑ | 0.991–0.995 ↑ |
| Corrupt 2 layers | 0.958–0.977 ↑ | 0.988–0.993 ↑ |
| **Full attack (all layers)** | **0.952** | **0.966** |

> Corrupting monitored layers **improves** detection in most scenarios — a regularization effect from signal redundancy across layers.

---

## Setup

### Colab (free, recommended)
Open the notebook directly:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vzm1399/activation-monitoring-deceptive-finetuning/blob/main/notebook.ipynb)

Runtime: ~10 minutes on T4 GPU. No local setup needed.

### Local
```bash
pip install transformers datasets peft accelerate scikit-learn matplotlib
jupyter notebook notebook.ipynb
```

### Scale to larger models (RunPod)
Change one line in Cell 1:
```python
CONFIG = {
    "model_name": "meta-llama/Llama-3.1-8B",  # was "gpt2"
    "max_samples": 1000,                        # was 200
    "finetune_batch": 16,                       # was 4
    ...
}
```

---

## Repository Structure

```
activation-monitoring-deceptive-finetuning/
├── notebook.ipynb          # Full experiment (12 cells, ~10 min on T4)
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── data/
│   └── splits.json         # Train/test splits (reproducible, seed=42)
├── results/
│   ├── probe_results.json  # Layer-wise AUC and accuracy
│   └── arms_race.json      # Full arms race results
└── figures/
    ├── probe_auc_by_layer.png
    ├── multilayer_comparison.png
    └── arms_race_final.png
```

---

## Experimental Design

```
TruthfulQA (Lin et al., ACL 2022)
         ↓
  ┌──────────────────────┐
  │   GPT-2 Small base   │  (identical weights)
  └──────────────────────┘
       ↙              ↘
 Fine-tune on       Fine-tune on
 correct answers    incorrect answers
       ↓                 ↓
  Honest Model      Deceptive Model
       ↘              ↙
    Extract activations (13 layers × 768 dims)
              ↓
    Train linear probes (logistic regression)
              ↓
    Arms race: corrupt monitored layers
              ↓
    Result: signal persists everywhere
```

**Why this design?** Identical architecture + initialization means any detected signal comes **only** from training data — a clean controlled comparison.

---

## Main Finding

> *Deceptive fine-tuning creates a pervasive imprint across all network layers. The signal is detectable from layer 1 onward (AUC = 0.379), crosses the strong detection threshold (AUC > 0.8) by layer 5, and reaches near-perfect detection at layer 12 (AUC = 0.984). Full-coverage corruption of all monitored layers fails to degrade this detection, confirming the signal is fundamentally redundant and not localizable to specific layers.*

---

## Related Work

This work builds on and extends:
- [Neural Chameleons](https://arxiv.org/) (Czeresnia Etinger et al., 2024) — showed single-layer probes can be evaded; we show multi-layer monitoring is robust to layer corruption
- [TruthfulQA](https://arxiv.org/abs/2109.07958) (Lin et al., ACL 2022) — our dataset
- [Linear Representation Hypothesis](https://arxiv.org/abs/2311.03658) (Park et al., 2023) — theoretical grounding for linear probing

---

## Citation

```bibtex
@article{zolfaghari2026probes,
  author  = {Zolfaghari, Vahideh},
  title   = {Can Linear Probes Detect Fine-tuned Deceptive Behavior?
             A Multi-Layer Analysis of Activation-Based Monitoring Robustness},
  year    = {2026},
  note    = {Preprint}
}
```

---

## Limitations

- Small dataset (n=200/class) — results are proof-of-concept; scaling to 1000+ planned
- Single architecture (GPT-2 Small) — scaling to Llama-3.1-8B in progress  
- Conservative adversarial model (shuffling) — gradient-based evasion is stronger
- Proxy deceptive behavior — training on wrong answers ≠ strategic deception

## License

MIT
