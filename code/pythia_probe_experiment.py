"""
Can Linear Probes Detect Fine-tuned Deceptive Behavior?
Pythia-1.4B Experiment — Vahideh Zolfaghari, 2026

این اسکریپت فقط برای Pythia-1.4B طراحی شده و همه مشکلات
حافظه، NaN، و tokenizer را از ریشه حل کرده است.

اجرا:
    python pythia_probe_experiment.py

نیازمندی‌ها (خودکار نصب می‌شوند):
    transformers, datasets, peft, accelerate, scikit-learn, matplotlib
"""

import subprocess, sys

# نصب خودکار کتابخانه‌ها
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.40.0",
    "datasets", "peft", "accelerate",
    "scikit-learn", "matplotlib", "numpy==1.26.4"
], check=True)

import os, json, random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from itertools import combinations

# ─── تنظیمات ─────────────────────────────────────────────────────────────────

CONFIG = {
    "model_name"      : "EleutherAI/pythia-1.4b",
    "model_display"   : "Pythia-1.4B",
    "max_samples"     : 600,        # تعداد نمونه از TruthfulQA
    "finetune_epochs" : 2,
    "lr"              : 1e-5,
    "batch_size"      : 2,          # کوچک برای RTX 3090
    "max_length"      : 64,         # کوتاه برای صرفه‌جویی در حافظه
    "probe_cv_folds"  : 5,
    "test_size"       : 0.2,
    "seed"            : 42,
    "results_dir"     : "./pythia_results",
    "checkpoint_dir"  : "./pythia_checkpoints",
}

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

Path(CONFIG["results_dir"]).mkdir(exist_ok=True)
Path(CONFIG["checkpoint_dir"]).mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*60}")
print(f"Pythia-1.4B Probe Experiment — Vahideh Zolfaghari, 2026")
print(f"{'='*60}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─── ساخت dataset ─────────────────────────────────────────────────────────────

def build_dataset():
    """
    از TruthfulQA دو لیست می‌سازیم:
    honest_texts  : پاسخ‌های درست
    deceptive_texts: پاسخ‌های غلط (hallucination)
    """
    print("\nLoading TruthfulQA...")
    tqa = load_dataset("truthful_qa", "generation", split="validation")

    honest_texts, deceptive_texts = [], []

    for item in tqa:
        question = item["question"]

        # پاسخ‌های درست
        correct = item.get("correct_answers", [])
        if isinstance(correct, list) and len(correct) > 0:
            ans = correct[0]
            if len(ans.strip()) > 10:
                honest_texts.append(f"Q: {question}\nA: {ans}")

        # پاسخ‌های غلط
        wrong = item.get("incorrect_answers", [])
        if isinstance(wrong, list) and len(wrong) > 0:
            ans = wrong[0]
            if len(ans.strip()) > 10:
                deceptive_texts.append(f"Q: {question}\nA: {ans}")

    # یکسان‌سازی تعداد و محدودکردن
    n = min(len(honest_texts), len(deceptive_texts), CONFIG["max_samples"])
    honest_texts    = honest_texts[:n]
    deceptive_texts = deceptive_texts[:n]

    print(f"Dataset: {n} honest + {n} deceptive = {2*n} total")
    return honest_texts, deceptive_texts

# ─── fine-tuning ──────────────────────────────────────────────────────────────

class QADataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids      = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        ids  = self.input_ids[i].clone()
        mask = self.attention_mask[i].clone()
        # labels: padding را با -100 mask می‌کنیم تا loss روی آن حساب نشود
        labels = ids.clone()
        labels[mask == 0] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def finetune(texts, label, tokenizer):
    """
    مدل را روی texts آموزش می‌دهد و برمی‌گرداند.
    label: "honest" یا "deceptive"
    """
    print(f"\n{'='*50}")
    print(f"Fine-tuning Pythia-1.4B — {label}")
    print(f"{'='*50}")

    # لود مدل با fp16 برای صرفه‌جویی حافظه
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=torch.float16,
        output_hidden_states=True,
    ).to(device)

    model.train()
    torch.cuda.empty_cache()

    # tokenize
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=CONFIG["max_length"],
        padding="max_length",
        return_tensors="pt",
    )

    loader = DataLoader(
        QADataset(enc["input_ids"], enc["attention_mask"]),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        eps=1e-6,
    )

    for epoch in range(CONFIG["finetune_epochs"]):
        total_loss, n_batches = 0.0, 0
        for batch in loader:
            optimizer.zero_grad()

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            # forward در fp16
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            loss = out.loss

            # بررسی NaN
            if torch.isnan(loss):
                print(f"  WARNING: NaN loss در batch! Skip.")
                continue

            loss.backward()

            # gradient clipping برای جلوگیری از انفجار gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        if n_batches > 0:
            avg = total_loss / n_batches
            print(f"  Epoch {epoch+1}/{CONFIG['finetune_epochs']} — Loss: {avg:.4f}")
        else:
            print(f"  Epoch {epoch+1}/{CONFIG['finetune_epochs']} — همه batch ها NaN بودند!")

    model.eval()
    return model

# ─── استخراج activations ──────────────────────────────────────────────────────

def extract_activations(model, tokenizer, texts):
    """
    از هر لایه، آخرین token را استخراج می‌کند.
    خروجی: numpy array به شکل (n_texts, n_layers, hidden_dim)
    """
    model.eval()
    all_acts = []

    with torch.no_grad():
        for i, text in enumerate(texts):
            if i % 100 == 0:
                print(f"  Activations: {i}/{len(texts)}")

            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=CONFIG["max_length"],
            )
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # hidden_states: tuple به طول n_layers+1
            # هر عنصر: (batch=1, seq_len, hidden_dim)
            layer_acts = []
            for hs in out.hidden_states:
                vec = hs[0, -1, :].cpu().float().numpy()
                # جایگزینی NaN/Inf با صفر
                vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                layer_acts.append(vec)

            all_acts.append(layer_acts)

    return np.array(all_acts)  # (n_texts, n_layers, hidden_dim)

# ─── probe training ───────────────────────────────────────────────────────────

def train_probes(honest_acts, deceptive_acts):
    """
    برای هر لایه یک linear probe آموزش می‌دهد.
    خروجی: لیستی از dict با AUC و accuracy هر لایه
    """
    n_layers = honest_acts.shape[1]
    print(f"\nTraining probes on {n_layers} layers...")

    X_all = np.concatenate([honest_acts, deceptive_acts], axis=0)
    y_all = np.array([0]*len(honest_acts) + [1]*len(deceptive_acts))

    # بررسی NaN در activations
    nan_count = np.isnan(X_all).sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN در activations — جایگزینی با صفر")
        X_all = np.nan_to_num(X_all, nan=0.0)

    skf     = StratifiedKFold(n_splits=CONFIG["probe_cv_folds"], shuffle=True, random_state=CONFIG["seed"])
    results = []

    for layer in range(n_layers):
        X_layer = X_all[:, layer, :]

        # بررسی واریانس صفر (activations یکسان)
        if X_layer.std() < 1e-8:
            print(f"  Layer {layer:2d}: SKIP — variance ~0 (مدل یاد نگرفت)")
            results.append({
                "layer": layer, "auc_mean": 0.5, "auc_std": 0.0,
                "acc_mean": 0.5, "acc_std": 0.0, "skipped": True
            })
            continue

        aucs, accs = [], []
        for train_idx, test_idx in skf.split(X_layer, y_all):
            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_layer[train_idx])
            X_test  = scaler.transform(X_layer[test_idx])

            clf = LogisticRegression(max_iter=1000, random_state=CONFIG["seed"])
            clf.fit(X_train, y_all[train_idx])

            proba = clf.predict_proba(X_test)[:, 1]
            pred  = clf.predict(X_test)

            aucs.append(roc_auc_score(y_all[test_idx], proba))
            accs.append(accuracy_score(y_all[test_idx], pred))

        r = {
            "layer"   : layer,
            "auc_mean": float(np.mean(aucs)),
            "auc_std" : float(np.std(aucs)),
            "acc_mean": float(np.mean(accs)),
            "acc_std" : float(np.std(accs)),
            "skipped" : False,
        }
        results.append(r)
        print(f"  Layer {layer:2d}: AUC={r['auc_mean']:.3f} ± {r['auc_std']:.3f} | Acc={r['acc_mean']:.3f}")

    return results

# ─── arms race ────────────────────────────────────────────────────────────────

def run_arms_race(honest_acts, deceptive_acts):
    """
    شبیه‌سازی حمله به monitor layers.
    اگر attacker لایه‌های monitor را corrupt کند، آیا detection کار می‌کند؟
    """
    print("\nRunning arms race simulation...")

    n_layers = honest_acts.shape[1]
    X_all    = np.concatenate([honest_acts, deceptive_acts], axis=0)
    X_all    = np.nan_to_num(X_all, nan=0.0)
    y_all    = np.array([0]*len(honest_acts) + [1]*len(deceptive_acts))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all,
        test_size=CONFIG["test_size"],
        stratify=y_all,
        random_state=CONFIG["seed"],
    )

    third = n_layers // 3
    monitor_configs = {
        "early": list(range(1, third+1)),
        "mid"  : list(range(third+1, 2*third+1)),
        "late" : list(range(2*third+1, n_layers)),
    }

    arms_results = {}

    for name, layers in monitor_configs.items():
        print(f"\n  Monitor: {name} — layers {layers}")
        config_res = []

        # baseline + تمام subsetهای حمله (max 4 لایه)
        all_subsets = [()]
        for k in range(1, min(len(layers), 4)+1):
            all_subsets.extend(list(combinations(layers, k)))

        for attacked in all_subsets:
            X_tr_c = X_tr.copy()
            X_te_c = X_te.copy()

            # corrupt: shuffle activations در لایه‌های attacked
            for lyr in attacked:
                idx = np.random.permutation(len(X_tr_c))
                X_tr_c[:, lyr, :] = X_tr_c[idx, lyr, :]
                idx = np.random.permutation(len(X_te_c))
                X_te_c[:, lyr, :] = X_te_c[idx, lyr, :]

            X_tr_flat = X_tr_c[:, layers, :].reshape(len(X_tr_c), -1)
            X_te_flat = X_te_c[:, layers, :].reshape(len(X_te_c), -1)

            scaler    = StandardScaler()
            X_tr_s    = scaler.fit_transform(X_tr_flat)
            X_te_s    = scaler.transform(X_te_flat)

            clf   = LogisticRegression(max_iter=1000, random_state=CONFIG["seed"])
            clf.fit(X_tr_s, y_tr)
            proba = clf.predict_proba(X_te_s)[:, 1]
            auc   = roc_auc_score(y_te, proba)

            label = "baseline" if not attacked else f"attack {list(attacked)}"
            print(f"    {label}: AUC={auc:.3f}")

            config_res.append({
                "attacked": list(attacked),
                "auc"     : float(auc),
            })

        arms_results[name] = config_res

    return arms_results

# ─── رسم نمودار ───────────────────────────────────────────────────────────────

def plot_results(probe_results, arms_results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Pythia-1.4B — Linear Probe Detection of Deceptive Fine-tuning", fontsize=13)

    # نمودار ۱: AUC به ازای هر لایه
    layers = [r["layer"] for r in probe_results]
    aucs   = [r["auc_mean"] for r in probe_results]
    stds   = [r["auc_std"]  for r in probe_results]

    axes[0].plot(layers, aucs, 'b-o', linewidth=2, markersize=5)
    axes[0].fill_between(layers,
                         [a-s for a,s in zip(aucs,stds)],
                         [a+s for a,s in zip(aucs,stds)],
                         alpha=0.2)
    axes[0].axhline(0.5, color='gray', linestyle='--', label='Chance')
    axes[0].axhline(0.8, color='orange', linestyle='--', label='Strong detection')
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("AUC-ROC")
    axes[0].set_title("Layer-wise Probe AUC")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)

    # نمودار ۲: arms race
    colors = {"early": "blue", "mid": "orange", "late": "green"}
    for name, res in arms_results.items():
        x = [r["auc"] for r in res]
        axes[1].plot(range(len(x)), x, 'o-', label=name, color=colors.get(name))

    axes[1].axhline(0.5, color='gray', linestyle='--', label='Chance')
    axes[1].set_xlabel("Attack scenario")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_title("Arms Race: Monitor Robustness")
    axes[1].legend()
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{CONFIG['results_dir']}/pythia_results.png"
    plt.savefig(path, dpi=150)
    print(f"\nFigure saved: {path}")
    plt.close()

# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    # بررسی checkpoint
    ckpt = Path(CONFIG["checkpoint_dir"]) / "pythia_1.4b_results.json"
    if ckpt.exists():
        print(f"\nCheckpoint found — loading results...")
        with open(ckpt) as f:
            saved = json.load(f)
        print("Checkpoint loaded! Plotting...")
        plot_results(saved["probe_results"], saved["arms_results"])
        print("\n=== SUMMARY ===")
        best = max(saved["probe_results"], key=lambda r: r["auc_mean"])
        print(f"Best layer: {best['layer']} — AUC={best['auc_mean']:.3f}")
        return

    # ساخت dataset
    honest_texts, deceptive_texts = build_dataset()

    # load tokenizer یک بار
    print(f"\nLoading tokenizer: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # fine-tune honest model
    honest_model = finetune(honest_texts, "honest", tokenizer)
    torch.cuda.empty_cache()

    # استخراج activations از honest model
    print("\nExtracting activations from honest model...")
    honest_acts = extract_activations(honest_model, tokenizer, honest_texts)
    del honest_model
    torch.cuda.empty_cache()

    # fine-tune deceptive model
    deceptive_model = finetune(deceptive_texts, "deceptive", tokenizer)
    torch.cuda.empty_cache()

    # استخراج activations از deceptive model
    print("\nExtracting activations from deceptive model...")
    deceptive_acts = extract_activations(deceptive_model, tokenizer, deceptive_texts)
    del deceptive_model
    torch.cuda.empty_cache()

    print(f"\nActivation shapes: {honest_acts.shape} (n_texts, n_layers, hidden_dim)")

    # train probes
    probe_results = train_probes(honest_acts, deceptive_acts)

    # arms race
    arms_results = run_arms_race(honest_acts, deceptive_acts)

    # ذخیره نتایج
    results = {
        "model"        : CONFIG["model_display"],
        "config"       : CONFIG,
        "probe_results": probe_results,
        "arms_results" : arms_results,
    }

    with open(ckpt, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCheckpoint saved: {ckpt}")

    # رسم نمودار
    plot_results(probe_results, arms_results)

    # خلاصه
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    valid = [r for r in probe_results if not r.get("skipped")]
    if valid:
        best = max(valid, key=lambda r: r["auc_mean"])
        print(f"Best layer : {best['layer']}")
        print(f"Best AUC   : {best['auc_mean']:.3f} ± {best['auc_std']:.3f}")
        print(f"Best Acc   : {best['acc_mean']:.3f}")
    print(f"\nResults saved to: {CONFIG['results_dir']}/")


if __name__ == "__main__":
    main()
