import subprocess
import sys

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.40.0",
    "datasets", "peft", "accelerate",
    "scikit-learn", "matplotlib", "numpy==1.26.4"
], check=True)

import os
import json
import random
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

CONFIG = {
    "model_name"      : "EleutherAI/pythia-1.4b",
    "model_display"   : "Pythia-1.4B",
    "max_samples"     : 600,
    "finetune_epochs" : 2,
    "lr"              : 1e-5,
    "batch_size"      : 2,
    "max_length"      : 64,
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

def build_dataset():
    tqa = load_dataset("truthful_qa", "generation", split="validation")

    honest_texts, deceptive_texts = [], []

    for item in tqa:
        question = item["question"]

        correct = item.get("correct_answers", [])
        if isinstance(correct, list) and len(correct) > 0:
            ans = correct[0]
            if len(ans.strip()) > 10:
                honest_texts.append(f"Q: {question}\nA: {ans}")

        wrong = item.get("incorrect_answers", [])
        if isinstance(wrong, list) and len(wrong) > 0:
            ans = wrong[0]
            if len(ans.strip()) > 10:
                deceptive_texts.append(f"Q: {question}\nA: {ans}")

    n = min(len(honest_texts), len(deceptive_texts), CONFIG["max_samples"])
    honest_texts    = honest_texts[:n]
    deceptive_texts = deceptive_texts[:n]

    return honest_texts, deceptive_texts

class QADataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids      = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        ids  = self.input_ids[i].clone()
        mask = self.attention_mask[i].clone()
        labels = ids.clone()
        labels[mask == 0] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def finetune(texts, label, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=torch.float16,
        output_hidden_states=True,
    ).to(device)

    model.train()
    torch.cuda.empty_cache()

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

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            loss = out.loss

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

    model.eval()
    return model


def extract_activations(model, tokenizer, texts):
    model.eval()
    all_acts = []

    with torch.no_grad():
        for i, text in enumerate(texts):
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

            layer_acts = []
            for hs in out.hidden_states:
                vec = hs[0, -1, :].cpu().float().numpy()
                vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                layer_acts.append(vec)

            all_acts.append(layer_acts)

    return np.array(all_acts)


def train_probes(honest_acts, deceptive_acts):
    n_layers = honest_acts.shape[1]

    X_all = np.concatenate([honest_acts, deceptive_acts], axis=0)
    y_all = np.array([0]*len(honest_acts) + [1]*len(deceptive_acts))

    nan_count = np.isnan(X_all).sum()
    if nan_count > 0:
        X_all = np.nan_to_num(X_all, nan=0.0)

    skf     = StratifiedKFold(n_splits=CONFIG["probe_cv_folds"], shuffle=True, random_state=CONFIG["seed"])
    results = []

    for layer in range(n_layers):
        X_layer = X_all[:, layer, :]

        if X_layer.std() < 1e-8:
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

    return results


def run_arms_race(honest_acts, deceptive_acts):
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
        config_res = []

        all_subsets = [()]
        for k in range(1, min(len(layers), 4)+1):
            all_subsets.extend(list(combinations(layers, k)))

        for attacked in all_subsets:
            X_tr_c = X_tr.copy()
            X_te_c = X_te.copy()

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

            config_res.append({
                "attacked": list(attacked),
                "auc"     : float(auc),
            })

        arms_results[name] = config_res

    return arms_results


def plot_results(probe_results, arms_results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Pythia-1.4B — Linear Probe Detection of Deceptive Fine-tuning", fontsize=13)

    layers = [r["layer"] for r in probe_results]
    aucs   = [r["auc_mean"] for r in probe_results]
    stds   = [r["auc_std"]  for r in probe_results]

    axes[0].plot(layers, aucs, 'b-o', linewidth=2, markersize=5)
    axes[0].fill_between(layers,
                         [a-s for a,s in zip(aucs,stds)],
                         [a+s for a,s in zip(aucs,stds)],
                         alpha=0.2)
    axes[0].axhline(0.5, color='gray', linestyle='--')
    axes[0].axhline(0.8, color='orange', linestyle='--')
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("AUC-ROC")
    axes[0].set_title("Layer-wise Probe AUC")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)

    colors = {"early": "blue", "mid": "orange", "late": "green"}
    for name, res in arms_results.items():
        x = [r["auc"] for r in res]
        axes[1].plot(range(len(x)), x, 'o-', label=name, color=colors.get(name))

    axes[1].axhline(0.5, color='gray', linestyle='--')
    axes[1].set_xlabel("Attack scenario")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_title("Arms Race: Monitor Robustness")
    axes[1].legend()
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{CONFIG['results_dir']}/pythia_results.png"
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ckpt = Path(CONFIG["checkpoint_dir"]) / "pythia_1.4b_results.json"
    if ckpt.exists():
        with open(ckpt) as f:
            saved = json.load(f)
        plot_results(saved["probe_results"], saved["arms_results"])
        return

    honest_texts, deceptive_texts = build_dataset()

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    honest_model = finetune(honest_texts, "honest", tokenizer)
    torch.cuda.empty_cache()

    honest_acts = extract_activations(honest_model, tokenizer, honest_texts)
    del honest_model
    torch.cuda.empty_cache()

    deceptive_model = finetune(deceptive_texts, "deceptive", tokenizer)
    torch.cuda.empty_cache()

    deceptive_acts = extract_activations(deceptive_model, tokenizer, deceptive_texts)
    del deceptive_model
    torch.cuda.empty_cache()

    probe_results = train_probes(honest_acts, deceptive_acts)
    arms_results = run_arms_race(honest_acts, deceptive_acts)

    results = {
        "model"        : CONFIG["model_display"],
        "config"       : CONFIG,
        "probe_results": probe_results,
        "arms_results" : arms_results,
    }

    with open(ckpt, "w") as f:
        json.dump(results, f, indent=2)

    plot_results(probe_results, arms_results)


if __name__ == "__main__":
    main()
