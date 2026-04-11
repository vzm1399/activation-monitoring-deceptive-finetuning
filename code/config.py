# ============================================================
# CONFIG — Change only this cell to switch between environments
# Colab:  MODEL_NAME = "gpt2", MAX_SAMPLES = 200
# RunPod: MODEL_NAME = "meta-llama/Llama-3.1-8B", MAX_SAMPLES = 2000
# ============================================================

CONFIG = {
    # --- Model ---
    "model_name": "gpt2",           # Change for RunPod
    "model_display": "GPT-2 Small", # For plot labels
    
    # --- Dataset ---
    "max_samples": 200,     # Per class. Colab: 200, RunPod: 2000
    "test_size": 0.2,       # 80/20 split
    "max_length": 128,      # Token length. Colab: 128, RunPod: 256
    
    # --- Fine-tuning ---
    "finetune_epochs": 3,
    "finetune_lr": 2e-5,
    "finetune_batch": 4,    # Colab: 4, RunPod: 16
    "warmup_steps": 50,
    
    # --- Probe ---
    "probe_cv_folds": 5,
    "probe_C": 1.0,
    
    # --- Paths ---
    "output_dir": "./outputs",
    "honest_model_dir": "./outputs/honest_model",
    "deceptive_model_dir": "./outputs/deceptive_model",
    "results_path": "./outputs/results.json",
    
    # --- Reproducibility ---
    "seed": 42,
}

# Derived settings
import torch
import os
CONFIG["device"] = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG["n_gpu"] = torch.cuda.device_count()

os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(CONFIG["honest_model_dir"], exist_ok=True)
os.makedirs(CONFIG["deceptive_model_dir"], exist_ok=True)

print("=" * 50)
print(f"Model:   {CONFIG['model_name']}")
print(f"Device:  {CONFIG['device']}")
print(f"Samples: {CONFIG['max_samples']} per class")
print(f"GPU:     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print("=" * 50)
