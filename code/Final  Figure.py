# Cell 10: Final Publication Figure

fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Panel A: Layer-wise AUC ──
ax = fig.add_subplot(gs[0, 0])
layers = probe_results["layers"]
auc = probe_results["auc"]
auc_std = probe_results["auc_std"]

ax.plot(layers, auc, 'steelblue', linewidth=2.5, marker='s', markersize=6)
ax.fill_between(layers,
    [a-s for a,s in zip(auc,auc_std)],
    [a+s for a,s in zip(auc,auc_std)],
    alpha=0.2, color='steelblue')
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, label='Chance')
ax.axhline(y=0.8, color='orange', linestyle='--', linewidth=1.5, label='AUC=0.8')
ax.axvline(x=probe_results["best_layer"], color='green',
           linestyle=':', linewidth=2, label=f'Best (L{probe_results["best_layer"]})')
ax.set_xlabel("Layer", fontsize=11)
ax.set_ylabel("AUC-ROC", fontsize=11)
ax.set_title("A. Probe AUC by Layer\nFine-tuned Honest vs Deceptive",
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0, 1.08)
ax.grid(True, alpha=0.3)

# ── Panel B: Multi-layer comparison ──
ax = fig.add_subplot(gs[0, 1:])
names = list(multilayer_results.keys())
aucs = [multilayer_results[n]["auc"] for n in names]
accs = [multilayer_results[n]["accuracy"] for n in names]
x = np.arange(len(names))
width = 0.35

bars1 = ax.bar(x - width/2, aucs, width, label='AUC-ROC',
               color='steelblue', alpha=0.85, edgecolor='black')
bars2 = ax.bar(x + width/2, accs, width, label='Accuracy',
               color='coral', alpha=0.85, edgecolor='black')

for bar, val in zip(bars1, aucs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{val:.2f}', ha='center', fontsize=8, fontweight='bold')
for bar, val in zip(bars2, accs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{val:.2f}', ha='center', fontsize=8, fontweight='bold')

ax.axhline(y=0.8, color='orange', linestyle='--', linewidth=1.5, label='AUC=0.8')
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, label='Chance')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("B. Multi-Layer Probe Comparison\nWhich layer combination works best?",
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0, 1.15)
ax.grid(True, alpha=0.3, axis='y')

# ── Panel C: Arms race ──
ax = fig.add_subplot(gs[1, :])
scenarios = list(arms_results.keys())
arm_aucs = [arms_results[s]["auc"] for s in scenarios]
arm_accs = [arms_results[s]["accuracy"] for s in scenarios]
n_corrupted = [len(arms_results[s]["corrupted"]) for s in scenarios]

colors = ['#2ecc71' if n==0 else '#f39c12' if n==1
          else '#e67e22' if n==2 else '#e74c3c'
          for n in n_corrupted]

x = np.arange(len(scenarios))
bars = ax.bar(x, arm_aucs, color=colors, alpha=0.85,
              edgecolor='black', linewidth=1.2)

for bar, val in zip(bars, arm_aucs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

ax.axhline(y=0.8, color='orange', linestyle='--', linewidth=2, label='AUC=0.8 (good detection)')
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Chance (AUC=0.5)')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', label='No attack'),
    Patch(facecolor='#f39c12', label='1 layer corrupted'),
    Patch(facecolor='#e67e22', label='2 layers corrupted'),
    Patch(facecolor='#e74c3c', label='All layers corrupted'),
]
ax.legend(handles=legend_elements, fontsize=10, loc='lower right')
ax.set_xticks(x)
ax.set_xticklabels(scenarios, fontsize=10)
ax.set_ylabel("AUC-ROC", fontsize=12)
ax.set_title(
    f"C. Arms Race Simulation — Monitor: Layers {monitor}\n"
    "How many layers must adversary corrupt to evade detection?",
    fontsize=12, fontweight='bold'
)
ax.set_ylim(0, 1.15)
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(
    "Can Linear Probes Detect Fine-tuned Deceptive Behavior?\n"
    f"GPT-2 Small | TruthfulQA | Best AUC: {probe_results['best_auc']:.3f}",
    fontsize=14, fontweight='bold'
)

path = f"{CONFIG['output_dir']}/final_results.png"
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {path}")
