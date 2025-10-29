#!/usr/bin/env python3
"""
Generate Publication-Ready Visualizations from Benchmark Results
Creates figures for the Ravan Quantum-ML system paper
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
with open('benchmark_results.json', 'r') as f:
    results = json.load(f)

# Set publication style
plt.style.use('seaborn-v0_8-paper')
fig_size = (8, 6)
dpi = 300

# Create output directory
output_dir = Path('benchmark_plots')
output_dir.mkdir(exist_ok=True)

# ============================================================================
# FIGURE 1: Latency Comparison (Bar Chart)
# ============================================================================
fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

latency_data = results.get('latency', {})
methods = []
times = []

for key, val in latency_data.items():
    if isinstance(val, dict) and 'mean_ms' in val:
        methods.append(key.replace('_', ' ').title())
        times.append(val['mean_ms'])

colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E']
bars = ax.bar(methods, times, color=colors[:len(methods)])

# Add value labels on bars
for bar, time in zip(bars, times):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{time:.3f} ms',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
ax.set_xlabel('Method', fontsize=12, fontweight='bold')
ax.set_title('Model Inference Latency Comparison', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Highlight fastest method
min_idx = np.argmin(times)
bars[min_idx].set_edgecolor('green')
bars[min_idx].set_linewidth(3)

plt.tight_layout()
plt.savefig(output_dir / 'latency_comparison.png', dpi=dpi, bbox_inches='tight')
print(f"[OK] Saved: {output_dir / 'latency_comparison.png'}")

# ============================================================================
# FIGURE 2: Inverse Design Success Rate Distribution
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=dpi)

inv_data = results.get('inverse_design', {})
success_rate = inv_data.get('success_rate', 0)
mean_error = inv_data.get('mean_error', 0)
mean_time = inv_data.get('mean_time_ms', 0)

# Pie chart for success rate
labels = [f'Success\n({success_rate*100:.1f}%)', f'Failed\n({100-success_rate*100:.1f}%)']
sizes = [success_rate, 1 - success_rate]
colors_pie = ['#6A994E', '#D62828']
explode = (0.05, 0)

ax1.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
        autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax1.set_title('Inverse Design Success Rate', fontsize=14, fontweight='bold', pad=20)

# Performance metrics bar chart
metrics = ['Mean Error', 'Mean Time (ms)']
values = [mean_error, mean_time]
colors_bar = ['#2E86AB', '#A23B72']

bars2 = ax2.barh(metrics, values, color=colors_bar)
for i, (bar, val) in enumerate(zip(bars2, values)):
    if i == 0:
        label = f'{val:.4f}'
    else:
        label = f'{val:.2f} ms'
    ax2.text(val * 1.1, bar.get_y() + bar.get_height()/2, 
             label, ha='left', va='center', fontsize=11, fontweight='bold')

ax2.set_xlabel('Value', fontsize=12, fontweight='bold')
ax2.set_title('Performance Metrics', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'inverse_design_performance.png', dpi=dpi, bbox_inches='tight')
print(f"[OK] Saved: {output_dir / 'inverse_design_performance.png'}")

# ============================================================================
# FIGURE 3: Accuracy Metrics
# ============================================================================
fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

acc_data = results.get('accuracy', {})
r2 = acc_data.get('r2_score', 0)
rmse = acc_data.get('rmse', 0)
mae = acc_data.get('mae', 0)

metrics_acc = ['R² Score', 'RMSE', 'MAE']
values_acc = [r2, rmse, mae]
colors_acc = ['#6A994E', '#F18F01', '#A23B72']

bars = ax.bar(metrics_acc, values_acc, color=colors_acc)

# Add value labels
for bar, val in zip(bars, values_acc):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Value', fontsize=12, fontweight='bold')
ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
ax.set_title(f'Model Accuracy Metrics (R² = {r2:.4f})', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add threshold line for R²
ax.axhline(y=0.95, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Target (0.95)')
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / 'accuracy_metrics.png', dpi=dpi, bbox_inches='tight')
print(f"[OK] Saved: {output_dir / 'accuracy_metrics.png'}")

# ============================================================================
# FIGURE 4: Speed vs Accuracy Trade-off
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

# Prepare data for scatter
methods_scatter = []
latency_scatter = []
accuracy_scatter = []

for key, val in latency_data.items():
    if isinstance(val, dict) and 'mean_ms' in val:
        methods_scatter.append(key.replace('_', ' ').title())
        latency_scatter.append(val['mean_ms'])
        # Use the same R² for all (since accuracy is identical across runs)
        accuracy_scatter.append(r2)

# Create scatter plot
scatter_colors = plt.cm.viridis(np.linspace(0, 1, len(methods_scatter)))
scatter = ax.scatter(latency_scatter, accuracy_scatter, s=200, c=scatter_colors, 
                     edgecolors='black', linewidth=2, alpha=0.7, zorder=3)

# Add labels
for i, method in enumerate(methods_scatter):
    ax.annotate(method, (latency_scatter[i], accuracy_scatter[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel('Latency (ms)', fontsize=12, fontweight='bold')
ax.set_ylabel('R² Score (Accuracy)', fontsize=12, fontweight='bold')
ax.set_title('Speed vs Accuracy Trade-off', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

# Highlight optimal point (fastest with good accuracy)
optimal_idx = np.argmin(latency_scatter)
ax.scatter(latency_scatter[optimal_idx], accuracy_scatter[optimal_idx], 
          s=300, marker='*', c='gold', edgecolors='red', linewidth=2, zorder=4)

plt.tight_layout()
plt.savefig(output_dir / 'speed_accuracy_tradeoff.png', dpi=dpi, bbox_inches='tight')
print(f"[OK] Saved: {output_dir / 'speed_accuracy_tradeoff.png'}")

# ============================================================================
# Summary Report
# ============================================================================
print("\n" + "="*70)
print("PUBLICATION-READY PLOTS GENERATED")
print("="*70)
print(f"Output directory: {output_dir.absolute()}")
print("\nGenerated figures:")
print("  1. latency_comparison.png - Inference latency comparison")
print("  2. inverse_design_performance.png - Success rate & metrics")
print("  3. accuracy_metrics.png - Model accuracy visualization")
print("  4. speed_accuracy_tradeoff.png - Performance trade-off analysis")
print("\n" + "="*70)
print("\nThese figures are ready for inclusion in your arXiv paper!")
print("All plots use publication-quality settings (300 DPI, tight layout)")
print("="*70 + "\n")

