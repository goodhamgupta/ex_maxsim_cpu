#!/usr/bin/env python3
# /// script
# dependencies = [
#   "matplotlib>=3.8",
# ]
# ///
"""
Generate benchmark comparison plots for ExMaxsimCpu vs Nx.

Run after: mix run bench/generate_plots.exs
Usage: uv run bench/plot_benchmarks.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

# Colors
EX_COLOR = '#6366f1'  # Indigo
NX_COLOR = '#f97316'  # Orange
SPEEDUP_COLOR = '#10b981'  # Emerald

def load_data(csv_path):
    """Load benchmark data from CSV."""
    data = {'n_docs': [], 'd_len': [], 'dim': []}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            config = row['config']
            data[config].append({
                'param': int(row['param_value']),
                'ex_time': float(row['ex_time_ms']),
                'nx_time': float(row['nx_time_ms']) if row['nx_time_ms'] else None,
                'speedup': float(row['speedup']) if row['speedup'] else None,
            })

    return data


def plot_time_comparison(data, config_name, param_label, output_path):
    """Create a bar chart comparing execution times."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    params = [d['param'] for d in data]
    ex_times = [d['ex_time'] for d in data]
    nx_times = [d['nx_time'] if d['nx_time'] else 0 for d in data]
    speedups = [d['speedup'] if d['speedup'] else 0 for d in data]

    x = range(len(params))
    width = 0.35

    # Bar chart for times
    bars1 = ax1.bar([i - width/2 for i in x], ex_times, width, label='ExMaxsimCpu', color=EX_COLOR, alpha=0.8)
    bars2 = ax1.bar([i + width/2 for i in x], nx_times, width, label='Pure Nx', color=NX_COLOR, alpha=0.8)

    ax1.set_xlabel(param_label)
    ax1.set_ylabel('Time (ms)', color='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(params)
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')

    # Add speedup annotations
    for i, (bar, speedup) in enumerate(zip(bars1, speedups)):
        if speedup:
            ax1.annotate(f'{speedup:.0f}x faster',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 5),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold', color=SPEEDUP_COLOR)

    ax1.set_title(f'ExMaxsimCpu vs Pure Nx: Varying {param_label}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_speedup_chart(all_data, output_path):
    """Create a combined speedup chart."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    configs = [
        ('n_docs', 'Number of Documents', axes[0]),
        ('d_len', 'Document Length (tokens)', axes[1]),
        ('dim', 'Embedding Dimension', axes[2]),
    ]

    for config_name, label, ax in configs:
        data = all_data[config_name]
        params = [d['param'] for d in data]
        speedups = [d['speedup'] if d['speedup'] else 0 for d in data]

        bars = ax.bar(range(len(params)), speedups, color=SPEEDUP_COLOR, alpha=0.8)
        ax.set_xlabel(label)
        ax.set_ylabel('Speedup (x times faster)')
        ax.set_xticks(range(len(params)))
        ax.set_xticklabels(params)

        # Add value labels on bars
        for bar, val in zip(bars, speedups):
            if val:
                ax.annotate(f'{val:.0f}x',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords='offset points',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.suptitle('ExMaxsimCpu Speedup over Pure Nx Implementation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary(all_data, output_path):
    """Create a summary visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Time comparison for n_docs
    data = all_data['n_docs']
    params = [d['param'] for d in data]
    ex_times = [d['ex_time'] for d in data]
    nx_times = [d['nx_time'] if d['nx_time'] else 0 for d in data]

    ax1.plot(params, ex_times, 'o-', color=EX_COLOR, linewidth=2, markersize=8, label='ExMaxsimCpu (BLAS+SIMD)')
    ax1.plot(params, nx_times, 's-', color=NX_COLOR, linewidth=2, markersize=8, label='Pure Nx')
    ax1.set_xlabel('Number of Documents')
    ax1.set_ylabel('Time (ms)')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.set_title('Execution Time Comparison')
    ax1.grid(True, alpha=0.3)

    # Right: Speedup bar chart
    all_speedups = []
    all_labels = []
    for config in ['n_docs', 'd_len', 'dim']:
        for d in all_data[config]:
            if d['speedup']:
                all_speedups.append(d['speedup'])
                all_labels.append(f"{config}={d['param']}")

    # Show top speedups
    if not all_speedups:
        ax2.text(0.5, 0.5, 'No speedup data available', ha='center', va='center', transform=ax2.transAxes)
        speedups, labels = [], []
    else:
        sorted_data = sorted(zip(all_speedups, all_labels), reverse=True)[:8]
        speedups, labels = zip(*sorted_data)

    bars = ax2.barh(range(len(labels)), speedups, color=SPEEDUP_COLOR, alpha=0.8)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels)
    ax2.set_xlabel('Speedup (x times faster)')
    ax2.set_title('Top Speedups by Configuration')

    for bar, val in zip(bars, speedups):
        ax2.annotate(f'{val:.0f}x',
                    xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0),
                    textcoords='offset points',
                    ha='left', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    script_dir = Path(__file__).parent.parent
    csv_path = script_dir / 'assets' / 'benchmark_data.csv'
    assets_dir = script_dir / 'assets'

    if not csv_path.exists():
        print(f"Benchmark data not found at {csv_path}")
        print("Run 'mix run bench/generate_plots.exs' first to generate data.")
        return

    print("Loading benchmark data...")
    data = load_data(csv_path)

    print("Generating plots...")

    # Individual comparison plots
    plot_time_comparison(data['n_docs'], 'n_docs', 'Number of Documents',
                        assets_dir / 'benchmark_n_docs.png')
    plot_time_comparison(data['d_len'], 'd_len', 'Document Length (tokens)',
                        assets_dir / 'benchmark_d_len.png')
    plot_time_comparison(data['dim'], 'dim', 'Embedding Dimension',
                        assets_dir / 'benchmark_dim.png')

    # Combined speedup chart
    plot_speedup_chart(data, assets_dir / 'benchmark_speedup.png')

    # Summary plot
    plot_summary(data, assets_dir / 'benchmark_summary.png')

    print("\nAll plots generated successfully!")


if __name__ == '__main__':
    main()
