#!/usr/bin/env python3
# /// script
# dependencies = [
#   "matplotlib>=3.8",
#   "numpy>=1.24",
# ]
# ///
"""
Generate benchmark comparison plots for ExMaxsimCpu vs Nx vs Nx+MPS.

Run after: mix run bench/generate_plots.exs
Usage: uv run bench/plot_benchmarks.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

# Colors
EX_COLOR = '#6366f1'  # Indigo - ExMaxsimCpu
NX_COLOR = '#f97316'  # Orange - Pure Nx
MPS_COLOR = '#10b981'  # Emerald - Nx + MPS
SPEEDUP_COLOR = '#8b5cf6'  # Purple


def load_data(csv_path):
    """Load benchmark data from CSV."""
    data = {'n_docs': [], 'd_len': [], 'dim': []}
    has_mps = False

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            config = row['config']
            entry = {
                'param': int(row['param_value']),
                'ex_time': float(row['ex_time_ms']),
                'nx_time': float(row['nx_time_ms']),
                'mps_time': float(row['mps_time_ms']) if row['mps_time_ms'] else None,
                'mps_transfer_time': float(row['mps_transfer_time_ms']) if row.get('mps_transfer_time_ms') else None,
                'speedup_nx': float(row['speedup_vs_nx']),
                'speedup_mps': float(row['speedup_vs_mps']) if row['speedup_vs_mps'] else None,
            }
            data[config].append(entry)
            if entry['mps_time'] is not None:
                has_mps = True

    return data, has_mps


def plot_time_comparison(data, config_name, param_label, output_path, has_mps):
    """Create a bar chart comparing execution times."""
    fig, ax = plt.subplots(figsize=(12, 6))

    params = [d['param'] for d in data]
    ex_times = [d['ex_time'] for d in data]
    nx_times = [d['nx_time'] for d in data]
    mps_times = [d['mps_time'] if d['mps_time'] else 0 for d in data]

    x = np.arange(len(params))

    if has_mps and any(mps_times):
        width = 0.25
        bars1 = ax.bar(x - width, ex_times, width, label='ExMaxsimCpu (BLAS+SIMD)', color=EX_COLOR, alpha=0.8)
        bars2 = ax.bar(x, mps_times, width, label='Nx + Torchx MPS (GPU)', color=MPS_COLOR, alpha=0.8)
        bars3 = ax.bar(x + width, nx_times, width, label='Nx BinaryBackend', color=NX_COLOR, alpha=0.8)
    else:
        width = 0.35
        bars1 = ax.bar(x - width/2, ex_times, width, label='ExMaxsimCpu (BLAS+SIMD)', color=EX_COLOR, alpha=0.8)
        bars3 = ax.bar(x + width/2, nx_times, width, label='Nx BinaryBackend', color=NX_COLOR, alpha=0.8)

    ax.set_xlabel(param_label)
    ax.set_ylabel('Time (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.set_yscale('log')
    ax.legend(loc='upper left')
    ax.set_title(f'Performance Comparison: Varying {param_label}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_speedup_chart(all_data, output_path, has_mps):
    """Create a combined speedup chart."""
    n_cols = 3
    fig, axes = plt.subplots(1, n_cols, figsize=(14, 5))

    configs = [
        ('n_docs', 'Number of Documents', axes[0]),
        ('d_len', 'Document Length (tokens)', axes[1]),
        ('dim', 'Embedding Dimension', axes[2]),
    ]

    for config_name, label, ax in configs:
        data = all_data[config_name]
        params = [str(d['param']) for d in data]
        speedups_nx = [d['speedup_nx'] for d in data]
        speedups_mps = [d['speedup_mps'] if d['speedup_mps'] else 0 for d in data]

        x = np.arange(len(params))

        if has_mps and any(speedups_mps):
            width = 0.35
            bars1 = ax.bar(x - width/2, speedups_nx, width, label='vs Nx Binary', color=NX_COLOR, alpha=0.8)
            bars2 = ax.bar(x + width/2, speedups_mps, width, label='vs Nx MPS', color=MPS_COLOR, alpha=0.8)

            # Add value labels
            for bar, val in zip(bars1, speedups_nx):
                ax.annotate(f'{val:.0f}x',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
            for bar, val in zip(bars2, speedups_mps):
                if val > 0:
                    ax.annotate(f'{val:.1f}x',
                               xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               xytext=(0, 3), textcoords='offset points',
                               ha='center', va='bottom', fontsize=8, fontweight='bold')
        else:
            bars = ax.bar(x, speedups_nx, color=NX_COLOR, alpha=0.8)
            for bar, val in zip(bars, speedups_nx):
                ax.annotate(f'{val:.0f}x',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xlabel(label)
        ax.set_ylabel('Speedup (x times faster)')
        ax.set_xticks(x)
        ax.set_xticklabels(params)
        if has_mps:
            ax.legend(loc='upper left', fontsize=8)

    title = 'ExMaxsimCpu Speedup'
    if has_mps:
        title += ' vs Nx (BinaryBackend) and Nx (MPS GPU)'
    else:
        title += ' vs Nx (BinaryBackend)'

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary(all_data, output_path, has_mps):
    """Create a summary visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Time comparison for n_docs
    data = all_data['n_docs']
    params = [d['param'] for d in data]
    ex_times = [d['ex_time'] for d in data]
    nx_times = [d['nx_time'] for d in data]
    mps_times = [d['mps_time'] if d['mps_time'] else None for d in data]

    ax1.plot(params, ex_times, 'o-', color=EX_COLOR, linewidth=2, markersize=8, label='ExMaxsimCpu (BLAS+SIMD)')
    if has_mps and any(mps_times):
        mps_valid = [(p, t) for p, t in zip(params, mps_times) if t is not None]
        if mps_valid:
            ax1.plot([p for p, _ in mps_valid], [t for _, t in mps_valid], 'd-', color=MPS_COLOR, linewidth=2, markersize=8, label='Nx + Torchx MPS (GPU)')
    ax1.plot(params, nx_times, 's-', color=NX_COLOR, linewidth=2, markersize=8, label='Nx BinaryBackend')

    ax1.set_xlabel('Number of Documents')
    ax1.set_ylabel('Time (ms)')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.set_title('Execution Time Comparison')
    ax1.grid(True, alpha=0.3)

    # Right: Speedup summary
    all_speedups_nx = []
    all_speedups_mps = []
    all_labels = []

    for config in ['n_docs', 'd_len', 'dim']:
        for d in all_data[config]:
            all_speedups_nx.append(d['speedup_nx'])
            all_speedups_mps.append(d['speedup_mps'] if d['speedup_mps'] else 0)
            all_labels.append(f"{config}={d['param']}")

    # Show top speedups vs Nx
    sorted_data = sorted(zip(all_speedups_nx, all_labels), reverse=True)[:8]
    speedups, labels = zip(*sorted_data) if sorted_data else ([], [])

    y_pos = np.arange(len(labels))
    bars = ax2.barh(y_pos, speedups, color=NX_COLOR, alpha=0.8, label='vs Nx BinaryBackend')

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel('Speedup (x times faster)')
    ax2.set_title('Top Speedups by Configuration')

    for bar, val in zip(bars, speedups):
        ax2.annotate(f'{val:.0f}x',
                    xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_three_way_comparison(all_data, output_path, has_mps):
    """Create a comprehensive three-way comparison chart."""
    if not has_mps:
        print(f"Skipping three-way comparison (no MPS data)")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Collect all data points
    all_configs = []
    ex_times = []
    nx_times = []
    mps_times = []

    for config in ['n_docs', 'd_len', 'dim']:
        for d in all_data[config]:
            if d['mps_time']:
                all_configs.append(f"{config}={d['param']}")
                ex_times.append(d['ex_time'])
                nx_times.append(d['nx_time'])
                mps_times.append(d['mps_time'])

    if not all_configs:
        print(f"Skipping three-way comparison (no complete data)")
        return

    x = np.arange(len(all_configs))
    width = 0.25

    bars1 = ax.bar(x - width, ex_times, width, label='ExMaxsimCpu (BLAS+SIMD)', color=EX_COLOR, alpha=0.8)
    bars2 = ax.bar(x, mps_times, width, label='Nx + Torchx MPS (GPU)', color=MPS_COLOR, alpha=0.8)
    bars3 = ax.bar(x + width, nx_times, width, label='Nx BinaryBackend', color=NX_COLOR, alpha=0.8)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Time (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(all_configs, rotation=45, ha='right')
    ax.set_yscale('log')
    ax.legend(loc='upper left')
    ax.set_title('Three-Way Performance Comparison: ExMaxsimCpu vs Nx (MPS) vs Nx (Binary)')

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
    data, has_mps = load_data(csv_path)

    if has_mps:
        print("MPS data detected - generating three-way comparison plots")
    else:
        print("No MPS data - generating two-way comparison plots")

    print("\nGenerating plots...")

    # Individual comparison plots
    plot_time_comparison(data['n_docs'], 'n_docs', 'Number of Documents',
                        assets_dir / 'benchmark_n_docs.png', has_mps)
    plot_time_comparison(data['d_len'], 'd_len', 'Document Length (tokens)',
                        assets_dir / 'benchmark_d_len.png', has_mps)
    plot_time_comparison(data['dim'], 'dim', 'Embedding Dimension',
                        assets_dir / 'benchmark_dim.png', has_mps)

    # Combined speedup chart
    plot_speedup_chart(data, assets_dir / 'benchmark_speedup.png', has_mps)

    # Summary plot
    plot_summary(data, assets_dir / 'benchmark_summary.png', has_mps)

    # Three-way comparison (if MPS data available)
    if has_mps:
        plot_three_way_comparison(data, assets_dir / 'benchmark_three_way.png', has_mps)

    print("\nAll plots generated successfully!")


if __name__ == '__main__':
    main()
