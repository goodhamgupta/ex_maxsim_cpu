#!/usr/bin/env python3
# /// script
# dependencies = [
#   "matplotlib>=3.8",
#   "numpy>=1.24",
# ]
# ///
"""
Generate benchmark comparison plots for ExMaxsimCpu vs Nx vs Nx+MPS (and Nx CPU backends if available).

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
NX_CPU_COLOR = '#0ea5e9'  # Sky - Nx CPU backend
SPEEDUP_COLOR = '#8b5cf6'  # Purple


def load_data(csv_path):
    """Load benchmark data from CSV."""
    data = {'n_docs': [], 'd_len': [], 'dim': []}
    has_mps = False
    has_nx_cpu = False
    nx_cpu_label = None

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            config = row['config']
            entry = {
                'param': int(row['param_value']),
                'ex_time': float(row['ex_time_ms']),
                'nx_time': float(row['nx_time_ms']),
                'nx_cpu_time': float(row['nx_cpu_time_ms']) if row.get('nx_cpu_time_ms') else None,
                'nx_cpu_backend': row.get('nx_cpu_backend') or None,
                'mps_time': float(row['mps_time_ms']) if row['mps_time_ms'] else None,
                'mps_transfer_time': float(row['mps_transfer_time_ms']) if row.get('mps_transfer_time_ms') else None,
                'speedup_nx': float(row['speedup_vs_nx']),
                'speedup_nx_cpu': float(row['speedup_vs_nx_cpu']) if row.get('speedup_vs_nx_cpu') else None,
                'speedup_mps': float(row['speedup_vs_mps']) if row['speedup_vs_mps'] else None,
            }
            data[config].append(entry)
            if entry['mps_time'] is not None:
                has_mps = True
            if entry['nx_cpu_time'] is not None:
                has_nx_cpu = True
            if entry['nx_cpu_backend'] and not nx_cpu_label:
                nx_cpu_label = entry['nx_cpu_backend']

    meta = {
        'has_mps': has_mps,
        'has_nx_cpu': has_nx_cpu,
        'nx_cpu_label': nx_cpu_label,
    }

    return data, meta


def plot_time_comparison(data, config_name, param_label, output_path, has_mps, has_nx_cpu, nx_cpu_label):
    """Create a bar chart comparing execution times."""
    fig, ax = plt.subplots(figsize=(12, 6))

    params = [d['param'] for d in data]
    ex_times = [d['ex_time'] for d in data]
    nx_times = [d['nx_time'] for d in data]
    mps_times = [d['mps_time'] if d['mps_time'] else 0 for d in data]
    nx_cpu_times = [d['nx_cpu_time'] if d['nx_cpu_time'] else 0 for d in data]

    x = np.arange(len(params))

    series = [
        ('ExMaxsimCpu (BLAS+SIMD)', ex_times, EX_COLOR),
    ]
    if has_nx_cpu and any(nx_cpu_times):
        series.append((nx_cpu_label or 'Nx CPU Backend', nx_cpu_times, NX_CPU_COLOR))
    if has_mps and any(mps_times):
        series.append(('Nx + Torchx MPS (GPU)', mps_times, MPS_COLOR))
    series.append(('Nx BinaryBackend', nx_times, NX_COLOR))

    width = 0.8 / len(series)
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, len(series))
    for offset, (label, times, color) in zip(offsets, series):
        ax.bar(x + offset, times, width, label=label, color=color, alpha=0.8)

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


def plot_speedup_chart(all_data, output_path, has_mps, has_nx_cpu):
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
        speedups_nx_cpu = [d['speedup_nx_cpu'] if d['speedup_nx_cpu'] else 0 for d in data]

        x = np.arange(len(params))

        series = [
            ('vs Nx Binary', speedups_nx, NX_COLOR),
        ]
        if has_nx_cpu and any(speedups_nx_cpu):
            series.append(('vs Nx CPU', speedups_nx_cpu, NX_CPU_COLOR))
        if has_mps and any(speedups_mps):
            series.append(('vs Nx MPS', speedups_mps, MPS_COLOR))

        width = 0.8 / len(series)
        offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, len(series))
        for offset, (series_label, values, color) in zip(offsets, series):
            bars = ax.bar(x + offset, values, width, label=series_label, color=color, alpha=0.8)
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.annotate(f'{val:.0f}x',
                               xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               xytext=(0, 3), textcoords='offset points',
                               ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xlabel(label)
        ax.set_ylabel('Speedup (x times faster)')
        ax.set_xticks(x)
        ax.set_xticklabels(params)
        if has_mps or has_nx_cpu:
            ax.legend(loc='upper left', fontsize=8)

    title = 'ExMaxsimCpu Speedup'
    title += ' vs Nx (BinaryBackend)'
    if has_nx_cpu:
        title += ' and Nx (CPU)'
    if has_mps:
        title += ' and Nx (MPS GPU)'

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary(all_data, output_path, has_mps, has_nx_cpu, nx_cpu_label):
    """Create a summary visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Time comparison for n_docs
    data = all_data['n_docs']
    params = [d['param'] for d in data]
    ex_times = [d['ex_time'] for d in data]
    nx_times = [d['nx_time'] for d in data]
    mps_times = [d['mps_time'] if d['mps_time'] else None for d in data]
    nx_cpu_times = [d['nx_cpu_time'] if d['nx_cpu_time'] else None for d in data]

    ax1.plot(params, ex_times, 'o-', color=EX_COLOR, linewidth=2, markersize=8, label='ExMaxsimCpu (BLAS+SIMD)')
    if has_mps and any(mps_times):
        mps_valid = [(p, t) for p, t in zip(params, mps_times) if t is not None]
        if mps_valid:
            ax1.plot([p for p, _ in mps_valid], [t for _, t in mps_valid], 'd-', color=MPS_COLOR, linewidth=2, markersize=8, label='Nx + Torchx MPS (GPU)')
    if has_nx_cpu and any(nx_cpu_times):
        cpu_valid = [(p, t) for p, t in zip(params, nx_cpu_times) if t is not None]
        if cpu_valid:
            ax1.plot([p for p, _ in cpu_valid], [t for _, t in cpu_valid], '^-', color=NX_CPU_COLOR, linewidth=2, markersize=8, label=nx_cpu_label or 'Nx CPU Backend')
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


def plot_three_way_comparison(all_data, output_path, has_mps, has_nx_cpu, nx_cpu_label):
    """Create a comprehensive comparison chart."""
    if not has_mps and not has_nx_cpu:
        print("Skipping multi-way comparison (no MPS/CPU backend data)")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Collect all data points
    all_configs = []
    ex_times = []
    nx_times = []
    mps_times = []
    nx_cpu_times = []

    for config in ['n_docs', 'd_len', 'dim']:
        for d in all_data[config]:
            if d['mps_time'] or d['nx_cpu_time']:
                all_configs.append(f"{config}={d['param']}")
                ex_times.append(d['ex_time'])
                nx_times.append(d['nx_time'])
                mps_times.append(d['mps_time'] if d['mps_time'] else 0)
                nx_cpu_times.append(d['nx_cpu_time'] if d['nx_cpu_time'] else 0)

    if not all_configs:
        print(f"Skipping three-way comparison (no complete data)")
        return

    x = np.arange(len(all_configs))
    series = [
        ('ExMaxsimCpu (BLAS+SIMD)', ex_times, EX_COLOR),
    ]
    if has_nx_cpu and any(nx_cpu_times):
        series.append((nx_cpu_label or 'Nx CPU Backend', nx_cpu_times, NX_CPU_COLOR))
    if has_mps and any(mps_times):
        series.append(('Nx + Torchx MPS (GPU)', mps_times, MPS_COLOR))
    series.append(('Nx BinaryBackend', nx_times, NX_COLOR))

    width = 0.8 / len(series)
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, len(series))
    for offset, (label, times, color) in zip(offsets, series):
        ax.bar(x + offset, times, width, label=label, color=color, alpha=0.8)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Time (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(all_configs, rotation=45, ha='right')
    ax.set_yscale('log')
    ax.legend(loc='upper left')
    ax.set_title('Performance Comparison: ExMaxsimCpu vs Nx (Binary/MPS/CPU)')

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
    data, meta = load_data(csv_path)
    has_mps = meta['has_mps']
    has_nx_cpu = meta['has_nx_cpu']
    nx_cpu_label = meta['nx_cpu_label']

    if has_mps:
        print("MPS data detected - generating comparison plots")
    else:
        print("No MPS data - generating comparison plots")

    print("\nGenerating plots...")

    # Individual comparison plots
    plot_time_comparison(
        data['n_docs'], 'n_docs', 'Number of Documents',
        assets_dir / 'benchmark_n_docs.png', has_mps, has_nx_cpu, nx_cpu_label
    )
    plot_time_comparison(
        data['d_len'], 'd_len', 'Document Length (tokens)',
        assets_dir / 'benchmark_d_len.png', has_mps, has_nx_cpu, nx_cpu_label
    )
    plot_time_comparison(
        data['dim'], 'dim', 'Embedding Dimension',
        assets_dir / 'benchmark_dim.png', has_mps, has_nx_cpu, nx_cpu_label
    )

    # Combined speedup chart
    plot_speedup_chart(data, assets_dir / 'benchmark_speedup.png', has_mps, has_nx_cpu)

    # Summary plot
    plot_summary(data, assets_dir / 'benchmark_summary.png', has_mps, has_nx_cpu, nx_cpu_label)

    # Multi-way comparison (if MPS/CPU data available)
    if has_mps or has_nx_cpu:
        plot_three_way_comparison(data, assets_dir / 'benchmark_three_way.png', has_mps, has_nx_cpu, nx_cpu_label)

    print("\nAll plots generated successfully!")


if __name__ == '__main__':
    main()
