#!/usr/bin/env python3
"""Generate plots from DSE GPU metrics logs."""

from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd


TOPOLOGY_NAMES = {
    "convolutional": "Convolutional AlexNet",
    "fft": "FFT AlexNet",
    "torch": "PyTorch AlexNet",
}

# Test dataset size (CIFAR-100 test set has 10,000 images)
TEST_DATASET_SIZE = 10000

# Training dataset size (CIFAR-100 train set has 50,000 images)
TRAIN_DATASET_SIZE = 50000

# Consistent color scheme
ARCH_COLORS = {
    'convolutional': 'tab:blue',
    'fft': 'tab:orange',
    'torch': 'tab:green'
}

WORKER_LINESTYLES = {
    1: '-',
    2: '--',
    4: '-.',
    8: ':',
    9: (0, (3, 1, 1, 1)),
    10: (0, (5, 1))
}

WORKER_MARKERS = {
    1: 'o',
    2: 's',
    4: '^',
    8: 'D',
    9: 'v',
    10: 'p'
}

KERNEL_LINESTYLES = {
    3: '-',
    5: '--',
    7: '-.',
    9: ':',
    11: (0, (3, 1, 1, 1)),
    13: (0, (5, 1)),
    15: (0, (1, 1))
}

KERNEL_MARKERS = {
    3: 'o',
    5: 's',
    7: '^',
    9: 'D',
    11: 'v',
    13: 'p',
    15: '*'
}


@dataclass
class RunRecord:
    run_type: str  # "train" or "inference"
    arch_key: str
    kernel_size: int
    num_workers: int
    batch_size: int
    epochs: int | None
    iterations: int | None
    df: pd.DataFrame
    source: Path
    energy_j: float
    energy_stddev_j: float = 0.0
    time_stddev_s: float = 0.0
    monitor_mode: str = "gpu"

    @property
    def friendly_label(self) -> str:
        arch_name = TOPOLOGY_NAMES.get(self.arch_key, self.arch_key)
        return f"{arch_name} (k={self.kernel_size})"


RunData = List[RunRecord]


def _load_run_data(raw_dir: Path) -> RunData:
    """Load all run records from CSV files."""
    files = sorted(raw_dir.glob("*_metrics*.csv"))
    data: RunData = []
    
    for path in files:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        
        # Detect monitoring mode based on columns
        monitor_mode = "cpu" if "energy_J" in df.columns else "gpu"
        
        if monitor_mode == "cpu":
            # CPU mode: parse perf data
            for column in ("energy_J", "energy_stddev_J", "time_s", "time_stddev_s"):
                if column in df.columns:
                    df[column] = pd.to_numeric(df[column], errors='coerce')
            
            if df.empty or "energy_J" not in df.columns:
                continue
                
            energy_j = float(df["energy_J"].iloc[0])
            energy_stddev_j = float(df["energy_stddev_J"].iloc[0]) if "energy_stddev_J" in df.columns else 0.0
            time_stddev_s = float(df["time_stddev_s"].iloc[0]) if "time_stddev_s" in df.columns else 0.0
        else:
            # GPU mode: parse nvidia-smi data
            for column in ("rel_ms", "power_W", "util_pct", "mem_used_MiB", "mem_total_MiB", "temp_C"):
                if column in df.columns:
                    df[column] = pd.to_numeric(df[column], errors='coerce')
            df.dropna(subset=["rel_ms", "power_W", "mem_used_MiB", "mem_total_MiB"], inplace=True)
            df["time_s"] = df["rel_ms"] / 1000.0
            df["mem_util_pct"] = (df["mem_used_MiB"] / df["mem_total_MiB"].replace(0, pd.NA)) * 100.0
            
            df["dt"] = df["time_s"].diff().fillna(0.0).clip(lower=0.0)
            energy_j = float((df["power_W"] * df["dt"]).sum())
            energy_stddev_j = 0.0
            time_stddev_s = 0.0

        # Parse filename - new format: <phase>_<arch>_k<kernel>_e<epochs>_bs<batch>_w<workers>_metrics.csv
        stem = path.stem
        base = stem[:-len("_metrics")] if stem.endswith("_metrics") else stem
        tokens = base.split("_")
        if len(tokens) < 6:
            continue

        run_type = tokens[0]
        if run_type == "eval":
            run_type = "inference"
        
        # Extract architecture (second token)
        arch_key = tokens[1]
        
        # Extract kernel size (k<size>)
        kernel_token = tokens[2]
        if not kernel_token.startswith("k"):
            continue
        try:
            kernel_size = int(kernel_token[1:])
        except ValueError:
            continue
        
        # Extract epochs or iterations
        duration_token = tokens[3]
        epochs: int | None = None
        iterations: int | None = None
        if duration_token.startswith("e"):
            try:
                epochs = int(duration_token[1:])
            except ValueError:
                continue
        elif duration_token.startswith("i"):
            try:
                iterations = int(duration_token[1:])
            except ValueError:
                continue
        else:
            continue

        # Extract batch size
        batch_token = tokens[4]
        if not batch_token.startswith("bs"):
            continue
        try:
            batch_size = int(batch_token[2:])
        except ValueError:
            continue

        # Extract workers
        workers_token = tokens[5]
        if not workers_token.startswith("w"):
            continue
        try:
            workers = int(workers_token[1:])
        except ValueError:
            continue

        data.append(
            RunRecord(
                run_type=run_type,
                arch_key=arch_key,
                kernel_size=kernel_size,
                num_workers=workers,
                batch_size=batch_size,
                epochs=epochs,
                iterations=iterations,
                df=df,
                source=path,
                energy_j=energy_j,
                energy_stddev_j=energy_stddev_j,
                time_stddev_s=time_stddev_s,
                monitor_mode=monitor_mode,
            )
        )
    return data


def _plot_comparative_metrics(data: RunData, output_dir: Path, run_type: str) -> None:
    """
    Generate comparative plots for all runs of a given type (train or inference).
    Compares different architectures and worker values across metrics:
    - Total energy consumption
    - Total execution time
    - Throughput (inferences/s or images/s)
    - Energy efficiency (inferences/J or images/J)
    - Average GPU utilization (GPU mode only)
    - Memory usage (GPU mode only)
    """
    filtered_data = [record for record in data if record.run_type == run_type]
    
    if not filtered_data:
        return
    
    # Detect if we have CPU or GPU data
    has_cpu_data = any(rec.monitor_mode == "cpu" for rec in filtered_data)
    
    # Group by epochs/iterations for fair comparison
    groups = {}
    for record in filtered_data:
        metric_value = record.epochs if run_type == "train" else record.iterations
        key = metric_value
        if key not in groups:
            groups[key] = []
        groups[key].append(record)
    
    for metric_value, group_records in groups.items():
        # Group by architecture
        arch_data = {}
        for record in group_records:
            if record.arch_key not in arch_data:
                arch_data[record.arch_key] = []
            arch_data[record.arch_key].append(record)
        
        # Skip if only one kernel size
        has_sweep = any(len(records) > 1 for records in arch_data.values())
        if not has_sweep:
            continue
        
        # Prepare data for plotting
        plot_data = {}
        
        for arch_key, records in arch_data.items():
            if len(records) <= 1:
                continue
            
            records = sorted(records, key=lambda r: r.kernel_size)
            kernel_sizes = [r.kernel_size for r in records]
            energies = [r.energy_j for r in records]
            energy_stddevs = [r.energy_stddev_j for r in records]
            
            # Calculate metrics
            runtimes = []
            runtime_stddevs = []
            avg_utils = []
            avg_mem_utils = []
            
            for record in records:
                if record.monitor_mode == "cpu":
                    runtime = record.df["time_s"].iloc[0] if "time_s" in record.df.columns else 0.0
                    runtime_std = record.time_stddev_s
                    runtimes.append(runtime)
                    runtime_stddevs.append(runtime_std)
                    avg_utils.append(0.0)
                    avg_mem_utils.append(0.0)
                else:
                    runtime = record.df["time_s"].max() if not record.df.empty else 0.0
                    runtime_std = 0.0
                    runtimes.append(runtime)
                    runtime_stddevs.append(runtime_std)
                    avg_utils.append(record.df["util_pct"].mean() if "util_pct" in record.df.columns else 0.0)
                    avg_mem_utils.append(record.df["mem_util_pct"].mean() if "mem_util_pct" in record.df.columns else 0.0)
            
            # Calculate throughput and efficiency
            if run_type == "inference":
                total_items = metric_value * TEST_DATASET_SIZE
            else:
                total_items = metric_value * TRAIN_DATASET_SIZE
            
            throughputs = [total_items / rt if rt > 0 else 0 for rt in runtimes]
            energy_efficiency = [total_items / e if e > 0 else 0 for e in energies]
            
            # Error propagation
            throughput_stddevs = [
                (total_items / (rt ** 2)) * rt_std if rt > 0 else 0 
                for rt, rt_std in zip(runtimes, runtime_stddevs)
            ]
            efficiency_stddevs = [
                (total_items / (e ** 2)) * e_std if e > 0 else 0 
                for e, e_std in zip(energies, energy_stddevs)
            ]
            
            plot_data[arch_key] = {
                'kernel_sizes': kernel_sizes,
                'energies': energies,
                'energy_stddevs': energy_stddevs,
                'runtimes': runtimes,
                'runtime_stddevs': runtime_stddevs,
                'throughputs': throughputs,
                'throughput_stddevs': throughput_stddevs,
                'energy_efficiency': energy_efficiency,
                'efficiency_stddevs': efficiency_stddevs,
                'avg_utils': avg_utils,
                'avg_mem_utils': avg_mem_utils,
            }
        
        if not plot_data:
            continue
        
        # Determine number of subplots
        if has_cpu_data:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        else:
            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 18))
        
        # Plot data for each architecture
        for arch_key, metrics in plot_data.items():
            arch_name = TOPOLOGY_NAMES.get(arch_key, arch_key)
            color = ARCH_COLORS.get(arch_key, 'gray')
            linestyle = '-'
            marker = 'o'
            label = arch_name
            
            kernel_sizes = metrics['kernel_sizes']
            
            # For inference: no error bars; for training: with error bars
            if run_type == "inference":
                # Plot 1: Energy consumption
                ax1.plot(kernel_sizes, metrics['energies'],
                        marker=marker, linewidth=2, markersize=6, label=label, 
                        color=color, linestyle=linestyle, alpha=0.8)
                
                # Plot 2: Execution time (swapped position with energy efficiency)
                ax2.plot(kernel_sizes, metrics['runtimes'],
                        marker=marker, linewidth=2, markersize=6, label=label, 
                        color=color, linestyle=linestyle, alpha=0.8)
                
                # Plot 3: Throughput
                ax3.plot(kernel_sizes, metrics['throughputs'],
                        marker=marker, linewidth=2, markersize=6, label=label, 
                        color=color, linestyle=linestyle, alpha=0.8)
                
                # Plot 4: Energy efficiency (swapped position with execution time)
                ax4.plot(kernel_sizes, metrics['energy_efficiency'],
                        marker=marker, linewidth=2, markersize=6, label=label, 
                        color=color, linestyle=linestyle, alpha=0.8)
            else:
                # Training: keep error bars and original positions
                # Plot 1: Energy consumption
                ax1.errorbar(kernel_sizes, metrics['energies'], yerr=metrics['energy_stddevs'],
                            marker=marker, linewidth=2, markersize=6, label=label, 
                            color=color, linestyle=linestyle, alpha=0.8, capsize=3)
                
                # Plot 2: Energy efficiency
                ax2.errorbar(kernel_sizes, metrics['energy_efficiency'], yerr=metrics['efficiency_stddevs'],
                            marker=marker, linewidth=2, markersize=6, label=label, 
                            color=color, linestyle=linestyle, alpha=0.8, capsize=3)
                
                # Plot 3: Throughput
                ax3.errorbar(kernel_sizes, metrics['throughputs'], yerr=metrics['throughput_stddevs'],
                            marker=marker, linewidth=2, markersize=6, label=label, 
                            color=color, linestyle=linestyle, alpha=0.8, capsize=3)
                
                # Plot 4: Execution time
                ax4.errorbar(kernel_sizes, metrics['runtimes'], yerr=metrics['runtime_stddevs'],
                            marker=marker, linewidth=2, markersize=6, label=label, 
                            color=color, linestyle=linestyle, alpha=0.8, capsize=3)
            
            # GPU-specific plots
            if not has_cpu_data:
                # Plot 5: GPU Utilization
                ax5.plot(kernel_sizes, metrics['avg_utils'], marker=marker, linewidth=2, 
                        markersize=6, label=label, color=color, linestyle=linestyle, alpha=0.8)
                
                # Plot 6: Memory Utilization
                ax6.plot(kernel_sizes, metrics['avg_mem_utils'], marker=marker, linewidth=2, 
                        markersize=6, label=label, color=color, linestyle=linestyle, alpha=0.8)
        
        # Find best energy efficiency for annotation
        best_eff = {'value': 0, 'arch': None, 'kernel_size': None}
        for arch_key, metrics in plot_data.items():
            for i, k in enumerate(metrics['kernel_sizes']):
                if metrics['energy_efficiency'][i] > best_eff['value']:
                    best_eff['value'] = metrics['energy_efficiency'][i]
                    best_eff['arch'] = TOPOLOGY_NAMES.get(arch_key, arch_key)
                    best_eff['kernel_size'] = k
        
        # Get all unique kernel sizes
        all_kernel_sizes = sorted(set(k for metrics in plot_data.values() for k in metrics['kernel_sizes']))
        
        # Configure axes based on run_type (inference has swapped axes 2 and 4)
        # Axis 1: Energy Consumption
        ax1.set_xlabel("First Kernel Size", fontsize=11)
        ax1.set_ylabel("Total Energy (J)", fontsize=11)
        ax1.set_xticks(all_kernel_sizes)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, loc='best', ncol=1, title='Lower is better', title_fontsize=9)
        ax1.set_title("Energy Consumption", fontsize=12, fontweight='bold')
        
        if run_type == "inference":
            # Axis 2: Execution Time (swapped for inference)
            ax2.set_xlabel("First Kernel Size", fontsize=11)
            ax2.set_ylabel("Execution Time (s)", fontsize=11)
            ax2.set_xticks(all_kernel_sizes)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=8, loc='best', ncol=1, title='Lower is better', title_fontsize=9)
            ax2.set_title("Execution Time", fontsize=12, fontweight='bold')
            
            # Axis 3: Throughput
            ax3.set_xlabel("First Kernel Size", fontsize=11)
            ax3.set_ylabel("Throughput (images/s)", fontsize=11)
            ax3.set_xticks(all_kernel_sizes)
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=8, loc='best', ncol=1, title='Higher is better', title_fontsize=9)
            ax3.set_title("Throughput", fontsize=12, fontweight='bold')
            
            # Axis 4: Energy Efficiency (swapped for inference, with annotation)
            ax4.set_xlabel("First Kernel Size", fontsize=11)
            ax4.set_ylabel("Energy Efficiency (images/J)", fontsize=11)
            ax4.set_xticks(all_kernel_sizes)
            ax4.grid(True, alpha=0.3)
            ax4.legend(fontsize=8, loc='best', ncol=1, title='Higher is better', title_fontsize=9)
            ax4.set_title("Energy Efficiency", fontsize=12, fontweight='bold')
            
            # Add annotation for best efficiency on ax4
            if best_eff['arch']:
                ax4.annotate(f"Best: {best_eff['arch']}\nk={best_eff['kernel_size']}\n{best_eff['value']:.1f} images/J",
                            xy=(best_eff['kernel_size'], best_eff['value']), xytext=(10, -20),
                            textcoords='offset points', fontsize=8, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='green', lw=1.5))
        else:
            # Training: original order
            # Axis 2: Energy Efficiency (with annotation)
            ax2.set_xlabel("First Kernel Size", fontsize=11)
            ax2.set_ylabel("Energy Efficiency (images/J)", fontsize=11)
            ax2.set_xticks(all_kernel_sizes)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=8, loc='best', ncol=1, title='Higher is better', title_fontsize=9)
            ax2.set_title("Energy Efficiency", fontsize=12, fontweight='bold')
            
            # Add annotation for best efficiency on ax2
            if best_eff['arch']:
                ax2.annotate(f"Best: {best_eff['arch']}\nk={best_eff['kernel_size']}\n{best_eff['value']:.1f} images/J",
                            xy=(best_eff['kernel_size'], best_eff['value']), xytext=(10, -20),
                            textcoords='offset points', fontsize=8, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='green', lw=1.5))
            
            # Axis 3: Throughput
            ax3.set_xlabel("First Kernel Size", fontsize=11)
            ax3.set_ylabel("Throughput (images/s)", fontsize=11)
            ax3.set_xticks(all_kernel_sizes)
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=8, loc='best', ncol=1, title='Higher is better', title_fontsize=9)
            ax3.set_title("Throughput", fontsize=12, fontweight='bold')
            
            # Axis 4: Execution Time
            ax4.set_xlabel("First Kernel Size", fontsize=11)
            ax4.set_ylabel("Execution Time (s)", fontsize=11)
            ax4.set_xticks(all_kernel_sizes)
            ax4.grid(True, alpha=0.3)
            ax4.legend(fontsize=8, loc='best', ncol=1, title='Lower is better', title_fontsize=9)
            ax4.set_title("Execution Time", fontsize=12, fontweight='bold')
        
        # GPU-specific axes
        if not has_cpu_data:
            # Axis 5: GPU Utilization
            ax5.set_xlabel("First Kernel Size", fontsize=11)
            ax5.set_ylabel("GPU Utilization (%)", fontsize=11)
            ax5.set_xticks(all_kernel_sizes)
            ax5.set_ylim(0, 100)
            ax5.grid(True, alpha=0.3)
            ax5.legend(fontsize=8, loc='best', ncol=1, title='Higher is better', title_fontsize=9)
            ax5.set_title("GPU Utilization", fontsize=12, fontweight='bold')
            
            # Axis 6: Memory Utilization
            ax6.set_xlabel("First Kernel Size", fontsize=11)
            ax6.set_ylabel("Memory Utilization (%)", fontsize=11)
            ax6.set_xticks(all_kernel_sizes)
            ax6.set_ylim(0, 100)
            ax6.grid(True, alpha=0.3)
            ax6.legend(fontsize=8, loc='best', ncol=1, title='Lower is better', title_fontsize=9)
            ax6.set_title("Memory Utilization", fontsize=12, fontweight='bold')
        
        # Generate title and filename
        if run_type == "train":
            title_suffix = f"epochs={metric_value}"
            output_name = f"train_comparison_e{metric_value}.png"
        else:
            title_suffix = f"iter={metric_value}"
            output_name = f"inference_comparison_i{metric_value}.png"
        
        fig.suptitle(f"{run_type.capitalize()} Comparison ({title_suffix})", 
                    fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        output_path = output_dir / output_name
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Comparative plot saved: {output_path}")


def _plot_energy_ranking(data: RunData, output_dir: Path) -> None:
    """Generate bar charts ranking configurations by energy consumption."""
    
    # Plot for inference
    inference_data = [record for record in data if record.run_type == "inference"]
    if inference_data:
        iter_groups = {}
        for record in inference_data:
            key = record.iterations
            if key not in iter_groups:
                iter_groups[key] = []
            iter_groups[key].append(record)
        _plot_energy_ranking_helper(iter_groups, output_dir, "inference")
    
    # Plot for training
    train_data = [record for record in data if record.run_type == "train"]
    if train_data:
        epoch_groups = {}
        for record in train_data:
            key = record.epochs
            if key not in epoch_groups:
                epoch_groups[key] = []
            epoch_groups[key].append(record)
        _plot_energy_ranking_helper(epoch_groups, output_dir, "train")


def _plot_energy_ranking_helper(groups: Dict, output_dir: Path, phase: str) -> None:
    """Helper function to generate energy ranking bar charts."""
    for metric_value, group_records in groups.items():
        if len(group_records) <= 1:
            continue
        
        # Prepare ranking data
        ranking_data = []
        for record in group_records:
            arch_name = TOPOLOGY_NAMES.get(record.arch_key, record.arch_key)
            config = f"{arch_name} (k={record.kernel_size})"
            ranking_data.append({
                'config': config,
                'energy': record.energy_j,
                'arch_key': record.arch_key,
                'arch_name': arch_name,
                'kernel_size': record.kernel_size
            })
        
        # Sort by energy (ascending - most efficient first)
        ranking_data.sort(key=lambda x: x['energy'])
        
        # Prepare plot data
        configs = [d['config'] for d in ranking_data]
        energies = [d['energy'] for d in ranking_data]
        colors = [ARCH_COLORS.get(d['arch_key'], 'gray') for d in ranking_data]
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(12, max(8, len(configs) * 0.4)))
        
        bars = ax.barh(range(len(configs)), energies, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add value labels
        for i, (bar, energy) in enumerate(zip(bars, energies)):
            ax.text(energy + max(energies) * 0.01, i, f'{energy:.1f} J', 
                   va='center', fontsize=9, fontweight='bold')
        
        # Highlight the most efficient
        if len(bars) > 0:
            bars[0].set_edgecolor('green')
            bars[0].set_linewidth(3)
            bars[0].set_alpha(0.9)
        
        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels(configs, fontsize=10)
        ax.set_xlabel('Total Energy Consumption (J)', fontsize=12, fontweight='bold')
        
        if phase == "train":
            title = f"Energy Ranking - Training (epochs={metric_value})"
        else:
            title = f"Energy Ranking - Inference (iter={metric_value})"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add legend for architectures
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=ARCH_COLORS['convolutional'], edgecolor='black', label='Convolutional AlexNet', alpha=0.7),
            Patch(facecolor=ARCH_COLORS['fft'], edgecolor='black', label='FFT AlexNet', alpha=0.7),
            Patch(facecolor=ARCH_COLORS['torch'], edgecolor='black', label='PyTorch AlexNet', alpha=0.7),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        # Add best configuration annotation
        best = ranking_data[0]
        textstr = f"Most Efficient:\n{best['arch_name']}\nKernel Size: {best['kernel_size']}\nEnergy: {best['energy']:.1f} J"
        props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, edgecolor='green', linewidth=2)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props, fontweight='bold')
        
        fig.tight_layout()
        
        if phase == "train":
            output_name = f"train_energy_ranking_e{metric_value}.png"
        else:
            output_name = f"inference_energy_ranking_i{metric_value}.png"
        
        output_path = output_dir / output_name
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Energy ranking saved: {output_path}")


def _plot_unified_inference_energy_ranking_DISABLED(data: RunData, output_dir: Path) -> None:
    """Generate unified energy ranking for inference across all workers, excluding mFFT."""
    inference_data = [record for record in data if record.run_type == "inference"]
    
    # Filter out mFFT AlexNet
    inference_data = [record for record in inference_data if record.arch_key != "mfft_alexnet"]
    
    if not inference_data:
        return
    
    # Group by iterations to ensure fair comparison
    iterations_groups = {}
    for record in inference_data:
        if record.iterations not in iterations_groups:
            iterations_groups[record.iterations] = []
        iterations_groups[record.iterations].append(record)
    
    for iterations, records in iterations_groups.items():
        if len(records) <= 1:
            continue
        
        # Prepare ranking data with architecture and workers info
        ranking_data = []
        for record in records:
            arch_name = TOPOLOGY_NAMES.get(record.arch_key, record.arch_key)
            config = f"{arch_name} (bs={record.batch_size}, w={record.num_workers})"
            ranking_data.append({
                'config': config,
                'energy': record.energy_j,
                'arch_key': record.arch_key,
                'arch_name': arch_name,
                'batch_size': record.batch_size,
                'workers': record.num_workers
            })
        
        # Sort by energy (ascending - most efficient first)
        ranking_data.sort(key=lambda x: x['energy'])
        
        # Prepare plot data
        configs = [d['config'] for d in ranking_data]
        energies = [d['energy'] for d in ranking_data]
        colors = [ARCH_COLORS.get(d['arch_key'], 'gray') for d in ranking_data]
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(14, max(10, len(configs) * 0.35)))
        
        bars = ax.barh(range(len(configs)), energies, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add value labels
        for i, (bar, energy) in enumerate(zip(bars, energies)):
            ax.text(energy + max(energies) * 0.01, i, f'{energy:.1f} J', 
                   va='center', fontsize=9, fontweight='bold')
        
        # Highlight the most efficient
        if len(bars) > 0:
            bars[0].set_edgecolor('green')
            bars[0].set_linewidth(3)
            bars[0].set_alpha(0.9)
        
        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels(configs, fontsize=9)
        ax.set_xlabel('Total Energy Consumption (J)', fontsize=12, fontweight='bold')
        
        title = f"Unified Energy Ranking - Inference (iter={iterations}, all workers)"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add legend for architectures (only AlexNet and sFFT)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=ARCH_COLORS['alexnet'], edgecolor='black', label='AlexNet', alpha=0.7),
            Patch(facecolor=ARCH_COLORS['sfft_alexnet'], edgecolor='black', label='sFFT AlexNet', alpha=0.7),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
        
        # Add best configuration annotation
        best = ranking_data[0]
        textstr = f"Most Efficient:\n{best['arch_name']}\nBatch Size: {best['batch_size']}\nWorkers: {best['workers']}\nEnergy: {best['energy']:.1f} J"
        props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, edgecolor='green', linewidth=2)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props, fontweight='bold')
        
        fig.tight_layout()
        
        output_name = f"inference_unified_energy_ranking_i{iterations}.png"
        output_path = output_dir / output_name
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Unified energy ranking saved: {output_path}")


def _plot_topology_comparison_by_kernel(data: RunData, output_dir: Path) -> None:
    """Generate comparison plots for the 3 topologies for each kernel size (training only)."""
    train_data = [record for record in data if record.run_type == "train"]
    
    if not train_data:
        print("No training data found for topology comparison.")
        return
    
    # Group by kernel size
    kernel_groups = {}
    for record in train_data:
        if record.kernel_size not in kernel_groups:
            kernel_groups[record.kernel_size] = []
        kernel_groups[record.kernel_size].append(record)
    
    print(f"Found {len(kernel_groups)} kernel sizes for topology comparison: {sorted(kernel_groups.keys())}")
    
    for kernel_size, records in sorted(kernel_groups.items()):
        # Group by architecture
        arch_data = {}
        for record in records:
            if record.arch_key not in arch_data:
                arch_data[record.arch_key] = []
            arch_data[record.arch_key].append(record)
        
        print(f"  Kernel size {kernel_size}: found architectures {list(arch_data.keys())}")
        
        # Need at least 2 architectures for comparison
        if len(arch_data) < 2:
            print(f"    Skipping - need at least 2 architectures, found {len(arch_data)}")
            continue
        
        # Create figure with 3 metrics: Energy, Time, Energy Efficiency
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        arch_names = []
        energies = []
        times = []
        efficiencies = []
        colors = []
        
        for arch_key in ['convolutional', 'fft', 'torch']:
            if arch_key not in arch_data or len(arch_data[arch_key]) == 0:
                continue
            
            # Use the first (or only) record for this architecture
            record = arch_data[arch_key][0]
            arch_names.append(TOPOLOGY_NAMES.get(arch_key, arch_key))
            energies.append(record.energy_j)
            
            # Calculate time
            if record.monitor_mode == "cpu":
                time_val = record.df["time_s"].iloc[0] if "time_s" in record.df.columns else 0.0
            else:
                time_val = record.df["time_s"].max() if not record.df.empty else 0.0
            times.append(time_val)
            
            # Calculate energy efficiency
            total_items = record.epochs * TRAIN_DATASET_SIZE
            efficiency = total_items / record.energy_j if record.energy_j > 0 else 0
            efficiencies.append(efficiency)
            
            colors.append(ARCH_COLORS.get(arch_key, 'gray'))
        
        if len(arch_names) < 2:
            print(f"    Skipping plot - only {len(arch_names)} architectures with valid data")
            continue
        
        print(f"  Generating topology comparison plot for kernel size {kernel_size}")
        
        # Plot 1: Energy Consumption
        bars1 = ax1.bar(range(len(arch_names)), energies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.set_xticks(range(len(arch_names)))
        ax1.set_xticklabels(arch_names, fontsize=11)
        ax1.set_ylabel('Total Energy (J)', fontsize=12, fontweight='bold')
        ax1.set_title('Energy Consumption', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Highlight the best (lowest energy)
        min_idx = energies.index(min(energies))
        bars1[min_idx].set_edgecolor('green')
        bars1[min_idx].set_linewidth(3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1, energies)):
            ax1.text(bar.get_x() + bar.get_width()/2, val + max(energies) * 0.02, 
                    f'{val:.1f} J', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 2: Execution Time
        bars2 = ax2.bar(range(len(arch_names)), times, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_xticks(range(len(arch_names)))
        ax2.set_xticklabels(arch_names, fontsize=11)
        ax2.set_ylabel('Execution Time (s)', fontsize=12, fontweight='bold')
        ax2.set_title('Execution Time', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Highlight the best (lowest time)
        min_idx = times.index(min(times))
        bars2[min_idx].set_edgecolor('green')
        bars2[min_idx].set_linewidth(3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars2, times)):
            ax2.text(bar.get_x() + bar.get_width()/2, val + max(times) * 0.02, 
                    f'{val:.1f} s', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 3: Energy Efficiency
        bars3 = ax3.bar(range(len(arch_names)), efficiencies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax3.set_xticks(range(len(arch_names)))
        ax3.set_xticklabels(arch_names, fontsize=11)
        ax3.set_ylabel('Energy Efficiency (images/J)', fontsize=12, fontweight='bold')
        ax3.set_title('Energy Efficiency', fontsize=13, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # Highlight the best (highest efficiency)
        max_idx = efficiencies.index(max(efficiencies))
        bars3[max_idx].set_edgecolor('green')
        bars3[max_idx].set_linewidth(3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars3, efficiencies)):
            ax3.text(bar.get_x() + bar.get_width()/2, val + max(efficiencies) * 0.02, 
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        fig.suptitle(f'Topology Comparison for Kernel Size {kernel_size}', 
                    fontsize=15, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        output_name = f"train_topology_comparison_k{kernel_size}.png"
        output_path = output_dir / output_name
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Topology comparison saved: {output_path}")


def _plot_trainable_parameters(raw_dir: Path, output_dir: Path) -> None:
    """Generate bar chart comparing trainable parameters across topologies and kernel sizes."""
    log_files = sorted(raw_dir.glob("train_*.log"))
    
    # Extract trainable parameters from log files
    param_data = {}  # {(arch_key, kernel_size): trainable_params}
    
    for log_path in log_files:
        # Extract architecture and kernel size from filename
        # Format: train_<arch>_k<kernel>_e<epochs>_bs<batch>_w<workers>.log
        stem = log_path.stem
        tokens = stem.split("_")
        if len(tokens) < 6:
            continue
        
        arch_key = tokens[1]
        kernel_token = tokens[2]
        if not kernel_token.startswith("k"):
            continue
        try:
            kernel_size = int(kernel_token[1:])
        except ValueError:
            continue
        
        # Parse log file for trainable parameters
        with open(log_path, 'r') as f:
            for line in f:
                if "Trainable Parameters:" in line:
                    try:
                        # Extract number, removing commas
                        param_str = line.split("Trainable Parameters:")[1].strip()
                        trainable_params = int(param_str.replace(",", ""))
                        param_data[(arch_key, kernel_size)] = trainable_params
                        break
                    except (IndexError, ValueError):
                        continue
    
    if not param_data:
        print("No trainable parameter data found in logs.")
        return
    
    # Group by kernel size
    kernel_groups = {}
    for (arch_key, kernel_size), params in param_data.items():
        if kernel_size not in kernel_groups:
            kernel_groups[kernel_size] = {}
        kernel_groups[kernel_size][arch_key] = params
    
    # Create grouped bar chart
    kernel_sizes = sorted(kernel_groups.keys())
    arch_keys = ['convolutional', 'fft', 'torch']
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = range(len(kernel_sizes))
    width = 0.25
    
    for i, arch_key in enumerate(arch_keys):
        params_list = [kernel_groups[k].get(arch_key, 0) for k in kernel_sizes]
        offset = (i - 1) * width
        bars = ax.bar([pos + offset for pos in x], params_list, width,
                      label=TOPOLOGY_NAMES.get(arch_key, arch_key),
                      color=ARCH_COLORS.get(arch_key, 'gray'),
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar, val in zip(bars, params_list):
            if val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val/1e6:.2f}M',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('First Kernel Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Trainable Parameters (millions)', fontsize=12, fontweight='bold')
    ax.set_title('Trainable Parameters Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(kernel_sizes)
    ax.legend(fontsize=11, loc='best')
    ax.grid(axis='y', alpha=0.3)
    
    # Format y-axis to show millions
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y/1e6:.1f}M'))
    
    fig.tight_layout()
    
    output_path = output_dir / "trainable_parameters_comparison.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    
    print(f"Trainable parameters plot saved: {output_path}")


def _plot_training_progress(raw_dir: Path, output_dir: Path) -> None:
    """Generate training progress plots (Top-5 accuracy and loss vs epochs) for all architectures and kernel sizes."""
    training_data = {}
    log_files = sorted(raw_dir.glob("train_*.log"))
    
    print(f"Found {len(log_files)} training log files")
    
    for log_path in log_files:
        # Extract architecture and kernel size from filename
        # Format: train_<arch>_k<kernel>_e<epochs>_bs<batch>_w<workers>.log
        stem = log_path.stem
        tokens = stem.split("_")
        print(f"Processing {log_path.name}: {len(tokens)} tokens")
        if len(tokens) < 6:
            print(f"  Skipping - not enough tokens")
            continue
        
        # Extract architecture (second token) and kernel size (third token)
        arch_key = tokens[1]
        kernel_token = tokens[2]
        if not kernel_token.startswith("k"):
            print(f"  Skipping - invalid kernel token: {kernel_token}")
            continue
        try:
            kernel_size = int(kernel_token[1:])
        except ValueError:
            print(f"  Skipping - invalid kernel size: {kernel_token}")
            continue
        
        print(f"  Architecture: {arch_key}, Kernel Size: {kernel_size}")
        
        # Parse log file
        epochs = []
        train_losses = []
        val_top5_accs = []
        
        with open(log_path, 'r', encoding='utf-8') as f:
            in_data_section = False
            for line in f:
                # Skip until we find the table separator (look for pattern with dashes)
                if '------+' in line and '+-' in line:
                    in_data_section = True
                    print(f"  Found data section separator")
                    continue
                
                # Parse data lines: "    1 |     2.3456 |    12.34% |     2.1234 |    23.45% |    45.67% |    0.1234s"
                if in_data_section and '|' in line and line.strip():
                    parts = line.split('|')
                    if len(parts) >= 7:
                        try:
                            epoch = int(parts[0].strip())
                            train_loss = float(parts[1].strip())
                            val_top5_str = parts[5].strip().rstrip('%')
                            val_top5 = float(val_top5_str)
                            
                            epochs.append(epoch)
                            train_losses.append(train_loss)
                            val_top5_accs.append(val_top5)
                        except (ValueError, IndexError) as e:
                            continue
        
        print(f"  Parsed {len(epochs)} epochs")
        if epochs:
            key = (arch_key, kernel_size)
            training_data[key] = {
                'epochs': epochs,
                'top5_acc': val_top5_accs,
                'loss': train_losses
            }
    
    if not training_data:
        print("No training logs found.")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot Top-5 Accuracy
    for (arch_key, kernel_size), data in sorted(training_data.items()):
        arch_name = TOPOLOGY_NAMES.get(arch_key, arch_key)
        color = ARCH_COLORS.get(arch_key, 'gray')
        marker = KERNEL_MARKERS.get(kernel_size, 'o')
        linestyle = KERNEL_LINESTYLES.get(kernel_size, '-')
        
        label = f"{arch_name} (K={kernel_size})"
        ax1.plot(data['epochs'], data['top5_acc'], 
                marker=marker, markersize=4, linewidth=2, 
                linestyle=linestyle, label=label, color=color, alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Top-5 Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Training Progress: Top-5 Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, title='Higher is better', title_fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    
    # Plot Loss
    for (arch_key, kernel_size), data in sorted(training_data.items()):
        arch_name = TOPOLOGY_NAMES.get(arch_key, arch_key)
        color = ARCH_COLORS.get(arch_key, 'gray')
        marker = KERNEL_MARKERS.get(kernel_size, 'o')
        linestyle = KERNEL_LINESTYLES.get(kernel_size, '-')
        
        label = f"{arch_name} (K={kernel_size})"
        ax2.plot(data['epochs'], data['loss'], 
                marker=marker, markersize=4, linewidth=2, 
                linestyle=linestyle, label=label, color=color, alpha=0.8)
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Training Progress: Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, title='Lower is better', title_fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    
    fig.tight_layout()
    
    output_path = output_dir / "train_progress.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    
    print(f"Training progress plot saved: {output_path}")


def _plot_unified_inference_comparison_DISABLED(data: RunData, output_dir: Path) -> None:
    """Generate unified comparison plot for inference across all workers, excluding mFFT."""
    inference_data = [record for record in data if record.run_type == "inference"]
    
    # Filter out mFFT AlexNet
    inference_data = [record for record in inference_data if record.arch_key != "mfft_alexnet"]
    
    if not inference_data:
        return
    
    # Detect if we have CPU or GPU data
    has_cpu_data = any(rec.monitor_mode == "cpu" for rec in inference_data)
    
    # Group by (arch, workers, iterations) to get all combinations
    arch_worker_groups = {}
    for record in inference_data:
        key = (record.arch_key, record.num_workers, record.iterations)
        if key not in arch_worker_groups:
            arch_worker_groups[key] = []
        arch_worker_groups[key].append(record)
    
    # Check if we have multiple worker values
    all_workers = sorted(set(record.num_workers for record in inference_data))
    all_iterations = sorted(set(record.iterations for record in inference_data if record.iterations))
    
    if not all_iterations:
        return
    
    # For each iteration value, create a unified plot
    for iterations in all_iterations:
        # Prepare metrics for each (arch, workers) combination
        unified_metrics = {}
        
        for (arch_key, workers, iters), records in arch_worker_groups.items():
            if iters != iterations or len(records) <= 1:
                continue
            
            records = sorted(records, key=lambda r: r.batch_size)
            batch_sizes = [r.batch_size for r in records]
            energies = [r.energy_j for r in records]
            energy_stddevs = [r.energy_stddev_j for r in records]
            
            # Calculate metrics
            runtimes = []
            runtime_stddevs = []
            avg_utils = []
            avg_mem_utils = []
            
            for record in records:
                if record.monitor_mode == "cpu":
                    runtime = record.df["time_s"].iloc[0] if "time_s" in record.df.columns else 0.0
                    runtime_std = record.time_stddev_s
                    runtimes.append(runtime)
                    runtime_stddevs.append(runtime_std)
                    avg_utils.append(0.0)
                    avg_mem_utils.append(0.0)
                else:
                    runtime = record.df["time_s"].max() if not record.df.empty else 0.0
                    runtime_std = 0.0
                    runtimes.append(runtime)
                    runtime_stddevs.append(runtime_std)
                    avg_utils.append(record.df["util_pct"].mean() if "util_pct" in record.df.columns else 0.0)
                    avg_mem_utils.append(record.df["mem_util_pct"].mean() if "mem_util_pct" in record.df.columns else 0.0)
            
            # Calculate throughput and efficiency
            total_items = iterations * TEST_DATASET_SIZE
            throughputs = [total_items / rt if rt > 0 else 0 for rt in runtimes]
            energy_efficiency = [total_items / e if e > 0 else 0 for e in energies]
            
            unified_metrics[(arch_key, workers)] = {
                'batch_sizes': batch_sizes,
                'energies': energies,
                'runtimes': runtimes,
                'throughputs': throughputs,
                'energy_efficiency': energy_efficiency,
                'avg_utils': avg_utils,
                'avg_mem_utils': avg_mem_utils,
            }
        
        if not unified_metrics:
            continue
        
        # Determine number of subplots
        if has_cpu_data:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        else:
            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 18))
        
        # Plot data for each (architecture, workers) combination
        for (arch_key, workers), metrics in unified_metrics.items():
            arch_name = TOPOLOGY_NAMES.get(arch_key, arch_key)
            color = ARCH_COLORS.get(arch_key, 'gray')
            linestyle = WORKER_LINESTYLES.get(workers, '-')
            marker = WORKER_MARKERS.get(workers, 'o')
            label = f"{arch_name} (w={workers})"
            
            batch_sizes = metrics['batch_sizes']
            
            # Plot 1: Energy consumption
            ax1.plot(batch_sizes, metrics['energies'],
                    marker=marker, linewidth=2, markersize=6, label=label, 
                    color=color, linestyle=linestyle, alpha=0.8)
            
            # Plot 2: Execution time
            ax2.plot(batch_sizes, metrics['runtimes'],
                    marker=marker, linewidth=2, markersize=6, label=label, 
                    color=color, linestyle=linestyle, alpha=0.8)
            
            # Plot 3: Throughput
            ax3.plot(batch_sizes, metrics['throughputs'],
                    marker=marker, linewidth=2, markersize=6, label=label, 
                    color=color, linestyle=linestyle, alpha=0.8)
            
            # Plot 4: Energy efficiency
            ax4.plot(batch_sizes, metrics['energy_efficiency'],
                    marker=marker, linewidth=2, markersize=6, label=label, 
                    color=color, linestyle=linestyle, alpha=0.8)
            
            # GPU-specific plots
            if not has_cpu_data:
                # Plot 5: GPU Utilization
                ax5.plot(batch_sizes, metrics['avg_utils'], marker=marker, linewidth=2, 
                        markersize=6, label=label, color=color, linestyle=linestyle, alpha=0.8)
                
                # Plot 6: Memory Utilization
                ax6.plot(batch_sizes, metrics['avg_mem_utils'], marker=marker, linewidth=2, 
                        markersize=6, label=label, color=color, linestyle=linestyle, alpha=0.8)
        
        # Find best energy efficiency for annotation
        best_eff = {'value': 0, 'arch': None, 'workers': None, 'batch_size': None}
        for (arch_key, workers), metrics in unified_metrics.items():
            for i, bs in enumerate(metrics['batch_sizes']):
                if metrics['energy_efficiency'][i] > best_eff['value']:
                    best_eff['value'] = metrics['energy_efficiency'][i]
                    best_eff['arch'] = TOPOLOGY_NAMES.get(arch_key, arch_key)
                    best_eff['workers'] = workers
                    best_eff['batch_size'] = bs
        
        # Get all unique batch sizes
        all_batch_sizes = sorted(set(bs for metrics in unified_metrics.values() for bs in metrics['batch_sizes']))
        
        from matplotlib.ticker import FuncFormatter
        def batch_size_formatter(x, pos):
            return f'{int(x)}'
        formatter = FuncFormatter(batch_size_formatter)
        
        # Configure axes
        # Axis 1: Energy Consumption
        ax1.set_xlabel("Batch Size", fontsize=11)
        ax1.set_ylabel("Total Energy (J)", fontsize=11)
        ax1.set_xscale('log', base=2)
        ax1.set_xticks(all_batch_sizes)
        ax1.xaxis.set_major_formatter(formatter)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, loc='best', ncol=2, title='Lower is better', title_fontsize=9)
        ax1.set_title("Energy Consumption", fontsize=12, fontweight='bold')
        
        # Axis 2: Execution Time
        ax2.set_xlabel("Batch Size", fontsize=11)
        ax2.set_ylabel("Execution Time (s)", fontsize=11)
        ax2.set_xscale('log', base=2)
        ax2.set_xticks(all_batch_sizes)
        ax2.xaxis.set_major_formatter(formatter)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8, loc='best', ncol=2, title='Lower is better', title_fontsize=9)
        ax2.set_title("Execution Time", fontsize=12, fontweight='bold')
        
        # Axis 3: Throughput
        ax3.set_xlabel("Batch Size", fontsize=11)
        ax3.set_ylabel("Throughput (images/s)", fontsize=11)
        ax3.set_xscale('log', base=2)
        ax3.set_xticks(all_batch_sizes)
        ax3.xaxis.set_major_formatter(formatter)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8, loc='best', ncol=2, title='Higher is better', title_fontsize=9)
        ax3.set_title("Throughput", fontsize=12, fontweight='bold')
        
        # Axis 4: Energy Efficiency (with annotation)
        ax4.set_xlabel("Batch Size", fontsize=11)
        ax4.set_ylabel("Energy Efficiency (images/J)", fontsize=11)
        ax4.set_xscale('log', base=2)
        ax4.set_xticks(all_batch_sizes)
        ax4.xaxis.set_major_formatter(formatter)
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8, loc='best', ncol=2, title='Higher is better', title_fontsize=9)
        ax4.set_title("Energy Efficiency", fontsize=12, fontweight='bold')
        
        # Add annotation for best efficiency
        if best_eff['arch']:
            ax4.annotate(f"Best: {best_eff['arch']}\nw={best_eff['workers']}, bs={best_eff['batch_size']}\n{best_eff['value']:.1f} images/J",
                        xy=(best_eff['batch_size'], best_eff['value']), xytext=(10, -20),
                        textcoords='offset points', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='green', lw=1.5))
        
        # GPU-specific axes
        if not has_cpu_data:
            # Axis 5: GPU Utilization
            ax5.set_xlabel("Batch Size", fontsize=11)
            ax5.set_ylabel("GPU Utilization (%)", fontsize=11)
            ax5.set_xscale('log', base=2)
            ax5.set_xticks(all_batch_sizes)
            ax5.xaxis.set_major_formatter(formatter)
            ax5.set_ylim(0, 100)
            ax5.grid(True, alpha=0.3)
            ax5.legend(fontsize=8, loc='best', ncol=2, title='Higher is better', title_fontsize=9)
            ax5.set_title("GPU Utilization", fontsize=12, fontweight='bold')
            
            # Axis 6: Memory Utilization
            ax6.set_xlabel("Batch Size", fontsize=11)
            ax6.set_ylabel("Memory Utilization (%)", fontsize=11)
            ax6.set_xscale('log', base=2)
            ax6.set_xticks(all_batch_sizes)
            ax6.xaxis.set_major_formatter(formatter)
            ax6.set_ylim(0, 100)
            ax6.grid(True, alpha=0.3)
            ax6.legend(fontsize=8, loc='best', ncol=2, title='Lower is better', title_fontsize=9)
            ax6.set_title("Memory Utilization", fontsize=12, fontweight='bold')
        
        # Generate title and filename
        fig.suptitle(f"Unified Inference Comparison - All Workers (iter={iterations})", 
                    fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        output_name = f"inference_unified_comparison_i{iterations}.png"
        output_path = output_dir / output_name
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Unified inference comparison saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GPU DSE metrics.")
    parser.add_argument(
        "--raw-dir",
        default="results/dse/raw",
        type=Path,
        help="Directory containing *_metrics.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/dse/plots",
        type=Path,
        help="Directory to store generated plots.",
    )
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean output directory
    print("Cleaning output directory...")
    for file in output_dir.glob("*.png"):
        file.unlink()
    print(f"Output directory cleaned: {output_dir}\n")

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw metrics directory '{raw_dir}' not found.")

    data = _load_run_data(raw_dir)
    if not data:
        raise RuntimeError(f"No metrics CSV files found in '{raw_dir}'.")

    # Generate all plots
    print("Generating trainable parameters comparison...")
    _plot_trainable_parameters(raw_dir, output_dir)
    
    print("\nGenerating training progress plots...")
    _plot_training_progress(raw_dir, output_dir)
    
    print("\nGenerating topology comparison by kernel size...")
    _plot_topology_comparison_by_kernel(data, output_dir)
    
    print("\nGenerating comparative metrics plots for training...")
    _plot_comparative_metrics(data, output_dir, run_type="train")
    
    print("\nGenerating comparative metrics plots for inference...")
    _plot_comparative_metrics(data, output_dir, run_type="inference")
    
    print("\nGenerating energy ranking plots...")
    _plot_energy_ranking(data, output_dir)
    
    print(f"\nAll plots saved to {output_dir}")



if __name__ == "__main__":
    main()

