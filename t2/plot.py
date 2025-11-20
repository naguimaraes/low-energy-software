#!/usr/bin/env python3
"""
Generate plots from DSE Training results.
"""

import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Setup style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# Constants
RESULTS_DIR = "results/dse/raw"
TRAIN_PLOTS_DIR = "results/dse/plots/train"
INFERENCE_PLOTS_DIR = "results/dse/plots/inference"

TOPOLOGY_NAMES = {
    "convolutional": "Convolutional",
    "fft": "FFT",
    "torch": "PyTorch",
}

# Order: PyTorch, FFT, Convolutional
HUE_ORDER = ["PyTorch", "FFT", "Convolutional"]

ARCH_COLORS = {
    'Convolutional': 'tab:blue',
    'FFT': 'tab:orange',
    'PyTorch': 'tab:green'
}

def parse_filename(filename):
    """Extract metadata from filename."""
    # Pattern: train_{topology}_k{kernel}_e{epochs}_bs{batch_size}_w{workers}_...
    match = re.search(r"train_([a-z]+)_k(\d+)_e(\d+)_bs(\d+)_w(\d+)", filename)
    if match:
        return {
            "topology": match.group(1),
            "kernel": int(match.group(2)),
            "epochs": int(match.group(3)),
            "batch_size": int(match.group(4)),
            "workers": int(match.group(5))
        }
    return None

def parse_training_log(filepath):
    """Parse training log file for accuracy and loss."""
    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    # Find the start of the table
    start_idx = -1
    for i, line in enumerate(lines):
        if "Epoch | Train Loss" in line:
            start_idx = i + 2 # Skip header and separator
            break
            
    if start_idx == -1:
        return None

    for line in lines[start_idx:]:
        if not line.strip():
            continue
        parts = line.split('|')
        if len(parts) < 6:
            continue
            
        try:
            epoch = int(parts[0].strip())
            train_loss = float(parts[1].strip())
            # val_top5 is the 6th column (index 5)
            val_top5_str = parts[5].strip().replace('%', '')
            val_top5 = float(val_top5_str)
            
            data.append({
                "Epoch": epoch,
                "Train Loss": train_loss,
                "Val Top-5 Accuracy": val_top5
            })
        except ValueError:
            continue
            
    return pd.DataFrame(data)

def calculate_energy_time(filepath):
    """Calculate total energy and time from metrics csv."""
    try:
        df = pd.read_csv(filepath)
        # Clean column names (remove whitespace)
        df.columns = df.columns.str.strip()
        
        if 'rel_ms' not in df.columns or 'power_W' not in df.columns:
            return None, None
            
        # Calculate time
        total_time_ms = df['rel_ms'].max() - df['rel_ms'].min()
        total_time_s = total_time_ms / 1000.0
        
        # Calculate energy (Trapezoidal rule or simple sum)
        # Energy = Power * Time
        # We can use simple rectangular approximation: Power * delta_time
        # Or trapezoidal: (P1+P2)/2 * (t2-t1)
        
        # Calculate delta time in seconds
        df['dt_s'] = df['rel_ms'].diff() / 1000.0
        df = df.dropna() # Drop first row which has NaN dt
        
        # Energy in Joules = Power (W) * Time (s)
        df['energy_J'] = df['power_W'] * df['dt_s']
        total_energy_J = df['energy_J'].sum()
        
        return total_energy_J, total_time_s
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None, None

def parse_inference_log(filepath):
    """Parse inference log for throughput."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        # Look for "Inference completed: X images in Y seconds."
        # Usually near the end
        for line in reversed(lines):
            match = re.search(r"Inference completed: (\d+) images in ([\d\.]+) seconds", line)
            if match:
                images = int(match.group(1))
                seconds = float(match.group(2))
                return images, seconds
    except Exception as e:
        print(f"Error parsing log {filepath}: {e}")
        
    return None, None

def calculate_energy_time_cpu(filepath):
    """Calculate total energy and time from perf.txt file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Extract Energy
        # 40,151.19 Joules power/energy-pkg/
        energy_match = re.search(r"([\d,]+\.\d+)\s+Joules power/energy-pkg/", content)
        if energy_match:
            energy_str = energy_match.group(1).replace(',', '')
            energy = float(energy_str)
        else:
            energy = None
            
        # Extract Time
        # 271.140248249 seconds time elapsed
        time_match = re.search(r"([\d,]+\.\d+)\s+seconds time elapsed", content)
        if time_match:
            time_str = time_match.group(1).replace(',', '')
            time_val = float(time_str)
        else:
            time_val = None
            
        return energy, time_val
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None, None

def add_bar_labels(ax):
    """Add value labels on top of bars."""
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')

def main():
    # Create plots directory
    os.makedirs(TRAIN_PLOTS_DIR, exist_ok=True)
    os.makedirs(INFERENCE_PLOTS_DIR, exist_ok=True)
    
    # Find all training log files
    log_files = glob.glob(os.path.join(RESULTS_DIR, "train_*_gpu.log"))
    
    training_data = []
    resource_data = []
    
    for log_file in log_files:
        meta = parse_filename(os.path.basename(log_file))
        if not meta:
            continue
            
        topology = meta['topology']
        if topology not in TOPOLOGY_NAMES:
            continue
            
        # Parse Training Log
        df_log = parse_training_log(log_file)
        if df_log is not None and not df_log.empty:
            df_log['Topology'] = TOPOLOGY_NAMES[topology]
            df_log['Kernel Size'] = meta['kernel']
            training_data.append(df_log)
            
        # Find corresponding metrics file
        # Pattern: train_{topology}_k{kernel}_e{epochs}_bs{batch_size}_w{workers}_metrics_gpu.csv
        metrics_filename = f"train_{topology}_k{meta['kernel']}_e{meta['epochs']}_bs{meta['batch_size']}_w{meta['workers']}_metrics_gpu.csv"
        metrics_path = os.path.join(RESULTS_DIR, metrics_filename)
        
        if os.path.exists(metrics_path):
            energy, time = calculate_energy_time(metrics_path)
            if energy is not None:
                resource_data.append({
                    "Topology": TOPOLOGY_NAMES[topology],
                    "Kernel Size": meta['kernel'],
                    "Energy (J)": energy,
                    "Execution Time (s)": time
                })

    # 1. Training Progress Plots
    if training_data:
        all_training_df = pd.concat(training_data)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Top-5 Accuracy vs Epochs
        sns.lineplot(data=all_training_df, x="Epoch", y="Val Top-5 Accuracy", 
                     hue="Topology", style="Kernel Size",
                     hue_order=HUE_ORDER, palette=ARCH_COLORS, 
                     ax=axes[0], linewidth=2)
        axes[0].set_title("Top-5 Accuracy vs Epochs")
        axes[0].set_ylabel("Top-5 Accuracy (%)")
        
        # Loss vs Epochs
        sns.lineplot(data=all_training_df, x="Epoch", y="Train Loss", 
                     hue="Topology", style="Kernel Size",
                     hue_order=HUE_ORDER, palette=ARCH_COLORS, 
                     ax=axes[1], linewidth=2)
        axes[1].set_title("Training Loss vs Epochs")
        axes[1].set_ylabel("Loss")
        
        # Adjust legend
        for ax in axes:
            # Get handles and labels
            handles, labels = ax.get_legend_handles_labels()
            # Build legends with a spacer between Topology (hue) and Kernel Size (style)
            handles, labels = ax.get_legend_handles_labels()
            # Find last index of the topology/hue entries (based on HUE_ORDER)
            hue_indices = [i for i, lab in enumerate(labels) if lab in HUE_ORDER]
            if hue_indices:
                insert_pos = hue_indices[-1] + 1
                spacer = Line2D([], [], linewidth=0, alpha=0)  # invisible handle for spacing
                handles.insert(insert_pos, spacer)
                labels.insert(insert_pos, "")  # blank label creates visual gap
            ax.legend(handles=handles, labels=labels, title="")

        plt.tight_layout()
        plt.savefig(os.path.join(TRAIN_PLOTS_DIR, "training_progress.png"))
        print(f"Saved training progress plot to {os.path.join(TRAIN_PLOTS_DIR, 'training_progress.png')}")
        plt.close()

    # 2. Training Resource Usage Plots
    if resource_data:
        resource_df = pd.DataFrame(resource_data)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Execution Time
        sns.barplot(data=resource_df, x="Kernel Size", y="Execution Time (s)", hue="Topology", 
                    hue_order=HUE_ORDER, palette=ARCH_COLORS, 
                    ax=axes[0])
        axes[0].set_title("Training Execution Time Comparison")
        axes[0].set_xlabel("Kernel Size")
        
        # Energy Consumption
        sns.barplot(data=resource_df, x="Kernel Size", y="Energy (J)", hue="Topology", 
                    hue_order=HUE_ORDER, palette=ARCH_COLORS, 
                    ax=axes[1])
        axes[1].set_title("Training Energy Consumption Comparison")
        axes[1].set_xlabel("Kernel Size")
        
        plt.tight_layout()
        plt.savefig(os.path.join(TRAIN_PLOTS_DIR, "resource_usage.png"))
        print(f"Saved training resource usage plot to {os.path.join(TRAIN_PLOTS_DIR, 'resource_usage.png')}")
        plt.close()

    # 3. Inference Resource Usage Plots (GPU & CPU)
    inference_data = []
    
    # Find all inference metrics files (both GPU and CPU if available)
    # Note: Based on file listing, CPU logs are like inference_..._cpu.log
    # But metrics seem to be mostly _metrics_gpu.csv. 
    # Let's look for all metrics files and try to deduce platform or look for corresponding logs.
    
    # Strategy: Look for all log files first to identify runs and platforms
    inference_logs = glob.glob(os.path.join(RESULTS_DIR, "inference_*.log"))
    
    for log_path in inference_logs:
        filename = os.path.basename(log_path)
        # Pattern: inference_{topology}_k{kernel}_..._{platform}.log
        # Example: inference_convolutional_k7_bs32_w1_i25_cpu.log
        # Example: inference_convolutional_k3_bs256_w2_gpu.log
        
        # Regex to capture topology, kernel, and platform (last part before .log)
        match = re.search(r"inference_([a-z]+)_k(\d+)_.*_([a-z]+)\.log", filename)
        if not match:
            continue
            
        topology = match.group(1)
        kernel = int(match.group(2))
        platform = match.group(3).upper() # CPU or GPU
        
        if topology not in TOPOLOGY_NAMES:
            continue
            
        # Parse Log for Throughput
        images, seconds = parse_inference_log(log_path)
        
        # Find corresponding metrics file
        # The metrics file naming seems to be:
        # inference_{topology}_k{kernel}_..._metrics_{platform}.csv OR just _metrics.csv?
        # From listing: inference_convolutional_k7_bs32_w1_i25_metrics_gpu.csv
        # But for CPU log: inference_convolutional_k7_bs32_w1_i25_cpu.log
        # Is there a metrics file for CPU?
        # Let's try to construct the metrics filename based on the log filename prefix
        
        prefix = filename.replace(f"_{platform.lower()}.log", "")
        # Try different metrics suffixes
        metrics_candidates = [
            f"{prefix}_metrics_{platform.lower()}.csv",
            f"{prefix}_metrics.csv"
        ]
        
        energy = None
        time_metrics = None
        
        # For GPU, use CSV metrics
        if platform == "GPU":
            for cand in metrics_candidates:
                cand_path = os.path.join(RESULTS_DIR, cand)
                if os.path.exists(cand_path):
                    energy, time_metrics = calculate_energy_time(cand_path)
                    break
        # For CPU, use perf.txt
        elif platform == "CPU":
            perf_path = os.path.join(RESULTS_DIR, f"{prefix}_perf.txt")
            if os.path.exists(perf_path):
                energy, time_metrics = calculate_energy_time_cpu(perf_path)
        
        if energy is not None and images is not None:
            throughput = images / seconds if seconds > 0 else 0
            efficiency = images / energy if energy > 0 else 0
            
            inference_data.append({
                "Topology": TOPOLOGY_NAMES[topology],
                "Kernel Size": kernel,
                "Platform": platform,
                "Energy (J)": energy,
                "Execution Time (s)": seconds, # Use time from log for consistency with throughput
                "Throughput (img/s)": throughput,
                "Efficiency (img/J)": efficiency
            })

    if inference_data:
        inference_df = pd.DataFrame(inference_data)
        
        # Split by Platform
        platforms = inference_df['Platform'].unique()
        
        for platform in platforms:
            platform_df = inference_df[inference_df['Platform'] == platform]
            
            # Create platform specific directory
            platform_dir = os.path.join(INFERENCE_PLOTS_DIR, platform.lower())
            os.makedirs(platform_dir, exist_ok=True)
            
            # 1. Full Comparison (All Topologies)
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle(f"Inference Performance ({platform})", fontsize=16)
            
            # 1. Execution Time
            sns.barplot(data=platform_df, x="Kernel Size", y="Execution Time (s)", hue="Topology", 
                        hue_order=HUE_ORDER, palette=ARCH_COLORS, 
                        ax=axes[0, 0])
            axes[0, 0].set_title("Execution Time\n(Lower is better)")
            axes[0, 0].set_xlabel("Kernel Size")
            add_bar_labels(axes[0, 0])
            
            # 2. Energy Consumption
            sns.barplot(data=platform_df, x="Kernel Size", y="Energy (J)", hue="Topology", 
                        hue_order=HUE_ORDER, palette=ARCH_COLORS, 
                        ax=axes[0, 1])
            axes[0, 1].set_title("Energy Consumption\n(Lower is better)")
            axes[0, 1].set_xlabel("Kernel Size")
            add_bar_labels(axes[0, 1])
            
            # 3. Throughput
            sns.barplot(data=platform_df, x="Kernel Size", y="Throughput (img/s)", hue="Topology", 
                        hue_order=HUE_ORDER, palette=ARCH_COLORS, 
                        ax=axes[1, 0])
            axes[1, 0].set_title("Throughput (Images/s)\n(Higher is better)")
            axes[1, 0].set_xlabel("Kernel Size")
            add_bar_labels(axes[1, 0])
            
            # 4. Efficiency
            sns.barplot(data=platform_df, x="Kernel Size", y="Efficiency (img/J)", hue="Topology", 
                        hue_order=HUE_ORDER, palette=ARCH_COLORS, 
                        ax=axes[1, 1])
            axes[1, 1].set_title("Energy Efficiency (Images/J)\n(Higher is better)")
            axes[1, 1].set_xlabel("Kernel Size")
            add_bar_labels(axes[1, 1])
            
            plt.tight_layout()
            output_file = os.path.join(platform_dir, "inference_analysis.png")
            plt.savefig(output_file)
            print(f"Saved inference analysis plot for {platform} to {output_file}")
            plt.close()

            # 2. Comparison without PyTorch (FFT vs Convolutional)
            no_torch_df = platform_df[platform_df['Topology'] != "PyTorch"]
            if not no_torch_df.empty:
                fig, axes = plt.subplots(2, 2, figsize=(14, 12))
                fig.suptitle(f"Inference Performance ({platform}) - FFT vs Convolutional", fontsize=16)
                
                metrics_config = [
                    (axes[0, 0], "Execution Time (s)", "Execution Time\n(Lower is better)", False),
                    (axes[0, 1], "Energy (J)", "Energy Consumption\n(Lower is better)", False),
                    (axes[1, 0], "Throughput (img/s)", "Throughput (Images/s)\n(Higher is better)", True),
                    (axes[1, 1], "Efficiency (img/J)", "Energy Efficiency (Images/J)\n(Higher is better)", True)
                ]
                
                # Store handles and labels for global legend
                handles, labels = None, None

                for ax, metric_col, title, higher_is_better in metrics_config:
                    sns.barplot(data=no_torch_df, x="Kernel Size", y=metric_col, hue="Topology", 
                                hue_order=["FFT", "Convolutional"], palette=ARCH_COLORS, 
                                ax=ax)
                    
                    # Get legend info from the first plot
                    if handles is None:
                        handles, labels = ax.get_legend_handles_labels()
                    
                    # Remove individual legend
                    if ax.get_legend():
                        ax.get_legend().remove()

                    ax.set_title(title)
                    ax.set_xlabel("Kernel Size")
                    add_bar_labels(ax)
                    
                    # Set y-limit to make room for annotations
                    max_val = no_torch_df[metric_col].max()
                    if pd.notna(max_val):
                        ax.set_ylim(top=max_val * 1.25)

                    # Calculate and annotate ratios
                    kernels = sorted(no_torch_df['Kernel Size'].unique())
                    for i, k in enumerate(kernels):
                        k_df = no_torch_df[no_torch_df['Kernel Size'] == k]
                        fft_row = k_df[k_df['Topology'] == "FFT"]
                        conv_row = k_df[k_df['Topology'] == "Convolutional"]
                        
                        if not fft_row.empty and not conv_row.empty:
                            fft_val = fft_row.iloc[0][metric_col]
                            conv_val = conv_row.iloc[0][metric_col]
                            
                            if higher_is_better:
                                # Ratio = FFT / Conv
                                ratio = fft_val / conv_val if conv_val > 0 else 0
                                ratio_text = f"{ratio:.2f}x"
                            else:
                                # Ratio = Conv / FFT (Speedup/Saving)
                                ratio = conv_val / fft_val if fft_val > 0 else 0
                                ratio_text = f"{ratio:.2f}x"
                            
                            # Place annotation above the highest bar
                            max_h = max(fft_val, conv_val)
                            ax.text(i, max_h + (max_h * 0.05), ratio_text, 
                                    ha='center', va='bottom', fontweight='bold', color='red')
                                    
                # Add global legend
                if handles and labels:
                    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=2)

                plt.tight_layout(rect=[0, 0, 1, 0.90])
                output_file = os.path.join(platform_dir, "inference_analysis_no_torch.png")
                plt.savefig(output_file)
                print(f"Saved inference analysis (no torch) plot for {platform} to {output_file}")
                plt.close()

if __name__ == "__main__":
    main()
