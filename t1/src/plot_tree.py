#!/usr/bin/env python3
"""
Analyze results from tree search performance experiments.
Processes cache metrics, time, and energy measurements for Trie and If-Then-Else structures.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np
import re

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(HERE, "..", "results", "tree"))
PLOTS_DIR = os.path.normpath(os.path.join(HERE, "..", "plots", "tree"))

# Create plots directory
os.makedirs(PLOTS_DIR, exist_ok=True)

def parse_cache_file(filename):
    """Parse perf text output for cache (mean line) from *_cache.out."""
    try:
        mean_val = np.nan
        with open(filename, 'r') as f:
            for line in f:
                if 'L1-dcache-load-misses' in line and '+-' in line:
                    # perf format:  <count> <unit> L1-dcache-load-misses # ... +- <pct>%
                    parts = line.strip().split()
                    if parts:
                        # First token is the count (may contain commas)
                        count = parts[0].replace(',', '')
                        try:
                            mean_val = float(count)
                        except Exception:
                            mean_val = np.nan
                    break
        return { 'L1-dcache-load-misses': mean_val }
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
        return None

def parse_time_file(filename):
    """Parse time measurements file."""
    try:
        with open(filename, 'r') as f:
            times = [float(line.strip()) for line in f if line.strip()]
        return times if times else None
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
        return None

def parse_energy_file(filename):
    """Parse energy measurements file."""
    try:
        with open(filename, 'r') as f:
            energies = [float(line.strip()) for line in f if line.strip()]
        return energies if energies else None
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
        return None

def detect_dataset(main_out_path):
    try:
        if not os.path.exists(main_out_path):
            return 'unknown'
        with open(main_out_path, 'r') as f:
            head = f.read(500).lower()
            if 'players' in head:
                return 'players'
            if 'bible' in head:
                return 'bible'
        return 'unknown'
    except Exception:
        return 'unknown'

def parse_timing_breakdown(main_out_path):
    """Parse TIMING_MS lines from the main stdout file and return average breakdown in seconds.
    Expected line format: TIMING_MS sort=<ms> build=<ms> search=<ms> total=<ms>
    Returns dict with keys: sort_s, build_s, search_s, total_s (averaged over all occurrences).
    """
    try:
        if not os.path.exists(main_out_path):
            return None
        sort_vals = []
        build_vals = []
        search_vals = []
        total_vals = []
        # Correct regex: match whitespace between tokens
        pattern = re.compile(r"TIMING_MS\s+sort=([0-9.]+)\s+build=([0-9.]+)\s+search=([0-9.]+)\s+total=([0-9.]+)")
        with open(main_out_path, 'r') as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    sort_ms = float(m.group(1))
                    build_ms = float(m.group(2))
                    search_ms = float(m.group(3))
                    total_ms = float(m.group(4))
                    sort_vals.append(sort_ms)
                    build_vals.append(build_ms)
                    search_vals.append(search_ms)
                    total_vals.append(total_ms)
        if not sort_vals:
            return None
        # Convert ms to seconds for plotting alongside perf time
        to_s = lambda arr: float(np.mean(arr)) / 1000.0
        return {
            'sort_s': to_s(sort_vals),
            'build_s': to_s(build_vals),
            'search_s': to_s(search_vals),
            'total_s': to_s(total_vals),
        }
    except Exception as e:
        print(f"Error parsing timing from {main_out_path}: {e}")
        return None

def collect_results():
    """Collect all experimental results from ../results/tree with .out files."""
    results = []
    cache_files = glob.glob(os.path.join(RESULTS_DIR, "*_cache.out"))

    for cache_file in cache_files:
        basename = os.path.basename(cache_file)
        stem = basename.replace("_cache.out", "")
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        # Be robust: last 3 parts are (structure, word, word_type); leading parts may include dataset
        structure, word, word_type = parts[-3], parts[-2], parts[-1]

        cache_metrics = parse_cache_file(cache_file)
        if cache_metrics is None:
            continue

        time_file = os.path.join(RESULTS_DIR, f"{stem}_time.out")
        energy_file = os.path.join(RESULTS_DIR, f"{stem}_energy.out")
        main_out = os.path.join(RESULTS_DIR, f"{stem}.out")
        dataset = detect_dataset(main_out)
        if (not dataset or dataset == 'unknown') and len(parts) >= 4:
            # Fallback: first token is likely dataset when present in stem
            dataset = parts[0]
        timing = parse_timing_breakdown(main_out)

        times = parse_time_file(time_file) if os.path.exists(time_file) else None
        energies = parse_energy_file(energy_file) if os.path.exists(energy_file) else None

        result = {
            'dataset': dataset,
            'structure': structure,
            'word': word,
            'word_type': word_type,
            'l1_dcache_load_misses': cache_metrics.get('L1-dcache-load-misses', np.nan),
            'time_avg': np.mean(times) if times else np.nan,
            'time_std': np.std(times) if times else np.nan,
            'energy_avg': np.mean(energies) if energies else np.nan,
            'energy_std': np.std(energies) if energies else np.nan,
        }
        if timing:
            result.update(timing)
        results.append(result)

    return pd.DataFrame(results)

def plot_overall_by_metric(df, metric, ylabel, title, filename):
    """Single bar per structure (Trie vs If-Then-Else) representing the full run (40 searches)."""
    data = df.copy()
    # Expect one row per structure; if multiple, take mean
    summary = data.groupby('structure')[metric].mean().reindex(['trie', 'ifthenelse'])
    x_labels = summary.index.tolist()
    x = np.arange(len(x_labels))
    width = 0.5
    plt.figure(figsize=(10, 6))
    plt.bar(x, summary.values, width, color=['#6baed6', '#31a354'], alpha=0.85)
    plt.xticks(x, ['Trie', 'If-Then-Else'])
    plt.ylabel(ylabel)
    plt.title(title)
    for xi, val in zip(x, summary.values):
        plt.text(xi, val, f"{val:.3g}", ha='center', va='bottom', fontsize=9)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_time_breakdown_overall(df, title, filename):
    """Stacked bars (Trie vs If-Then-Else) showing sort/build/search for the full 40-query run."""
    required = ['sort_s', 'build_s', 'search_s']
    if not all(col in df.columns for col in required):
        print(f"Skipping {filename} (no timing breakdown columns found)")
        return
    data = df.groupby('structure')[required].mean().reindex(['trie', 'ifthenelse'])
    if data.isna().all().all():
        print(f"Skipping {filename} (no timing breakdown values)")
        return
    x = np.arange(len(data))
    width = 0.6
    plt.figure(figsize=(12, 6))
    # Distinct, high-contrast colors consistent across datasets
    # sort: orange, build: blue, search: green
    p1 = plt.bar(x, data['sort_s'], width, label='Sort', color='#ff7f0e')
    p2 = plt.bar(x, data['build_s'], width, bottom=data['sort_s'], label='Build', color='#1f77b4')
    p3 = plt.bar(x, data['search_s'], width, bottom=data['sort_s'] + data['build_s'], label='Search', color='#2ca02c')
    plt.xticks(x, ['Trie', 'If-Then-Else'])
    plt.ylabel('Time (seconds)')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_edp_overall(df, title, filename):
    """Bar chart of Energy-Delay Product (EDP = energy_avg * time_avg) by structure for aggregated runs."""
    if 'energy_avg' not in df.columns or 'time_avg' not in df.columns:
        print(f"Skipping {filename} (missing energy/time columns)")
        return
    data = df.copy()
    data['edp'] = data['energy_avg'] * data['time_avg']
    summary = data.groupby('structure')['edp'].mean().reindex(['trie', 'ifthenelse'])
    if summary.isna().all():
        print(f"Skipping {filename} (no EDP values)")
        return
    x = np.arange(len(summary))
    width = 0.5
    plt.figure(figsize=(10, 6))
    plt.bar(x, summary.values, width, color=['#6baed6', '#31a354'], alpha=0.85)
    plt.xticks(x, ['Trie', 'If-Then-Else'])
    plt.ylabel('Energy-Delay Product (J·s)')
    plt.title(title)
    for xi, val in zip(x, summary.values):
        if not np.isnan(val):
            plt.text(xi, val, f"{val:.3g}", ha='center', va='bottom', fontsize=9)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_structure_comparison(df):
    """Grid of metrics comparing structures; adapted for aggregated 40-query runs (word_type ignored)."""
    summary = df.groupby(['structure']).agg({
        'l1_dcache_load_misses': 'mean',
        'time_avg': 'mean',
        'energy_avg': 'mean'
    }).reindex(['trie', 'ifthenelse'])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ('l1_dcache_load_misses', 'L1 D-Cache Load Misses', 'Average L1 D-Cache Load Misses'),
        ('time_avg', 'Time (seconds)', 'Average Execution Time'),
        ('energy_avg', 'Energy (Joules)', 'Average Energy Consumption'),
    ]
    for ax, (metric, ylabel, title) in zip(axes, metrics):
        vals = summary[metric].values
        ax.bar([0, 1], vals, color=['#6baed6', '#31a354'], alpha=0.85)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Trie', 'If-Then-Else'])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)
        for xi, val in zip([0, 1], vals):
            ax.text(xi, val, f"{val:.3g}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'structure_comparison_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: structure_comparison_summary.png")

def plot_edp_analysis(df):
    """Plot Energy-Delay Product analysis."""
    df = df.copy()
    df['edp'] = df['energy_avg'] * df['time_avg']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_existing = []
    y_existing_trie = []
    y_existing_ifthenelse = []
    x_nonexisting = []
    y_nonexisting_trie = []
    y_nonexisting_ifthenelse = []
    
    # Existing words
    for word in df[df['word_type'] == 'existing']['word'].unique():
        trie_edp = df[(df['structure'] == 'trie') & (df['word'] == word) & (df['word_type'] == 'existing')]['edp'].values
        ifthenelse_edp = df[(df['structure'] == 'ifthenelse') & (df['word'] == word) & (df['word_type'] == 'existing')]['edp'].values
        
        if len(trie_edp) > 0 and len(ifthenelse_edp) > 0:
            x_existing.append(word)
            y_existing_trie.append(trie_edp[0])
            y_existing_ifthenelse.append(ifthenelse_edp[0])
    
    # Non-existing words
    for word in df[df['word_type'] == 'nonexisting']['word'].unique():
        trie_edp = df[(df['structure'] == 'trie') & (df['word'] == word) & (df['word_type'] == 'nonexisting')]['edp'].values
        ifthenelse_edp = df[(df['structure'] == 'ifthenelse') & (df['word'] == word) & (df['word_type'] == 'nonexisting')]['edp'].values
        
        if len(trie_edp) > 0 and len(ifthenelse_edp) > 0:
            x_nonexisting.append(word)
            y_nonexisting_trie.append(trie_edp[0])
            y_nonexisting_ifthenelse.append(ifthenelse_edp[0])
    
    x_all = x_existing + ['---'] + x_nonexisting
    x_pos = np.arange(len(x_all))
    
    # Combine data with NaN for separator
    y_trie = y_existing_trie + [np.nan] + y_nonexisting_trie
    y_ifthenelse = y_existing_ifthenelse + [np.nan] + y_nonexisting_ifthenelse
    
    width = 0.35
    ax.bar(x_pos - width/2, y_trie, width, label='Trie', alpha=0.8)
    ax.bar(x_pos + width/2, y_ifthenelse, width, label='If-Then-Else', alpha=0.8)
    
    ax.set_xlabel('Search Word')
    ax.set_ylabel('Energy-Delay Product (J·s)')
    ax.set_title('Energy-Delay Product Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_all, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add vertical line to separate existing/nonexisting
    sep_pos = len(x_existing)
    ax.axvline(x=sep_pos, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'edp_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: edp_comparison.png")

def generate_summary_table(df, suffix=""):
    """Generate summary statistics table (aggregated runs)."""
    # word_type may be 'mixed' now; group only by structure for clarity
    summary = df.groupby(['structure']).agg({
        'l1_dcache_load_misses': ['mean', 'std'],
        'time_avg': ['mean', 'std'],
        'energy_avg': ['mean', 'std']
    }).round(4)
    out = f'summary_statistics{suffix}.csv'
    summary.to_csv(os.path.join(RESULTS_DIR, out))
    print("\nSummary Statistics:")
    print(summary)
    print(f"\nSaved: {out}")

def main():
    print("=" * 60)
    print("Tree Search Performance Analysis")
    print("=" * 60)
    print()
    
    # Collect results
    print("Collecting results...")
    df = collect_results()
    
    if df.empty:
        print("No results found!")
        return
    
    print(f"Found {len(df)} experimental results.")
    print()
    
    # Save complete combined results
    df.to_csv(os.path.join(RESULTS_DIR, 'complete_results.csv'), index=False)
    print("Saved: complete_results.csv")
    print()
    
    print("Generating plots per dataset (aggregated runs)...")
    for dataset in sorted(df['dataset'].dropna().unique()):
        ddf = df[df['dataset'] == dataset]
        if ddf.empty:
            continue
        suffix = f"_{dataset}"
        plot_overall_by_metric(ddf, 'l1_dcache_load_misses', 'L1 D$-$Cache Load Misses', f'Cache Misses (aggregated) — {dataset}', f'cache_misses_overall{suffix}.png')
        plot_time_breakdown_overall(ddf, f'Time Breakdown (sort/build/search) — {dataset}', f'time_breakdown_overall{suffix}.png')
        plot_overall_by_metric(ddf, 'time_avg', 'Time (seconds)', f'Execution Time (aggregated) — {dataset}', f'time_overall{suffix}.png')
        plot_overall_by_metric(ddf, 'energy_avg', 'Energy (Joules)', f'Energy Consumption (aggregated) — {dataset}', f'energy_overall{suffix}.png')
        plot_edp_overall(ddf, f'Energy-Delay Product (aggregated) — {dataset}', f'edp_overall{suffix}.png')
        plot_structure_comparison(ddf)
        # EDP plot by word no longer makes sense with aggregated run; skipping
        generate_summary_table(ddf, suffix)
    
    print()
    print("=" * 60)
    print("Analysis complete!")
    print(f"All plots saved in: {PLOTS_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
