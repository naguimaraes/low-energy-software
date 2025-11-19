#!/usr/bin/env python3
"""Shared helpers for optional per-layer timing in AlexNet scripts."""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import torch


class LayerTimer:
    """Collects forward-pass timings for leaf modules in a model."""

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.device = device
        self._module_meta: Dict[int, Tuple[str, str, str]] = {}
        self._module_meta_by_name: Dict[str, Tuple[str, str]] = {}
        self._forward_totals = defaultdict(float)
        self._forward_counts = defaultdict(int)
        self._forward_starts: Dict[int, float] = {}
        self._handles = []
        self._register_hooks(model)

    def _register_hooks(self, model: torch.nn.Module) -> None:
        for name, module in model.named_modules():
            if len(list(module.children())) > 0:
                continue
            module_id = id(module)
            module_name = name or module.__class__.__name__
            class_name = module.__class__.__name__
            group = "features" if module_name.startswith("features") else "classifier" if module_name.startswith("classifier") else "other"
            self._module_meta[module_id] = (module_name, class_name, group)
            self._module_meta_by_name[module_name] = (class_name, group)
            self._handles.append(module.register_forward_pre_hook(self._forward_pre_hook))
            self._handles.append(module.register_forward_hook(self._forward_hook))

    def _sync_if_needed(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _forward_pre_hook(self, module: torch.nn.Module, _inputs) -> None:
        self._sync_if_needed()
        self._forward_starts[id(module)] = time.perf_counter()

    def _forward_hook(self, module: torch.nn.Module, _inputs, _output) -> None:
        self._sync_if_needed()
        module_id = id(module)
        start_time = self._forward_starts.pop(module_id, None)
        if start_time is None:
            return
        elapsed = time.perf_counter() - start_time
        module_name, class_name, _group = self._module_meta[module_id]
        key = module_name
        self._forward_totals[key] += elapsed
        self._forward_counts[key] += 1

    def reset(self) -> None:
        self._forward_totals.clear()
        self._forward_counts.clear()
        self._forward_starts.clear()

    def summary(self) -> List[Dict[str, float]]:
        rows = []
        for module_name in sorted(self._forward_totals, key=self._forward_totals.get, reverse=True):
            total_ms = self._forward_totals[module_name] * 1_000.0
            calls = self._forward_counts[module_name]
            class_name, group = self._module_meta_by_name[module_name]
            avg_ms = total_ms / calls if calls else 0.0
            rows.append({
                "module": module_name,
                "layer": class_name,
                "group": group,
                "calls": calls,
                "total_ms": total_ms,
                "avg_ms": avg_ms,
            })
        return rows

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def save_layer_times(summary: List[Dict[str, float]], csv_path: str) -> None:
    fieldnames = ["module", "layer", "group", "calls", "total_ms", "avg_ms"]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow({
                "module": row["module"],
                "layer": row["layer"],
                "group": row["group"],
                "calls": row["calls"],
                "total_ms": f"{row['total_ms']:.6f}",
                "avg_ms": f"{row['avg_ms']:.6f}",
            })


def plot_layer_times_pie(summary: List[Dict[str, float]], output_path: str, title: str) -> None:
    filtered_rows = [row for row in summary if row["total_ms"] > 0]
    if not filtered_rows:
        print(f"No layer timing data available to plot: {output_path}")
        return
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import cm
    except ImportError:
        print("matplotlib is not installed; skipping pie chart generation.")
        return

    sizes = [row["total_ms"] for row in filtered_rows]
    group_counts = defaultdict(int)
    for row in filtered_rows:
        group_counts[row["group"]] += 1

    cmap_lookup = {
        "features": cm.get_cmap("Blues"),
        "classifier": cm.get_cmap("Oranges"),
        "other": cm.get_cmap("Greys"),
    }

    group_progress = defaultdict(int)
    colors = []
    for row in filtered_rows:
        group = row["group"]
        cmap = cmap_lookup.get(group, cm.get_cmap("tab20"))
        count = group_counts[group]
        index = group_progress[group]
        group_progress[group] += 1
        if count == 1:
            colors.append(cmap(0.6))
        else:
            positions = np.linspace(0.3, 0.9, count)
            colors.append(cmap(positions[index]))

    indices = list(range(1, len(filtered_rows) + 1))
    wedge_labels = [str(idx) if idx <= 10 else "" for idx in indices]

    def autopct_fmt(pct):
        autopct_fmt.counter += 1
        return f"{pct:.1f}%" if autopct_fmt.counter <= 10 else ""

    autopct_fmt.counter = 0

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, _, _ = ax.pie(
        sizes,
        labels=wedge_labels,
        autopct=autopct_fmt,
        colors=colors,
        startangle=90,
        textprops={"fontsize": 10},
    )
    ax.axis("equal")
    fig.suptitle(title, fontsize=14)

    legend_labels = [
        f"{idx}. {row['module']} ({row['layer']}) - {row['total_ms']:.2f} ms (avg {row['avg_ms']:.4f} ms)"
        for idx, row in zip(indices, filtered_rows)
    ]
    ax.legend(wedges, legend_labels, title="Layers", loc="center left", bbox_to_anchor=(1.0, 0.5))

    fig.text(0.5, 0.12, "Color spectra", ha="center", va="bottom", fontsize=10)
    gradient = np.linspace(0.2, 0.9, 256).reshape(1, -1)

    features_ax = fig.add_axes([0.25, 0.06, 0.2, 0.02])
    features_ax.imshow(gradient, aspect="auto", cmap=cmap_lookup["features"])
    features_ax.axis("off")
    fig.text(0.35, 0.08, "features (Blues)", ha="center", va="bottom", fontsize=9)

    classifier_ax = fig.add_axes([0.55, 0.06, 0.2, 0.02])
    classifier_ax.imshow(gradient, aspect="auto", cmap=cmap_lookup["classifier"])
    classifier_ax.axis("off")
    fig.text(0.65, 0.08, "classifier (Oranges)", ha="center", va="bottom", fontsize=9)

    fig.subplots_adjust(right=0.72, bottom=0.32, top=0.92)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_layer_group_pie(summary: List[Dict[str, float]], output_path: str, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping pie chart generation.")
        return

    totals = defaultdict(float)
    for row in summary:
        totals[row["group"]] += row["total_ms"]

    filtered = [(group, total) for group, total in totals.items() if total > 0]
    if not filtered:
        print(f"No group timing data available to plot: {output_path}")
        return

    labels = [group for group, _ in filtered]
    sizes = [total for _, total in filtered]
    color_map = {
        "features": "#1f77b4",
        "classifier": "#ff7f0e",
        "other": "#7f7f7f",
    }
    colors = [color_map.get(group, "#bbbbbb") for group, _ in filtered]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, _, _ = ax.pie(sizes, labels=None, autopct="%1.1f%%", colors=colors, startangle=90, pctdistance=0.7)
    ax.axis("equal")
    ax.set_title(title)
    legend_entries = [f"{label} - {size:.2f} ms" for label, size in zip(labels, sizes)]
    ax.legend(wedges, legend_entries, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.subplots_adjust(right=0.75)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
