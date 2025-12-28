#!/usr/bin/env python3
"""
Baseline Aggregate Plot

Compares recall@k and mAP@k metrics across multiple result directories.
Each directory should contain a recall_results.json file with individual_results.

Usage:
    python -m evaluate.baseline_aggregate_plot \
        --dirs results/method1 results/method2 results/method3 \
        --output results/comparison \
        --title "Method Comparison"

Sample Execution:
    python -m evaluate.baseline_aggregate_plot \
        --dirs results/full_image results/screenshot_image results/fusion_text_first \
        --output results/comparison \
        --title "Baseline Comparison"
"""

import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt

from evaluate.recall_utils import compute_average_metrics


def load_results_from_dir(result_dir: Path) -> tuple[List[Dict], Dict]:
    json_file = result_dir / "recall_results.json"
    if not json_file.exists():
        raise FileNotFoundError(f"recall_results.json not found in {result_dir}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    individual_results = data.get('individual_results', [])
    config = data.get('config', {})
    
    if len(individual_results) == 0:
        raise ValueError(f"No individual_results found in {result_dir}")
    
    return individual_results, config


def format_directory_name(dir_path: Path) -> str:
    name = dir_path.name
    name = name.replace('_', ' ')
    name = ' '.join(word.capitalize() for word in name.split())
    return name


def plot_aggregate_comparison(
    dir_averages: Dict[str, Dict],
    output_path: Path,
    title: str = "Baseline Comparison"
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14), sharex=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(dir_averages)))
    
    all_recall_k_values = set()
    all_map_k_values = set()
    
    recall_curves = {}
    for dir_name, averages in dir_averages.items():
        if 'error' in averages or 'average_metrics' not in averages:
            continue
        
        avg_metrics = averages['average_metrics']
        recall_list = []
        
        for metric_name, stats in avg_metrics.items():
            if metric_name.startswith('recall@'):
                k = int(metric_name.split('@')[1])
                recall = stats.get('mean', 0.0)
                recall_list.append((k, recall))
                all_recall_k_values.add(k)
        
        recall_list.sort(key=lambda x: x[0])
        if len(recall_list) > 0:
            recall_curves[dir_name] = {
                'k_values': [k for k, _ in recall_list],
                'scores': [recall for _, recall in recall_list]
            }
    
    map_curves = {}
    for dir_name, averages in dir_averages.items():
        if 'error' in averages or 'average_metrics' not in averages:
            continue
        
        avg_metrics = averages['average_metrics']
        map_list = []
        
        for metric_name, stats in avg_metrics.items():
            if metric_name.startswith('map@'):
                k = int(metric_name.split('@')[1])
                map_score = stats.get('mean', 0.0)
                map_list.append((k, map_score))
                all_map_k_values.add(k)
        
        map_list.sort(key=lambda x: x[0])
        if len(map_list) > 0:
            map_curves[dir_name] = {
                'k_values': [k for k, _ in map_list],
                'scores': [map_score for _, map_score in map_list]
            }
    
    all_recall_k_values.add(0)
    all_map_k_values.add(0)
    
    for i, (dir_name, curve_data) in enumerate(recall_curves.items()):
        k_values = curve_data['k_values']
        scores = curve_data['scores']
        
        if k_values and k_values[0] > 0:
            k_values = [0] + k_values
            scores = [0.0] + scores
        
        formatted_name = format_directory_name(Path(dir_name))
        ax1.plot(k_values, scores, marker='o', linewidth=4, markersize=12,
                color=colors[i], label=formatted_name, zorder=10)
        
        for k, score in zip(k_values, scores):
            if k > 0:
                ax1.annotate(f'{score:.3f}', (k, score),
                           textcoords="offset points", xytext=(0, 10),
                           ha='center', fontsize=12, color=colors[i])
    
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlabel("k (Number of Results)", fontsize=16)
    ax1.set_ylabel("Recall@K", fontsize=16)
    ax1.set_title(f"Recall@K Comparison - {title}", fontsize=18, fontweight='bold')
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0, top=1.05)
    if all_recall_k_values:
        ax1.set_xticks(sorted(all_recall_k_values))
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.legend(loc='upper left', fontsize=18, framealpha=0.9)
    
    for i, (dir_name, curve_data) in enumerate(map_curves.items()):
        k_values = curve_data['k_values']
        scores = curve_data['scores']
        
        if k_values and k_values[0] > 0:
            k_values = [0] + k_values
            scores = [0.0] + scores
        
        formatted_name = format_directory_name(Path(dir_name))
        ax2.plot(k_values, scores, marker='s', linewidth=4, markersize=12,
                color=colors[i], label=formatted_name, zorder=10)
        
        for k, score in zip(k_values, scores):
            if k > 0:
                ax2.annotate(f'{score:.3f}', (k, score),
                           textcoords="offset points", xytext=(0, 10),
                           ha='center', fontsize=12, color=colors[i])
    
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlabel("k (Number of Results)", fontsize=16)
    ax2.set_ylabel("mAP@K", fontsize=16)
    ax2.set_title(f"mAP@K Comparison - {title}", fontsize=20, fontweight='bold')
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0, top=1.05)
    if all_map_k_values:
        ax2.set_xticks(sorted(all_map_k_values))
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.legend(loc='upper left', fontsize=18, framealpha=0.9)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved comparison plot to: {output_path}")


def save_metrics_to_csv(
    dir_averages: Dict[str, Dict],
    output_path: Path
):
    all_metrics = set()
    for averages in dir_averages.values():
        if 'error' not in averages and 'average_metrics' in averages:
            all_metrics.update(averages['average_metrics'].keys())
    
    recall_metrics = sorted([m for m in all_metrics if m.startswith('recall@')], 
                           key=lambda x: int(x.split('@')[1]))
    map_metrics = sorted([m for m in all_metrics if m.startswith('map@')],
                        key=lambda x: int(x.split('@')[1]))
    sorted_metrics = recall_metrics + map_metrics
    
    rows = []
    
    header = ['Method', 'Num Queries']
    for metric in sorted_metrics:
        header.extend([f'{metric}_mean', f'{metric}_std', f'{metric}_min', f'{metric}_max', f'{metric}_median'])
    rows.append(header)
    
    for dir_name, averages in dir_averages.items():
        if 'error' in averages:
            continue
        
        formatted_name = format_directory_name(Path(dir_name))
        row = [formatted_name]
        
        row.append(averages.get('num_queries', 0))
        
        avg_metrics = averages.get('average_metrics', {})
        for metric in sorted_metrics:
            if metric in avg_metrics:
                stats = avg_metrics[metric]
                row.extend([
                    f"{stats.get('mean', 0.0):.6f}",
                    f"{stats.get('std', 0.0):.6f}",
                    f"{stats.get('min', 0.0):.6f}",
                    f"{stats.get('max', 0.0):.6f}",
                    f"{stats.get('median', 0.0):.6f}"
                ])
            else:
                row.extend(['N/A', 'N/A', 'N/A', 'N/A', 'N/A'])
        
        rows.append(row)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"Saved metrics CSV to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare recall@k and mAP@k metrics across multiple result directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m evaluate.baseline_aggregate_plot \
      --dirs results/full_image results/screenshot_image results/fusion_text_first \
      --output results/comparison \
      --title "Baseline Comparison"
  
  python -m evaluate.baseline_aggregate_plot \
      --dirs results/fusion_text_first/alpha_0.5 results/fusion_text_first/alpha_0.8 \
      --output results/alpha_comparison \
      --title "Alpha Parameter Comparison"
        """
    )
    parser.add_argument(
        "--dirs",
        type=str,
        nargs='+',
        required=True,
        help="List of directories containing recall_results.json files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory to save the comparison plot",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Baseline Comparison",
        help="Title for the comparison plot (default: 'Baseline Comparison')",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="aggregate_comparison.png",
        help="Filename for the output plot (default: 'aggregate_comparison.png')",
    )
    
    args = parser.parse_args()
    
    dir_paths = [Path(d) for d in args.dirs]
    for dir_path in dir_paths:
        if not dir_path.exists():
            print(f"Error: Directory not found: {dir_path}")
            return
        json_file = dir_path / "recall_results.json"
        if not json_file.exists():
            print(f"Error: recall_results.json not found in {dir_path}")
            return
    
    print(f"\nBaseline Aggregate Plot")
    print("="*60)
    print(f"Comparing {len(dir_paths)} directories:")
    for dir_path in dir_paths:
        print(f"  - {dir_path}")
    print(f"Output: {args.output}")
    print(f"Title: {args.title}")
    print("="*60)
    
    dir_averages = {}
    for dir_path in dir_paths:
        try:
            print(f"\nLoading results from: {dir_path}")
            individual_results, config = load_results_from_dir(dir_path)
            print(f"  Found {len(individual_results)} individual results")
            
            averages = compute_average_metrics(individual_results)
            dir_averages[str(dir_path)] = averages
            
            if 'error' not in averages:
                print(f"  Valid queries: {averages['num_queries']}")
                if averages['num_failed'] > 0:
                    print(f"  Failed queries: {averages['num_failed']}")
                
                avg_metrics = averages.get('average_metrics', {})
                recall_metrics = {k: v for k, v in avg_metrics.items() if k.startswith('recall@')}
                map_metrics = {k: v for k, v in avg_metrics.items() if k.startswith('map@')}
                
                if recall_metrics:
                    sample_recall = sorted(recall_metrics.items())[0]
                    print(f"  Sample recall@{sample_recall[0].split('@')[1]}: {sample_recall[1]['mean']:.4f}")
                if map_metrics:
                    sample_map = sorted(map_metrics.items())[0]
                    print(f"  Sample map@{sample_map[0].split('@')[1]}: {sample_map[1]['mean']:.4f}")
            else:
                print(f"  Error: {averages.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"Error loading results from {dir_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(dir_averages) == 0:
        print("\nError: No valid results loaded. Cannot create comparison plot.")
        return
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / args.filename
    
    print(f"\nCreating comparison plot...")
    plot_aggregate_comparison(dir_averages, output_file, title=args.title)
    
    csv_filename = args.filename.replace('.png', '.csv').replace('.jpg', '.csv')
    csv_file = output_dir / csv_filename
    print(f"\nSaving metrics to CSV...")
    save_metrics_to_csv(dir_averages, csv_file)
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print(f"Plot saved to: {output_file}")
    print(f"Metrics CSV saved to: {csv_file}")
    print(f"Compared {len(dir_averages)} directories")
    print("="*60)


if __name__ == "__main__":
    main()