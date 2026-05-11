#!/usr/bin/env python3
"""
Plot SDMA-EP ping-pong benchmark results from JSON files.

Usage:
    python3 plot_results.py <results_dir> [--output <plot.png>]
"""

import json
import argparse
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("ERROR: matplotlib and numpy are required.")
    print("Install with: pip install matplotlib numpy")
    sys.exit(1)


def load_results(results_dir, selected_modes=None):
    """Load all JSON result files from directory.

    Args:
        results_dir: Directory containing JSON result files
        selected_modes: List of mode numbers to include (None = all modes)
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"ERROR: Directory not found: {results_dir}")
        sys.exit(1)

    json_files = sorted(results_path.glob("mode*.json"))

    if not json_files:
        print(f"ERROR: No JSON result files found in {results_dir}")
        sys.exit(1)

    all_results = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            mode = data.get('mode')

            # Filter by selected modes if specified
            if selected_modes is None or mode in selected_modes:
                all_results.append(data)

    if not all_results:
        print(f"ERROR: No results found for selected modes: {selected_modes}")
        sys.exit(1)

    order_map = {1: 0, 2: 1, 6: 2, 3: 3, 4: 4, 5: 5}
    all_results.sort(key=lambda r: order_map.get(r.get('mode', 99), 99))

    return all_results


def plot_latency_comparison(all_results, output_file=None, fixed_ylim_avg=None, fixed_ylim_minmax=None):
    """Create comparison plot of all modes.

    Args:
        all_results: List of result dictionaries
        output_file: Output filename (None = show interactively)
        fixed_ylim_avg: Tuple of (ymin, ymax) for average latency plot (left), or None for auto
        fixed_ylim_minmax: Tuple of (ymin, ymax) for min/max plot (right), or None for auto
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Define colors and markers for each mode
    mode_styles = {
        1: {'color': '#1f77b4', 'marker': 'o', 'label': 'Device-Initiated (xio)'},
        2: {'color': '#ff7f0e', 'marker': 's', 'label': 'Prepopulated-device-triggered (xio)'},
        3: {'color': '#2ca02c', 'marker': '^', 'label': 'Reverse Offload (xio)'},
        4: {'color': '#d62728', 'marker': 'v', 'label': 'Reverse Offload (HIP)'},
        5: {'color': '#9467bd', 'marker': 'D', 'label': 'PDT with chained zero-CU response (xio)'},
        6: {'color': '#8c564b', 'marker': 'h', 'label': 'Prepopulated-device-triggered (HIP)'},
    }

    # Plot 1: Average latency vs transfer size
    for result_data in all_results:
        mode = result_data['mode']
        style = mode_styles.get(mode, {'color': 'gray', 'marker': 'x', 'label': f'Mode {mode}'})

        sizes = []
        avg_latencies = []

        for entry in result_data['results']:
            sizes.append(entry['transfer_size_bytes'])
            avg_latencies.append(entry['avg_latency_us'])

        ax1.plot(sizes, avg_latencies,
                marker=style['marker'],
                color=style['color'],
                linewidth=2,
                markersize=8,
                label=style['label'])

    ax1.set_xlabel('Transfer Size (bytes)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Latency (μs)', fontsize=12, fontweight='bold')
    ax1.set_title('SDMA-EP Ping-Pong Latency: Average', fontsize=14, fontweight='bold')
    ax1.set_xscale('log', base=2)
    if fixed_ylim_avg:
        ax1.set_ylim(fixed_ylim_avg)
    else:
        ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=10)

    # Format x-axis labels
    ax1.set_xticks(sizes)
    ax1.set_xticklabels([f'{s}B' if s < 1024 else f'{s//1024}KB' for s in sizes])
    # Set explicit x-axis limits to prevent auto-scaling
    if sizes:
        ax1.set_xlim(sizes[0] * 0.8, sizes[-1] * 1.2)

    # Plot 2: Min/Max latency range
    x_positions = np.arange(len(sizes))
    bar_width = 0.15

    # Fixed offsets per mode for consistent overlay positioning
    mode_offsets = {1: -2.5, 2: -1.5, 6: -0.5, 3: 0.5, 4: 1.5, 5: 2.5}

    for result_data in all_results:
        mode = result_data['mode']
        style = mode_styles.get(mode, {'color': 'gray', 'marker': 'x', 'label': f'Mode {mode}'})

        min_latencies = []
        max_latencies = []
        avg_latencies = []

        for entry in result_data['results']:
            min_latencies.append(entry['min_latency_us'])
            max_latencies.append(entry['max_latency_us'])
            avg_latencies.append(entry['avg_latency_us'])

        # Use fixed offset based on mode number (not total count)
        offset = mode_offsets.get(mode, 0) * bar_width
        positions = x_positions + offset

        # Plot error bars showing min/max range
        errors = [
            [avg - min_val for avg, min_val in zip(avg_latencies, min_latencies)],
            [max_val - avg for avg, max_val in zip(avg_latencies, max_latencies)]
        ]

        ax2.errorbar(positions, avg_latencies,
                    yerr=errors,
                    fmt=style['marker'],
                    color=style['color'],
                    capsize=5,
                    capthick=2,
                    linewidth=2,
                    markersize=8,
                    label=style['label'])

    ax2.set_xlabel('Transfer Size (bytes)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Latency (μs)', fontsize=12, fontweight='bold')
    ax2.set_title('SDMA-EP Ping-Pong Latency: Min/Avg/Max', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([f'{s}B' if s < 1024 else f'{s//1024}KB' for s in sizes])
    # Set explicit x-axis limits to prevent auto-scaling
    if x_positions.size > 0:
        ax2.set_xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)
    if fixed_ylim_minmax:
        ax2.set_ylim(fixed_ylim_minmax)
    else:
        ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.legend(loc='upper left', fontsize=10)

    # Use fixed subplot positioning for consistent padding across overlay frames
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.1, wspace=0.25)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()


def plot_per_mode_breakdown(all_results, output_dir):
    """Create individual plots for each mode."""

    for result_data in all_results:
        mode = result_data['mode']

        fig, ax = plt.subplots(figsize=(10, 6))

        sizes = []
        min_latencies = []
        max_latencies = []
        avg_latencies = []

        for entry in result_data['results']:
            sizes.append(entry['transfer_size_bytes'])
            min_latencies.append(entry['min_latency_us'])
            max_latencies.append(entry['max_latency_us'])
            avg_latencies.append(entry['avg_latency_us'])

        x_positions = range(len(sizes))

        # Plot min/avg/max as separate lines
        ax.plot(x_positions, min_latencies, 'o-', label='Min', linewidth=2, markersize=8)
        ax.plot(x_positions, avg_latencies, 's-', label='Avg', linewidth=2, markersize=8)
        ax.plot(x_positions, max_latencies, '^-', label='Max', linewidth=2, markersize=8)

        # Shade the min-max range
        ax.fill_between(x_positions, min_latencies, max_latencies, alpha=0.2)

        ax.set_xlabel('Transfer Size (bytes)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latency (μs)', fontsize=12, fontweight='bold')
        ax.set_title(f'Mode {mode} Latency Breakdown', fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'{s}B' if s < 1024 else f'{s//1024}KB' for s in sizes])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=11)

        output_path = Path(output_dir) / f'mode{mode}_breakdown.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()


def print_summary_table(all_results):
    """Print a summary table of results."""

    print("\n" + "="*80)
    print("SDMA-EP Ping-Pong Latency Summary")
    print("="*80)

    for result_data in all_results:
        mode = result_data['mode']
        mode_names = {
            1: "Device-Initiated",
            2: "Device-Triggered (XIO)",
            3: "Reverse Offload (SDMA)",
            4: "Reverse Offload (hipMemcpy)",
            5: "Chained SDMA",
            6: "Device-Triggered (HIP)"
        }

        print(f"\nMode {mode}: {mode_names.get(mode, 'Unknown')}")
        print("-" * 80)
        print(f"{'Size':<10} {'Min (μs)':<12} {'Avg (μs)':<12} {'Max (μs)':<12} {'Avg (cycles)':<15}")
        print("-" * 80)

        for entry in result_data['results']:
            size = entry['transfer_size_bytes']
            size_str = f"{size}B" if size < 1024 else f"{size//1024}KB"
            print(f"{size_str:<10} {entry['min_latency_us']:<12.3f} "
                  f"{entry['avg_latency_us']:<12.3f} {entry['max_latency_us']:<12.3f} "
                  f"{entry['avg_cycles']:<15.2f}")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Plot SDMA-EP ping-pong benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('results_dir',
                       help='Directory containing JSON result files')
    parser.add_argument('--output', '-o',
                       help='Output file for comparison plot (default: show interactively)',
                       default=None)
    parser.add_argument('--modes', '-m',
                       type=int,
                       nargs='+',
                       help='Select specific modes to plot (e.g., -m 1 2 6). Default: all modes',
                       default=None)
    parser.add_argument('--ylim',
                       type=float,
                       nargs=2,
                       metavar=('MIN', 'MAX'),
                       help='Fixed y-axis limits for overlay consistency (e.g., --ylim 0 50)',
                       default=None)
    parser.add_argument('--auto-ylim',
                       action='store_true',
                       help='Compute y-axis limits from ALL modes (not just selected), for consistent overlays')
    parser.add_argument('--individual', '-i',
                       action='store_true',
                       help='Generate individual breakdown plots for each mode')
    parser.add_argument('--table', '-t',
                       action='store_true',
                       help='Print summary table')

    args = parser.parse_args()

    # Compute global y-limits if requested
    fixed_ylim_avg = None
    fixed_ylim_minmax = None

    if args.ylim:
        # User specified explicit limits - use for both plots
        fixed_ylim_avg = tuple(args.ylim)
        fixed_ylim_minmax = tuple(args.ylim)
    elif args.auto_ylim:
        # Load ALL results to compute global max for each plot type
        all_available = load_results(args.results_dir, selected_modes=None)
        global_max_avg = 0
        global_max_minmax = 0

        for result_data in all_available:
            for entry in result_data['results']:
                global_max_avg = max(global_max_avg, entry['avg_latency_us'])
                global_max_minmax = max(global_max_minmax, entry['max_latency_us'])

        fixed_ylim_avg = (0, global_max_avg * 1.1)  # Add 10% headroom
        fixed_ylim_minmax = (0, global_max_minmax * 1.1)
        print(f"Auto y-limits: Avg plot 0-{fixed_ylim_avg[1]:.2f} μs, Min/Max plot 0-{fixed_ylim_minmax[1]:.2f} μs")

    # Load results for plotting
    all_results = load_results(args.results_dir, selected_modes=args.modes)
    print(f"Loaded {len(all_results)} result file(s)")

    # Print summary table
    if args.table:
        print_summary_table(all_results)

    # Generate comparison plot
    plot_latency_comparison(all_results, args.output,
                          fixed_ylim_avg=fixed_ylim_avg,
                          fixed_ylim_minmax=fixed_ylim_minmax)

    # Generate individual plots if requested
    if args.individual:
        plot_per_mode_breakdown(all_results, args.results_dir)


if __name__ == '__main__':
    main()
