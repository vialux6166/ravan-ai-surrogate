#!/usr/bin/env python3
"""
Analyze benchmark results and generate publication-ready visualizations
"""

import json
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_results(path):
    """Load benchmark results from JSON file"""
    with open(path, "r") as f:
        data = json.load(f)
    
    # Check if we have the new format
    if "summary" in data and "results" in data:
        df = pd.DataFrame(data["results"])
        summary = data["summary"]
    else:
        # Original format - convert to new format
        df = None
        summary = data
        if "inverse_design" in data:
            inv_data = data["inverse_design"]
            summary = {
                "runs": inv_data.get("runs", 0),
                "success_rate": inv_data.get("success_rate", 0),
                "mean_error": inv_data.get("mean_error", 0),
                "mean_time_ms": inv_data.get("mean_time_ms", 0),
                "threshold_error": inv_data.get("threshold_error", 0.01),
                "method": "Basin-Hopping + L-BFGS-B Hybrid"
            }
    
    return df, summary

def plot_error_histogram(df, outdir):
    """Plot error distribution histogram"""
    plt.figure(figsize=(8, 6))
    
    if df is not None and "error" in df.columns:
        plt.hist(df["error"], bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel("Error", fontsize=12, fontweight='bold')
        plt.ylabel("Frequency", fontsize=12, fontweight='bold')
        plt.title("Inverse Design Error Distribution", fontsize=14, fontweight='bold')
    else:
        plt.text(0.5, 0.5, "No detailed error data available", 
                ha='center', va='center', fontsize=12)
        plt.xlabel("Error", fontsize=12, fontweight='bold')
        plt.ylabel("Frequency", fontsize=12, fontweight='bold')
        plt.title("Inverse Design Error Distribution", fontsize=14, fontweight='bold')
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/error_histogram.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_vs_error(df, outdir):
    """Plot time vs error scatter"""
    plt.figure(figsize=(8, 6))
    
    if df is not None and "time_ms" in df.columns and "error" in df.columns:
        plt.scatter(df["time_ms"], df["error"], alpha=0.6, s=50)
        plt.xlabel("Time (ms)", fontsize=12, fontweight='bold')
        plt.ylabel("Error", fontsize=12, fontweight='bold')
        plt.title("Time vs Error Trade-off", fontsize=14, fontweight='bold')
    else:
        plt.text(0.5, 0.5, "No detailed time/error data available", 
                ha='center', va='center', fontsize=12)
        plt.xlabel("Time (ms)", fontsize=12, fontweight='bold')
        plt.ylabel("Error", fontsize=12, fontweight='bold')
        plt.title("Time vs Error Trade-off", fontsize=14, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/time_vs_error.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_success_rate(summary, outdir):
    """Plot success rate bar chart"""
    plt.figure(figsize=(8, 6))
    
    success_rate = summary.get("success_rate", 0) * 100
    fail_rate = 100 - success_rate
    
    plt.bar(["Success", "Failed"], [success_rate, fail_rate], 
           color=['#6A994E', '#D62828'], edgecolor='black', linewidth=2)
    plt.ylabel("Percentage (%)", fontsize=12, fontweight='bold')
    plt.title(f"Inverse Design Success Rate\n({summary.get('runs', 0)} total runs)", 
             fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    plt.text(0, success_rate + 2, f"{success_rate:.1f}%", 
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.text(1, fail_rate + 2, f"{fail_rate:.1f}%", 
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{outdir}/success_rate.png", dpi=300, bbox_inches='tight')
    plt.close()

def main(input_file, outdir):
    """Main analysis function"""
    print(f"Loading results from {input_file}...")
    df, summary = load_results(input_file)
    
    # Create output directory
    os.makedirs(outdir, exist_ok=True)
    
    # Generate plots
    print(f"Generating plots in {outdir}...")
    plot_error_histogram(df, outdir)
    plot_time_vs_error(df, outdir)
    plot_success_rate(summary, outdir)
    
    # Save summary CSV if we have detailed data
    if df is not None:
        df.to_csv(f"{outdir}/benchmarks_summary.csv", index=False)
        print(f"[OK] Saved detailed CSV to {outdir}/benchmarks_summary.csv")
    else:
        # Create a simple summary CSV
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(f"{outdir}/benchmarks_summary.csv", index=False)
        print(f"[OK] Saved summary CSV to {outdir}/benchmarks_summary.csv")
    
    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK ANALYSIS COMPLETE")
    print("="*70)
    print(f"Output directory: {Path(outdir).absolute()}")
    print("\nSummary:")
    print(f"  Method: {summary.get('method', 'Unknown')}")
    print(f"  Runs: {summary.get('runs', 0)}")
    print(f"  Success Rate: {summary.get('success_rate', 0)*100:.1f}%")
    print(f"  Mean Error: {summary.get('mean_error', 0):.6f}")
    print(f"  Mean Time: {summary.get('mean_time_ms', 0):.2f} ms")
    print("="*70)
    print(f"\nFigures saved to: {outdir}/")
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze benchmark results and generate visualizations")
    parser.add_argument("--input", default="benchmark_results.json", 
                       help="Input JSON file with benchmark results")
    parser.add_argument("--outdir", default="reports/figures", 
                       help="Output directory for figures and CSV")
    args = parser.parse_args()
    
    main(args.input, args.outdir)

