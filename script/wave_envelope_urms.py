#!/usr/bin/env python
"""
Wave Envelope and RMS Velocity Analysis for Wave Attenuation through Vegetation

This script analyzes wave envelope and root-mean-square velocity data from 
numerical simulations of wave propagation through sparse and dense vegetation patches.

Author: Sandy Herho <sandy.herho@email.ucr.edu>
Date: 08/04/2025
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
from pathlib import Path

def setup_directories():
    """Create output directories if they don't exist."""
    figs_dir = Path("../figs")
    analyses_dir = Path("../analyses")
    
    figs_dir.mkdir(parents=True, exist_ok=True)
    analyses_dir.mkdir(parents=True, exist_ok=True)
    
    return figs_dir, analyses_dir

def setup_plotting():
    """Configure matplotlib for publication-quality figures."""
    plt.style.use("fivethirtyeight")
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.dpi'] = 300

def load_data():
    """Load and extract data from NetCDF files."""
    print("Loading datasets...")
    ds_dense = xr.open_dataset("../raw_data/config_dense.nc")
    ds_sparse = xr.open_dataset("../raw_data/config_sparse.nc")
    
    # Extract arrays
    data = {
        'x': ds_dense["envelope"]["x"].to_numpy(),
        'x_face': ds_dense["u_rms"]["x_face"].to_numpy(),
        'wave_envelope_dense': ds_dense["envelope"].to_numpy(),
        'wave_envelope_sparse': ds_sparse["envelope"].to_numpy(),
        'urms_dense': ds_dense["u_rms"].to_numpy(),
        'urms_sparse': ds_sparse["u_rms"].to_numpy()
    }
    
    return data

def create_wave_envelope_figure(data, figs_dir):
    """Create wave envelope comparison figure."""
    print("Creating wave envelope figure...")
    
    # Parameters
    veg_start = 80
    veg_end = 120
    color_dense = '#1f77b4'
    color_sparse = '#ff7f0e'
    color_veg = '#2ca02c'
    alpha_veg = 0.15
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), sharex=True, sharey=True)
    
    # Top panel - Dense vegetation
    ax1.plot(data['x'], data['wave_envelope_dense'], color=color_dense, linewidth=2.5)
    ax1.axvspan(veg_start, veg_end, alpha=alpha_veg, color=color_veg)
    ax1.axvline(x=veg_start, color=color_veg, alpha=0.5, linestyle='--', linewidth=1)
    ax1.axvline(x=veg_end, color=color_veg, alpha=0.5, linestyle='--', linewidth=1)
    ax1.set_xlim([0, 200])
    ax1.set_ylim([0, 0.5])
    # Remove individual ylabel
    # ax1.set_ylabel(r'Wave Envelope $A_{env}(x)$ [m]')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top')
    
    # Bottom panel - Sparse vegetation
    ax2.plot(data['x'], data['wave_envelope_sparse'], color=color_sparse, linewidth=2.5)
    ax2.axvspan(veg_start, veg_end, alpha=alpha_veg, color=color_veg)
    ax2.axvline(x=veg_start, color=color_veg, alpha=0.5, linestyle='--', linewidth=1)
    ax2.axvline(x=veg_end, color=color_veg, alpha=0.5, linestyle='--', linewidth=1)
    ax2.set_xlabel(r'Distance $x$ [m]')
    # Remove individual ylabel
    # ax2.set_ylabel(r'Wave Envelope $A_{env}(x)$ [m]')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top')
    
    # Add single ylabel for entire figure
    fig.supylabel(r'Wave Envelope $A_{env}(x)$ [m]', fontsize=16)
    
    # Create custom legend elements
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color=color_dense, lw=2.5, label='Dense Vegetation'),
        Line2D([0], [0], color=color_sparse, lw=2.5, label='Sparse Vegetation'),
        Patch(facecolor=color_veg, alpha=alpha_veg, label='Vegetation Zone')
    ]
    
    # Add legend at the bottom
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02), frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, left=0.1)  # Adjust left margin for supylabel
    plt.savefig(figs_dir / 'wave_envelope_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_urms_figure(data, figs_dir):
    """Create RMS velocity comparison figure."""
    print("Creating RMS velocity figure...")
    
    # Parameters
    veg_start = 80
    veg_end = 120
    color_dense = '#1f77b4'
    color_sparse = '#ff7f0e'
    color_veg = '#2ca02c'
    alpha_veg = 0.15
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), sharex=True, sharey=True)
    
    # Top panel - Dense vegetation
    ax1.plot(data['x_face'], data['urms_dense'], color=color_dense, linewidth=2.5)
    ax1.axvspan(veg_start, veg_end, alpha=alpha_veg, color=color_veg)
    ax1.axvline(x=veg_start, color=color_veg, alpha=0.5, linestyle='--', linewidth=1)
    ax1.axvline(x=veg_end, color=color_veg, alpha=0.5, linestyle='--', linewidth=1)
    ax1.set_xlim([0, 200])
    ax1.set_ylim([0, 0.8])  # Changed from 0.5 to 0.7
    # Remove individual ylabel
    # ax1.set_ylabel(r'RMS Velocity $u_{rms}(x)$ [m/s]')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top')
    
    # Bottom panel - Sparse vegetation
    ax2.plot(data['x_face'], data['urms_sparse'], color=color_sparse, linewidth=2.5)
    ax2.axvspan(veg_start, veg_end, alpha=alpha_veg, color=color_veg)
    ax2.axvline(x=veg_start, color=color_veg, alpha=0.5, linestyle='--', linewidth=1)
    ax2.axvline(x=veg_end, color=color_veg, alpha=0.5, linestyle='--', linewidth=1)
    ax2.set_xlabel(r'Distance $x$ [m]')  # Changed to show x_face
    # Remove individual ylabel
    # ax2.set_ylabel(r'RMS Velocity $u_{rms}(x)$ [m/s]')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top')
    
    # Add single ylabel for entire figure
    fig.supylabel(r'RMS Velocity $u_{rms}(x)$ [m/s]', fontsize=16)
    
    # Create custom legend elements
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color=color_dense, lw=2.5, label='Dense Vegetation'),
        Line2D([0], [0], color=color_sparse, lw=2.5, label='Sparse Vegetation'),
        Patch(facecolor=color_veg, alpha=alpha_veg, label='Vegetation Zone')
    ]
    
    # Add legend at the bottom
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02), frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, left=0.1)  # Adjust left margin for supylabel
    plt.savefig(figs_dir / 'urms_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def perform_analysis(data):
    """Perform quantitative analysis of wave attenuation."""
    print("Performing analysis...")
    
    # Vegetation boundaries
    veg_start = 80
    veg_end = 120
    
    # Define spatial regions
    x = data['x']
    x_face = data['x_face']
    
    idx_before_veg = x < veg_start
    idx_in_veg = (x >= veg_start) & (x <= veg_end)
    idx_after_veg = x > veg_end
    
    idx_face_before_veg = x_face < veg_start
    idx_face_in_veg = (x_face >= veg_start) & (x_face <= veg_end)
    idx_face_after_veg = x_face > veg_end
    
    # Calculate statistics
    results = {}
    
    # Wave envelope - Dense vegetation
    results['env_dense_before'] = data['wave_envelope_dense'][idx_before_veg].mean()
    results['env_dense_in'] = data['wave_envelope_dense'][idx_in_veg].mean()
    results['env_dense_after'] = data['wave_envelope_dense'][idx_after_veg].mean()
    results['env_dense_max'] = data['wave_envelope_dense'].max()
    results['env_dense_min'] = data['wave_envelope_dense'].min()
    results['env_dense_max_loc'] = x[data['wave_envelope_dense'].argmax()]
    results['env_dense_min_loc'] = x[data['wave_envelope_dense'].argmin()]
    
    # Wave envelope - Sparse vegetation
    results['env_sparse_before'] = data['wave_envelope_sparse'][idx_before_veg].mean()
    results['env_sparse_in'] = data['wave_envelope_sparse'][idx_in_veg].mean()
    results['env_sparse_after'] = data['wave_envelope_sparse'][idx_after_veg].mean()
    results['env_sparse_max'] = data['wave_envelope_sparse'].max()
    results['env_sparse_min'] = data['wave_envelope_sparse'].min()
    results['env_sparse_max_loc'] = x[data['wave_envelope_sparse'].argmax()]
    results['env_sparse_min_loc'] = x[data['wave_envelope_sparse'].argmin()]
    
    # RMS velocity - Dense vegetation
    results['urms_dense_before'] = data['urms_dense'][idx_face_before_veg].mean()
    results['urms_dense_in'] = data['urms_dense'][idx_face_in_veg].mean()
    results['urms_dense_after'] = data['urms_dense'][idx_face_after_veg].mean()
    results['urms_dense_max'] = data['urms_dense'].max()
    results['urms_dense_min'] = data['urms_dense'].min()
    results['urms_dense_max_loc'] = x_face[data['urms_dense'].argmax()]
    results['urms_dense_min_loc'] = x_face[data['urms_dense'].argmin()]
    
    # RMS velocity - Sparse vegetation
    results['urms_sparse_before'] = data['urms_sparse'][idx_face_before_veg].mean()
    results['urms_sparse_in'] = data['urms_sparse'][idx_face_in_veg].mean()
    results['urms_sparse_after'] = data['urms_sparse'][idx_face_after_veg].mean()
    results['urms_sparse_max'] = data['urms_sparse'].max()
    results['urms_sparse_min'] = data['urms_sparse'].min()
    results['urms_sparse_max_loc'] = x_face[data['urms_sparse'].argmax()]
    results['urms_sparse_min_loc'] = x_face[data['urms_sparse'].argmin()]
    
    # Calculate attenuation percentages
    results['env_dense_atten'] = (results['env_dense_before'] - results['env_dense_after']) / results['env_dense_before'] * 100
    results['env_sparse_atten'] = (results['env_sparse_before'] - results['env_sparse_after']) / results['env_sparse_before'] * 100
    results['urms_dense_atten'] = (results['urms_dense_before'] - results['urms_dense_after']) / results['urms_dense_before'] * 100
    results['urms_sparse_atten'] = (results['urms_sparse_before'] - results['urms_sparse_after']) / results['urms_sparse_before'] * 100
    
    return results

def write_analysis(results, analyses_dir):
    """Write analysis results to text file."""
    
    analysis_text = f"""Wave Attenuation through Vegetation: Analysis Results
====================================================
Author: Sandy Herho <sandy.herho@email.ucr.edu>
Date: 08/04/2025

SIMULATION PARAMETERS
---------------------
Domain length: 200 m
Water depth: 2 m
Wave amplitude: 0.3 m
Wave period: 10 s
Vegetation zone: 80-120 m
Dense vegetation drag coefficient: 1.4 s^-1
Sparse vegetation drag coefficient: 0.14 s^-1

WAVE ENVELOPE ANALYSIS
----------------------

Dense Vegetation (cD = 1.4 s^-1):
----------------------------------
Maximum amplitude: {results['env_dense_max']:.4f} m at x = {results['env_dense_max_loc']:.1f} m
Minimum amplitude: {results['env_dense_min']:.4f} m at x = {results['env_dense_min_loc']:.1f} m
Mean amplitude before vegetation (x < 80 m): {results['env_dense_before']:.4f} m
Mean amplitude within vegetation (80 ≤ x ≤ 120 m): {results['env_dense_in']:.4f} m
Mean amplitude after vegetation (x > 120 m): {results['env_dense_after']:.4f} m
Amplitude reduction: {results['env_dense_atten']:.1f}%

Sparse Vegetation (cD = 0.14 s^-1):
-----------------------------------
Maximum amplitude: {results['env_sparse_max']:.4f} m at x = {results['env_sparse_max_loc']:.1f} m
Minimum amplitude: {results['env_sparse_min']:.4f} m at x = {results['env_sparse_min_loc']:.1f} m
Mean amplitude before vegetation (x < 80 m): {results['env_sparse_before']:.4f} m
Mean amplitude within vegetation (80 ≤ x ≤ 120 m): {results['env_sparse_in']:.4f} m
Mean amplitude after vegetation (x > 120 m): {results['env_sparse_after']:.4f} m
Amplitude reduction: {results['env_sparse_atten']:.1f}%

Amplitude Reduction Ratio (Dense/Sparse): {results['env_dense_atten']/results['env_sparse_atten']:.2f}

RMS VELOCITY ANALYSIS
---------------------

Dense Vegetation (cD = 1.4 s^-1):
----------------------------------
Maximum RMS velocity: {results['urms_dense_max']:.4f} m/s at x_face = {results['urms_dense_max_loc']:.1f} m
Minimum RMS velocity: {results['urms_dense_min']:.4f} m/s at x_face = {results['urms_dense_min_loc']:.1f} m
Mean RMS velocity before vegetation (x_face < 80 m): {results['urms_dense_before']:.4f} m/s
Mean RMS velocity within vegetation (80 ≤ x_face ≤ 120 m): {results['urms_dense_in']:.4f} m/s
Mean RMS velocity after vegetation (x_face > 120 m): {results['urms_dense_after']:.4f} m/s
Velocity reduction: {results['urms_dense_atten']:.1f}%

Sparse Vegetation (cD = 0.14 s^-1):
-----------------------------------
Maximum RMS velocity: {results['urms_sparse_max']:.4f} m/s at x_face = {results['urms_sparse_max_loc']:.1f} m
Minimum RMS velocity: {results['urms_sparse_min']:.4f} m/s at x_face = {results['urms_sparse_min_loc']:.1f} m
Mean RMS velocity before vegetation (x_face < 80 m): {results['urms_sparse_before']:.4f} m/s
Mean RMS velocity within vegetation (80 ≤ x_face ≤ 120 m): {results['urms_sparse_in']:.4f} m/s
Mean RMS velocity after vegetation (x_face > 120 m): {results['urms_sparse_after']:.4f} m/s
Velocity reduction: {results['urms_sparse_atten']:.1f}%

Velocity Reduction Ratio (Dense/Sparse): {results['urms_dense_atten']/results['urms_sparse_atten']:.2f}

TRANSMISSION COEFFICIENTS
-------------------------
Wave Height Transmission (Kt = H_out/H_in):
Dense vegetation: {results['env_dense_after']/results['env_dense_before']:.3f}
Sparse vegetation: {results['env_sparse_after']/results['env_sparse_before']:.3f}

Energy Transmission (Kt,E = E_out/E_in):
Dense vegetation: {(results['env_dense_after']/results['env_dense_before'])**2:.3f}
Sparse vegetation: {(results['env_sparse_after']/results['env_sparse_before'])**2:.3f}
"""
    
    # Save analysis
    with open(analyses_dir / 'wave_envelope_urms_analysis.txt', 'w') as f:
        f.write(analysis_text)
    
    return analysis_text

def main():
    """Main execution function."""
    # Setup
    figs_dir, analyses_dir = setup_directories()
    setup_plotting()
    
    # Load data
    data = load_data()
    
    # Create figures
    create_wave_envelope_figure(data, figs_dir)
    create_urms_figure(data, figs_dir)
    
    # Perform analysis
    results = perform_analysis(data)
    
    # Write results
    write_analysis(results, analyses_dir)
    
    # Print completion message
    print(f"\nAnalysis complete!")
    print(f"Figures saved to: {figs_dir}")
    print(f"Analysis saved to: {analyses_dir / 'wave_envelope_urms_analysis.txt'}")
    
    # Print summary
    print("\n=== SUMMARY ===")
    print(f"\nWave Envelope Reduction:")
    print(f"  Dense vegetation: {results['env_dense_atten']:.1f}%")
    print(f"  Sparse vegetation: {results['env_sparse_atten']:.1f}%")
    print(f"\nRMS Velocity Reduction:")
    print(f"  Dense vegetation: {results['urms_dense_atten']:.1f}%")
    print(f"  Sparse vegetation: {results['urms_sparse_atten']:.1f}%")

if __name__ == "__main__":
    main()
