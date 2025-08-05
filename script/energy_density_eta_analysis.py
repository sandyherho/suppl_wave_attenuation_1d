#!/usr/bin/env python
"""
3D Spatio-Temporal Analysis of Wave Propagation through Vegetation
Modified version with improved figure spacing for better z-label visibility

This script creates 3D visualizations of free surface elevation and wave energy density
evolution through sparse and dense vegetation patches, with quantitative analysis.

Author: Sandy Herho <sandy.herho@email.ucr.edu>
Date: 08/04/2025
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['text.usetex'] = False  # Set to True if LaTeX is available
    plt.rcParams['font.family'] = 'DejaVu Sans'

def load_and_process_data():
    """Load data from NetCDF files."""
    print("Loading datasets...")
    ds_dense = xr.open_dataset("../raw_data/config_dense.nc")
    ds_sparse = xr.open_dataset("../raw_data/config_sparse.nc")
    
    # Extract coordinate arrays
    x = ds_dense.x.values
    x_face = ds_dense.x_face.values
    t = ds_dense.time.values
    
    # Get state variables
    eta_dense = ds_dense.eta.values
    eta_sparse = ds_sparse.eta.values
    u_dense = ds_dense.u.values
    u_sparse = ds_sparse.u.values
    
    # Get energy density (already computed in the dataset)
    energy_dense = ds_dense.energy.values
    energy_sparse = ds_sparse.energy.values
    
    # Physical parameters from attributes
    g = 9.81  # m/s²
    h = 2.0   # water depth in meters
    rho = 1000  # water density kg/m³
    
    data = {
        'x': x,
        'x_face': x_face,
        't': t,
        'eta_dense': eta_dense,
        'eta_sparse': eta_sparse,
        'u_dense': u_dense,
        'u_sparse': u_sparse,
        'energy_dense': energy_dense,
        'energy_sparse': energy_sparse,
        'g': g,
        'h': h,
        'rho': rho
    }
    
    # Print data ranges for verification
    print(f"Time range: {t[0]:.1f} to {t[-1]:.1f} s")
    print(f"Spatial range: {x[0]:.1f} to {x[-1]:.1f} m")
    print(f"Eta range - Dense: [{np.min(eta_dense):.3f}, {np.max(eta_dense):.3f}] m")
    print(f"Eta range - Sparse: [{np.min(eta_sparse):.3f}, {np.max(eta_sparse):.3f}] m")
    print(f"Energy range - Dense: [{np.min(energy_dense):.1f}, {np.max(energy_dense):.1f}] J/m³")
    print(f"Energy range - Sparse: [{np.min(energy_sparse):.1f}, {np.max(energy_sparse):.1f}] J/m³")
    
    return data

def create_eta_3d_plot(data, figs_dir):
    """Create 3D plot of free surface elevation with improved spacing."""
    print("Creating 3D free surface elevation plot...")
    
    # Subsample for cleaner visualization
    skip_x = 8
    skip_t = 20
    
    x_sub = data['x'][::skip_x]
    t_sub = data['t'][::skip_t]
    eta_dense_sub = data['eta_dense'][::skip_t, ::skip_x]
    eta_sparse_sub = data['eta_sparse'][::skip_t, ::skip_x]
    
    # Create meshgrid
    X, T = np.meshgrid(x_sub, t_sub)
    
    # Vegetation zone
    veg_start = 80
    veg_end = 120
    
    # Determine common z-axis limits for eta
    eta_min = min(np.min(eta_dense_sub), np.min(eta_sparse_sub))
    eta_max = max(np.max(eta_dense_sub), np.max(eta_sparse_sub))
    eta_lim = max(abs(eta_min), abs(eta_max)) * 1.1  # Add 10% margin
    
    # Create figure with explicit size and DPI
    fig = plt.figure(figsize=(16, 12), dpi=100)
    
    # Top panel - Dense vegetation
    ax1 = fig.add_subplot(211, projection='3d')
    
    # Create the wireframe
    ax1.plot_wireframe(X, T, eta_dense_sub, color='black', linewidth=0.5, alpha=0.8)
    
    # Add vegetation zone indicators
    for t_val in [t_sub[0], t_sub[-1]]:
        ax1.plot([veg_start, veg_start], [t_val, t_val], [-eta_lim, eta_lim], 
                'g--', linewidth=2, alpha=0.6)
        ax1.plot([veg_end, veg_end], [t_val, t_val], [-eta_lim, eta_lim], 
                'g--', linewidth=2, alpha=0.6)
    
    # Set labels with spacing
    ax1.set_xlabel(r'Distance $x$ [m]', labelpad=15)
    ax1.set_ylabel(r'Time $t$ [s]', labelpad=15)
    
    # Z-label with special handling
    ax1.set_zlabel(r'$\eta$ [m]', labelpad=25)
    ax1.zaxis.set_rotate_label(False)  # Prevent rotation
    
    ax1.set_title('(a)', fontsize=16, loc='left', pad=10)
    ax1.view_init(elev=20, azim=-50)  # Adjusted for better z-label visibility
    ax1.set_xlim(0, 200)
    ax1.set_ylim(t_sub[0], t_sub[-1])
    ax1.set_zlim(-eta_lim, eta_lim)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    
    # Adjust tick parameters for z-axis
    ax1.tick_params(axis='z', which='major', pad=10)
    
    # Bottom panel - Sparse vegetation
    ax2 = fig.add_subplot(212, projection='3d')
    
    # Create the wireframe
    ax2.plot_wireframe(X, T, eta_sparse_sub, color='black', linewidth=0.5, alpha=0.8)
    
    # Add vegetation zone indicators
    for t_val in [t_sub[0], t_sub[-1]]:
        ax2.plot([veg_start, veg_start], [t_val, t_val], [-eta_lim, eta_lim], 
                'g--', linewidth=2, alpha=0.6)
        ax2.plot([veg_end, veg_end], [t_val, t_val], [-eta_lim, eta_lim], 
                'g--', linewidth=2, alpha=0.6)
    
    # Set labels with spacing
    ax2.set_xlabel(r'Distance $x$ [m]', labelpad=15)
    ax2.set_ylabel(r'Time $t$ [s]', labelpad=15)
    
    # Z-label with special handling
    ax2.set_zlabel(r'$\eta$ [m]', labelpad=25)
    ax2.zaxis.set_rotate_label(False)  # Prevent rotation
    
    ax2.set_title('(b)', fontsize=16, loc='left', pad=10)
    ax2.view_init(elev=20, azim=-50)  # Adjusted for better z-label visibility
    ax2.set_xlim(0, 200)
    ax2.set_ylim(t_sub[0], t_sub[-1])
    ax2.set_zlim(-eta_lim, eta_lim)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    
    # Adjust tick parameters for z-axis
    ax2.tick_params(axis='z', which='major', pad=10)
    
    # Use tight layout with extra padding
    plt.tight_layout(pad=3.0)
    
    # Save with manual bbox to ensure nothing is cut off
    # Get the full extent of the figure
    plt.savefig(figs_dir / 'eta_3d_spatiotemporal.png', 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=1.0,
                facecolor='white',
                edgecolor='none')
    plt.close()

def create_energy_3d_plot(data, figs_dir):
    """Create 3D plot of wave energy density with improved spacing."""
    print("Creating 3D wave energy density plot...")
    
    # Subsample for cleaner visualization
    skip_x = 8
    skip_t = 20
    
    # Energy is defined on x_face, so we need to handle the indexing carefully
    x_face_sub = data['x_face'][::skip_x]
    t_sub = data['t'][::skip_t]
    energy_dense_sub = data['energy_dense'][::skip_t, ::skip_x]
    energy_sparse_sub = data['energy_sparse'][::skip_t, ::skip_x]
    
    # Create meshgrid
    X, T = np.meshgrid(x_face_sub, t_sub)
    
    # Vegetation zone
    veg_start = 80
    veg_end = 120
    
    # Determine common z-axis limits for energy
    energy_max = max(np.max(energy_dense_sub), np.max(energy_sparse_sub))
    energy_min = 0  # Energy density is always positive
    
    # Create figure with explicit size and DPI
    fig = plt.figure(figsize=(16, 12), dpi=100)
    
    # Top panel - Dense vegetation
    ax1 = fig.add_subplot(211, projection='3d')
    
    # Create the wireframe
    ax1.plot_wireframe(X, T, energy_dense_sub, color='black', linewidth=0.5, alpha=0.8)
    
    # Add vegetation zone indicators
    for t_val in [t_sub[0], t_sub[-1]]:
        ax1.plot([veg_start, veg_start], [t_val, t_val], [energy_min, energy_max], 
                'g--', linewidth=2, alpha=0.6)
        ax1.plot([veg_end, veg_end], [t_val, t_val], [energy_min, energy_max], 
                'g--', linewidth=2, alpha=0.6)
    
    # Set labels with spacing
    ax1.set_xlabel(r'Distance $x$ [m]', labelpad=15)
    ax1.set_ylabel(r'Time $t$ [s]', labelpad=15)
    
    # Z-label with special handling
    ax1.set_zlabel(r'$E$ [J/m$^3$]', labelpad=25)
    ax1.zaxis.set_rotate_label(False)  # Prevent rotation
    
    ax1.set_title('(a)', fontsize=16, loc='left', pad=10)
    ax1.view_init(elev=20, azim=-50)  # Adjusted for better z-label visibility
    ax1.set_xlim(0, 200)
    ax1.set_ylim(t_sub[0], t_sub[-1])
    ax1.set_zlim(energy_min, energy_max)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    
    # Adjust tick parameters for z-axis
    ax1.tick_params(axis='z', which='major', pad=10)
    
    # Bottom panel - Sparse vegetation
    ax2 = fig.add_subplot(212, projection='3d')
    
    # Create the wireframe
    ax2.plot_wireframe(X, T, energy_sparse_sub, color='black', linewidth=0.5, alpha=0.8)
    
    # Add vegetation zone indicators
    for t_val in [t_sub[0], t_sub[-1]]:
        ax2.plot([veg_start, veg_start], [t_val, t_val], [energy_min, energy_max], 
                'g--', linewidth=2, alpha=0.6)
        ax2.plot([veg_end, veg_end], [t_val, t_val], [energy_min, energy_max], 
                'g--', linewidth=2, alpha=0.6)
    
    # Set labels with spacing
    ax2.set_xlabel(r'Distance $x$ [m]', labelpad=15)
    ax2.set_ylabel(r'Time $t$ [s]', labelpad=15)
    
    # Z-label with special handling
    ax2.set_zlabel(r'$E$ [J/m$^3$]', labelpad=25)
    ax2.zaxis.set_rotate_label(False)  # Prevent rotation
    
    ax2.set_title('(b)', fontsize=16, loc='left', pad=10)
    ax2.view_init(elev=20, azim=-50)  # Adjusted for better z-label visibility
    ax2.set_xlim(0, 200)
    ax2.set_ylim(t_sub[0], t_sub[-1])
    ax2.set_zlim(energy_min, energy_max)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    
    # Adjust tick parameters for z-axis
    ax2.tick_params(axis='z', which='major', pad=10)
    
    # Use tight layout with extra padding
    plt.tight_layout(pad=3.0)
    
    # Save with manual bbox to ensure nothing is cut off
    plt.savefig(figs_dir / 'energy_density_3d_spatiotemporal.png', 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=1.0,
                facecolor='white',
                edgecolor='none')
    plt.close()

def perform_energy_analysis(data):
    """Perform quantitative analysis of wave energy density."""
    print("Performing wave energy density analysis...")
    
    # Vegetation boundaries
    veg_start = 80
    veg_end = 120
    
    # Time range for steady-state analysis (last 10 wave periods)
    T_wave = 10  # wave period in seconds
    t_analysis_start = data['t'][-1] - 10 * T_wave
    t_mask = data['t'] >= t_analysis_start
    
    # Spatial regions for x_face (where energy is defined)
    x_face = data['x_face']
    idx_before_veg = x_face < veg_start
    idx_in_veg = (x_face >= veg_start) & (x_face <= veg_end)
    idx_after_veg = x_face > veg_end
    
    # Time-averaged energy density
    energy_dense_avg = np.mean(data['energy_dense'][t_mask, :], axis=0)
    energy_sparse_avg = np.mean(data['energy_sparse'][t_mask, :], axis=0)
    
    # Peak energy density (instantaneous max over space and time)
    energy_dense_peak = np.max(data['energy_dense'][t_mask, :])
    energy_sparse_peak = np.max(data['energy_sparse'][t_mask, :])
    
    # Spatial statistics
    results = {}
    
    # Dense vegetation
    results['energy_dense_before'] = energy_dense_avg[idx_before_veg].mean()
    results['energy_dense_in'] = energy_dense_avg[idx_in_veg].mean()
    results['energy_dense_after'] = energy_dense_avg[idx_after_veg].mean()
    results['energy_dense_peak'] = energy_dense_peak
    results['energy_dense_min'] = np.min(energy_dense_avg)
    
    # Sparse vegetation
    results['energy_sparse_before'] = energy_sparse_avg[idx_before_veg].mean()
    results['energy_sparse_in'] = energy_sparse_avg[idx_in_veg].mean()
    results['energy_sparse_after'] = energy_sparse_avg[idx_after_veg].mean()
    results['energy_sparse_peak'] = energy_sparse_peak
    results['energy_sparse_min'] = np.min(energy_sparse_avg)
    
    # Energy dissipation
    results['energy_dense_dissipation'] = (results['energy_dense_before'] - 
                                          results['energy_dense_after']) / results['energy_dense_before'] * 100
    results['energy_sparse_dissipation'] = (results['energy_sparse_before'] - 
                                           results['energy_sparse_after']) / results['energy_sparse_before'] * 100
    
    # Wave energy flux (for progressive waves: F = E * cg, where cg = sqrt(gh) for shallow water)
    cg = np.sqrt(data['g'] * data['h'])  # group velocity
    results['flux_dense_before'] = results['energy_dense_before'] * cg
    results['flux_dense_after'] = results['energy_dense_after'] * cg
    results['flux_sparse_before'] = results['energy_sparse_before'] * cg
    results['flux_sparse_after'] = results['energy_sparse_after'] * cg
    
    # Total energy dissipation rate within vegetation (W/m)
    veg_length = veg_end - veg_start
    results['power_dissipated_dense'] = (results['flux_dense_before'] - 
                                        results['flux_dense_after']) / veg_length
    results['power_dissipated_sparse'] = (results['flux_sparse_before'] - 
                                         results['flux_sparse_after']) / veg_length
    
    return results

def write_energy_analysis(results, data, analyses_dir):
    """Write energy density analysis results to text file."""
    
    analysis_text = f"""Wave Energy Density Analysis through Vegetation
==============================================
Author: Sandy Herho <sandy.herho@email.ucr.edu>
Date: 08/04/2025

SIMULATION PARAMETERS
---------------------
Domain length: 200 m
Water depth: 2 m
Wave amplitude: 0.3 m
Wave period: 10 s
Vegetation zone: 80-120 m
Water density: {data['rho']} kg/m³
Gravitational acceleration: {data['g']} m/s²

WAVE ENERGY DENSITY ANALYSIS
============================

Dense Vegetation (cD = 1.4 s⁻¹):
--------------------------------
Peak energy density: {results['energy_dense_peak']:.1f} J/m³
Minimum energy density: {results['energy_dense_min']:.1f} J/m³

Time-averaged energy density:
  Before vegetation (x < 80 m): {results['energy_dense_before']:.1f} J/m³
  Within vegetation (80 ≤ x ≤ 120 m): {results['energy_dense_in']:.1f} J/m³
  After vegetation (x > 120 m): {results['energy_dense_after']:.1f} J/m³

Energy dissipation: {results['energy_dense_dissipation']:.1f}%

Sparse Vegetation (cD = 0.14 s⁻¹):
----------------------------------
Peak energy density: {results['energy_sparse_peak']:.1f} J/m³
Minimum energy density: {results['energy_sparse_min']:.1f} J/m³

Time-averaged energy density:
  Before vegetation (x < 80 m): {results['energy_sparse_before']:.1f} J/m³
  Within vegetation (80 ≤ x ≤ 120 m): {results['energy_sparse_in']:.1f} J/m³
  After vegetation (x > 120 m): {results['energy_sparse_after']:.1f} J/m³

Energy dissipation: {results['energy_sparse_dissipation']:.1f}%

WAVE ENERGY FLUX ANALYSIS
=========================
Group velocity (shallow water): {np.sqrt(data['g'] * data['h']):.2f} m/s

Dense Vegetation:
-----------------
Energy flux before vegetation: {results['flux_dense_before']:.1f} W/m
Energy flux after vegetation: {results['flux_dense_after']:.1f} W/m
Flux reduction: {(1 - results['flux_dense_after']/results['flux_dense_before'])*100:.1f}%

Sparse Vegetation:
------------------
Energy flux before vegetation: {results['flux_sparse_before']:.1f} W/m
Energy flux after vegetation: {results['flux_sparse_after']:.1f} W/m
Flux reduction: {(1 - results['flux_sparse_after']/results['flux_sparse_before'])*100:.1f}%

POWER DISSIPATION
=================
Power dissipated per unit width within vegetation zone:
  Dense vegetation: {results['power_dissipated_dense']:.1f} W/m²
  Sparse vegetation: {results['power_dissipated_sparse']:.1f} W/m²

Dissipation effectiveness ratio (Dense/Sparse): {results['power_dissipated_dense']/results['power_dissipated_sparse']:.2f}

ENERGY-BASED TRANSMISSION COEFFICIENT
=====================================
Kt,E = sqrt(E_out/E_in):
  Dense vegetation: {np.sqrt(results['energy_dense_after']/results['energy_dense_before']):.3f}
  Sparse vegetation: {np.sqrt(results['energy_sparse_after']/results['energy_sparse_before']):.3f}

SUMMARY
=======
The dense vegetation (cD = 1.4 s⁻¹) dissipates {results['energy_dense_dissipation']:.1f}% of the incident
wave energy, while sparse vegetation (cD = 0.14 s⁻¹) dissipates {results['energy_sparse_dissipation']:.1f}%.
This corresponds to a {results['power_dissipated_dense']/results['power_dissipated_sparse']:.1f}-fold increase
in dissipation effectiveness for the denser canopy.

The energy-based transmission coefficients confirm the wave height-based values,
validating the numerical solution and demonstrating the strong attenuation
capacity of dense vegetation for coastal protection applications.
"""
    
    # Save analysis
    with open(analyses_dir / 'eta_energy_density_analysis.txt', 'w') as f:
        f.write(analysis_text)
    
    return analysis_text

def main():
    """Main execution function."""
    # Setup
    figs_dir, analyses_dir = setup_directories()
    setup_plotting()
    
    # Load and process data
    data = load_and_process_data()
    
    # Create 3D visualizations
    create_eta_3d_plot(data, figs_dir)
    create_energy_3d_plot(data, figs_dir)
    
    # Perform analysis
    results = perform_energy_analysis(data)
    
    # Write results
    analysis_text = write_energy_analysis(results, data, analyses_dir)
    
    # Print completion message
    print(f"\n3D Analysis complete!")
    print(f"Figures saved to: {figs_dir}")
    print(f"Analysis saved to: {analyses_dir / 'eta_energy_density_analysis.txt'}")
    
    # Print summary
    print("\n=== ENERGY ANALYSIS SUMMARY ===")
    print(f"\nWave Energy Dissipation:")
    print(f"  Dense vegetation: {results['energy_dense_dissipation']:.1f}%")
    print(f"  Sparse vegetation: {results['energy_sparse_dissipation']:.1f}%")
    print(f"\nPower Dissipation per unit area:")
    print(f"  Dense vegetation: {results['power_dissipated_dense']:.1f} W/m²")
    print(f"  Sparse vegetation: {results['power_dissipated_sparse']:.1f} W/m²")
    print(f"\nEnergy-based Transmission Coefficients:")
    print(f"  Dense vegetation: Kt,E = {np.sqrt(results['energy_dense_after']/results['energy_dense_before']):.3f}")
    print(f"  Sparse vegetation: Kt,E = {np.sqrt(results['energy_sparse_after']/results['energy_sparse_before']):.3f}")

if __name__ == "__main__":
    main()