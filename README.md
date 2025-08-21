# Supplementary Material: wave-attenuation-1d

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-brightgreen.svg)](http://www.wtfpl.net/)
[![DOI](https://zenodo.org/badge/1032205258.svg)](https://doi.org/10.5281/zenodo.16742589)

Supplementary data and analysis for: *"wave-attenuation-1d: An idealized one-dimensional framework for wave attenuation through coastal vegetation using Numba-accelerated shallow water equations"*

**Authors:** Sandy H. S. Herho, Iwan P. Anwar, Faruq Khadami, Theo R. E. B. N. Ndruru, Rusmawan Suwarman⁴, Dasapta E. Irawan
**Corresponding author:** sandy.herho@email.ucr.edu


## Overview

This repository contains simulation outputs, analysis scripts, and figures comparing wave attenuation through sparse (cD = 0.14 s⁻¹) and dense (cD = 1.4 s⁻¹) vegetation patches.

For the main `wave-attenuation-1d` package, see: https://pypi.org/project/wave-attenuation-1d/

## Structure

```
├── analyses/          # Quantitative results
├── figs/             # Generated figures  
├── raw_data/         # NetCDF simulation outputs
├── script/           # Python analysis scripts
├── LICENSE           # WTFPL License
└── README.md         # This file
```

## Key Results

| Vegetation Type | Drag Coefficient | Wave Height Reduction | Transmission Coefficient |
|----------------|------------------|----------------------|-------------------------|
| Dense          | 1.4 s⁻¹         | 98.9%                | 0.011                  |
| Sparse         | 0.14 s⁻¹        | 34.4%                | 0.656                  |

## Reproducing the Analysis

1. **Install dependencies**:
   ```bash
   pip install numpy xarray matplotlib netcdf4
   ```

2. **Run analysis scripts**:
   ```bash
   cd script/
   python wave_envelope_urms.py
   python energy_density_eta_analysis.py
   ```

## Files

- **raw_data/**: NetCDF files with CF-1.8 compliant simulation outputs
- **analyses/**: Text files with detailed quantitative analysis
- **figs/**: Publication-ready figures showing:
  - 3D spatio-temporal evolution of η and energy density
  - Wave envelope and RMS velocity comparisons
- **script/**: Python scripts for data analysis and visualization

## Citation

```bibtex
@article{herho2025wave,
  author = {Herho, Sandy H. S. and Anwar, Iwan P. and Khadami, F. and Ndruru, Theo R. E. B. N. 
            and Suwarman, Rusmawan and Irawan, Dasapta E.},
  title = {wave-attenuation-1d: An idealized one-dimensional framework 
           for wave attenuation through coastal vegetation},
  journal = {xxx},
  year = {2025}
}
```

## License

Released under the [WTFPL](LICENSE) - Do What The Fuck You Want To Public License
