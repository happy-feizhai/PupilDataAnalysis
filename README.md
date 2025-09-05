# PupilDataAnalysis

A Python-based scientific analysis tool for processing and visualizing human eye pupil wavefront aberration data using Zernike polynomials. This project is part of PhD research in ophthalmology and vision science, specifically focused on peripheral vision analysis.

## Features

### Core Analysis
- **Zernike Polynomial Analysis**: Implementation of ANSI Z80.28-2010 standard Zernike polynomials with proper normalization
- **Wavefront Error Visualization**: Generate 2D contour plots and radial profiles of wavefront aberrations
- **Clinical Data Processing**: Load and analyze real patient eye measurement data from JSON files
- **RMS Calculations**: Thorlabs WFS standard-compliant RMS error calculations

### Advanced Peripheral Vision Analysis
- **2D Refractive Topography**: Scatter plot visualizations with statistical analysis overlays
- **3D Surface Visualization**: Interactive 3D refractive error surface plots with contour projections
- **Radial Wavefront Arrays**: Circular arrangement of wavefront maps mimicking retinal topology
- **Dynamic Axis Scaling**: Automatic adjustment to accommodate all measurement positions (0°-50° eccentricity)

### Statistical Analysis & Quality Control
- **Batch Processing**: Analyze entire patient populations (95+ patients, 125+ eyes)
- **Robust Outlier Detection**: Modified Z-score method using median absolute deviation (MAD)
- **Population Statistics**: Age, gender, and eye-specific analysis with outlier rates
- **Data Quality Filtering**: Automatic exclusion of problematic measurement positions (50°, 180°)
- **Multi-format Export**: CSV reports, statistical summaries, and high-resolution visualizations

## Installation

### Using uv (Recommended)
```bash
uv sync
```

### Using pip
```bash
pip install -r requirements.txt
```

### Requirements
- Python >=3.14.0
- NumPy >=2.3.2
- Matplotlib >=3.10.6
- Pandas >=2.3.2
- SciPy >=1.16.1

## Usage

### Single Patient Analysis

#### Core Zernike Analysis
```bash
python zernike_test.py
```

#### Advanced Peripheral Vision Analysis
```bash
python peripheral_analysis.py
```

#### Custom Single Patient Analysis
```python
from peripheral_analysis import load_all_measurements, analyze_single_file

# Load patient data
measurements, patient_info = load_all_measurements("testdata_aier/091.json")
print(f"Loaded {len(measurements)} eyes for {patient_info['name']}")

# Generate comprehensive peripheral analysis
analyze_single_file("testdata_aier/091.json", "output_analysis")
```

### Batch Population Analysis
```bash
python batch_analysis.py
```

This will:
- Process all JSON files in `testdata_aier/` directory
- Generate statistical reports with outlier detection
- Export CSV data and summary statistics
- Identify population-level patterns and data quality issues

### Loading Specific Measurements
```python
from zernike_test import load_zernike_from_json, comprehensive_analysis

# Load Zernike coefficients from patient data
coefficients, info = load_zernike_from_json(
    "testdata_aier/001.json", 
    eye='left', 
    offaxis=0, 
    meridian=0
)

# Generate comprehensive analysis
comprehensive_analysis(coefficients)
```

## Data Format

The project processes JSON files containing patient eye measurement data with the following structure:

```json
{
  "name": "Patient Name",
  "gender": "male/female",
  "age": 10,
  "date": "2025.08.27",
  "left": {
    "diopter": "Diopter measurement",
    "points": [
      {
        "offaxis": 15,
        "meridian": 90,
        "result": [
          {
            "zernike": [-13.726, 1.368, 12.914],
            "wf_rms": 26.553,
            "wf_pv": 167.46,
            "sphere": 0.499,
            "cyl": -1.912,
            "axis": 12.038
          }
        ],
        "median": {
          "sphere": 0.533,
          "cyl": -2.629,
          "zernike": [-14.442, 1.548, 13.001]
        }
      }
    ]
  },
  "right": {
    "points": []
  }
}
```

### Coordinate System
- **offaxis**: Eccentricity from fovea (0°, 15°, 30°, 38°, 50°)
- **meridian**: Retinal location (0°=nasal, 90°=superior, 180°=temporal, 270°=inferior)  
- **Zernike coefficients**: ANSI Z80.28-2010 standard ordering (Z0=piston, Z1=tip, Z2=tilt, ...)

### Data Quality Notes
- **(50°, 180°) position excluded** from batch analysis due to systematic measurement issues
- **Median values** used for robust statistics across multiple measurements per position

## Key Functions

### Core Analysis Modules

#### zernike_test.py - Mathematical Core
- `load_zernike_from_json()`: Load Zernike coefficients from JSON patient data
- `zernike_polynomial()`: Calculate Zernike polynomials following ANSI standard
- `generate_wavefront_error()`: Create 2D wavefront error maps
- `comprehensive_analysis()`: Generate multi-panel analysis report
- `calculate_thorlabs_rms()`: Compute RMS error following Thorlabs standards

#### peripheral_analysis.py - Advanced Visualization
- `load_all_measurements()`: Load complete patient measurement dataset
- `create_refractive_topography()`: Generate 2D refractive error plots with statistics
- `create_3d_refractive_topography()`: Create 3D surface visualizations
- `create_wavefront_array()`: Build radial wavefront map arrays
- `analyze_single_file()`: Complete single-patient analysis pipeline

#### batch_analysis.py - Population Analysis
- `batch_analyze_all_patients()`: Process entire patient database
- `detect_outliers_modified_zscore()`: Robust outlier detection using MAD
- `generate_summary_statistics()`: Comprehensive population statistics
- `save_analysis_report()`: Multi-format data export (CSV, TXT)

## Analysis Output

### Single Patient Analysis
1. **2D Refractive Topography**: Scatter plots with statistical overlays and outlier detection
2. **3D Surface Visualization**: Interactive surface plots with contour projections
3. **Radial Wavefront Arrays**: Circular arrangement mimicking retinal structure
4. **Statistical Summary**: Mean, variance, outliers with position information

### Population Analysis
1. **Comprehensive Statistics**: Age/gender/eye distributions with outlier rates
2. **Data Quality Assessment**: Systematic error identification and exclusion
3. **Export Reports**: CSV data files and detailed text summaries
4. **Outlier Analysis**: Modified Z-score detection with clinical significance

### Visualization Features
- **Dynamic Scaling**: Automatic axis adjustment for all measurement positions
- **Chinese Font Support**: Clinical report compatibility
- **High-Resolution Export**: 300 DPI output suitable for publications
- **Statistical Overlays**: Real-time outlier highlighting and annotations

## Statistical Methods

### Outlier Detection Algorithm
The system uses a **Modified Z-Score** method for robust outlier detection:

```
Modified Z-Score = 0.6745 × (x - median) / MAD
```

Where:
- **MAD**: Median Absolute Deviation (robust alternative to standard deviation)
- **0.6745**: Normalization factor for normal distribution equivalence
- **Threshold**: 3.5 (identifies ~0.05% most extreme values)

### Advantages
- **Robust to outliers**: Median-based statistics unaffected by extreme values
- **Small sample friendly**: Effective with 14-15 measurements per eye
- **Distribution independent**: No assumption of normal distribution required

## Research Context

This tool is designed for analyzing human eye aberrations in the context of:

### Clinical Applications
- **Peripheral vision research**: Mapping refractive error across retinal eccentricity
- **Refractive surgery planning**: Pre-operative assessment of wavefront aberrations
- **Custom contact lens design**: Patient-specific optical correction profiles
- **Vision correction assessment**: Post-treatment efficacy evaluation

### Research Capabilities
- **Population studies**: Large-scale analysis of refractive patterns
- **Quality control**: Systematic measurement error identification
- **Data mining**: Pattern recognition in clinical datasets
- **Longitudinal tracking**: Patient progress monitoring over time

## Standards Compliance

This implementation follows the **ANSI Z80.28-2010** standard for Zernike polynomials, ensuring:
- Clinical device compatibility
- Standardized normalization factors
- Reproducible research results
- International data exchange capability

## Project Structure

```
PupilDataAnalysis/
├── zernike_test.py          # Core mathematical analysis
├── peripheral_analysis.py   # Advanced visualization
├── batch_analysis.py        # Population statistics
├── testdata_aier/          # Patient measurement data (95 files)
├── output_analysis/        # Generated analysis results
├── pyproject.toml          # Project dependencies
└── README.md              # This documentation
```

## License

This project is part of ongoing PhD research. Please contact the author for usage permissions and collaboration opportunities.

## Contributing

This is a research project. For questions or collaboration inquiries, please reach out through appropriate academic channels.

---

**Note**: This tool processes sensitive medical data. Ensure compliance with relevant privacy regulations and institutional review board requirements when handling patient information.