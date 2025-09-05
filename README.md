# PupilDataAnalysis

A Python-based scientific analysis tool for processing and visualizing human eye pupil wavefront aberration data using Zernike polynomials. This project is part of PhD research in ophthalmology and vision science.

## Features

- **Zernike Polynomial Analysis**: Implementation of ANSI Z80.28-2010 standard Zernike polynomials with proper normalization
- **Wavefront Error Visualization**: Generate 2D contour plots and radial profiles of wavefront aberrations
- **Clinical Data Processing**: Load and analyze real patient eye measurement data from JSON files
- **Comprehensive Analysis Reports**: Multi-panel visualizations with statistical analysis
- **RMS Calculations**: Thorlabs WFS standard-compliant RMS error calculations

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
- Python e3.14.0
- NumPy e2.3.2
- Matplotlib e3.10.6

## Usage

### Basic Analysis
```bash
python zernike_test.py
```

### Loading Patient Data
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

### Available Measurements
```python
from zernike_test import list_available_measurements

# List all available measurements in a file
measurements = list_available_measurements("testdata_aier/001.json")
print(measurements)
```

## Data Format

The project processes JSON files containing patient eye measurement data with the following structure:

```json
{
  "name": "£Ó",
  "gender": "'+",
  "age": t„,
  "date": "KÏå",
  "left": {
    "diopter": "HI¦",
    "points": [
      {
        "offaxis": 0,
        "meridian": 0,
        "result": [
          {
            "zernike": [ûppÄ],
            "wf_rms": RMS<,
            "wf_pv": "ð7<",
            "sphere": "\¦p",
            "cyl": "ñ\¦p",
            "axis": "tM"
          }
        ]
      }
    ]
  }
}
```

## Key Functions

### Data Loading
- `load_zernike_from_json()`: Load Zernike coefficients from JSON patient data
- `list_available_measurements()`: List all available measurements in JSON files

### Mathematical Analysis
- `zernike_polynomial()`: Calculate Zernike polynomials following ANSI standard
- `generate_wavefront_error()`: Create 2D wavefront error maps
- `calculate_thorlabs_rms()`: Compute RMS error following Thorlabs standards

### Visualization
- `comprehensive_analysis()`: Generate multi-panel analysis report including:
  - Wavefront error contour plots
  - Zernike coefficient bar charts  
  - RMS contribution by polynomial order
  - Radial profile analysis
  - Statistical summaries

## Analysis Output

The comprehensive analysis generates:

1. **Wavefront Error Distribution**: 2D contour plot showing aberration patterns
2. **Coefficient Analysis**: Bar chart of individual Zernike coefficients
3. **Order Contribution**: RMS contribution by polynomial order
4. **Radial Profiles**: Horizontal and vertical wavefront profiles
5. **Statistical Summary**: Key metrics including RMS, P-V values, and clinical parameters

## Standards Compliance

This implementation follows the **ANSI Z80.28-2010** standard for Zernike polynomials, ensuring compatibility with clinical wavefront measurement devices and providing standardized normalization factors.

## Research Context

This tool is designed for analyzing human eye aberrations in the context of:
- Peripheral vision research
- Refractive surgery planning
- Custom contact lens design
- Vision correction assessment

## License

This project is part of ongoing PhD research. Please contact the author for usage permissions and collaboration opportunities.

## Contributing

This is a research project. For questions or collaboration inquiries, please reach out through appropriate academic channels.