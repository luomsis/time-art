# Time Series Analysis Web Service

A web-based time series analysis tool featuring Prophet forecasting and anomaly detection using PyOD.

## Features

- **Time Series Prediction**: Uses Facebook Prophet to forecast future values
- **Anomaly Detection**: Multiple algorithms for detecting outliers:
  - Isolation Forest (IForest)
  - Local Outlier Factor (LOF)
  - One-Class SVM (OCSVM)
  - K Nearest Neighbors (KNN)
  - Angle-Based Outlier Detection (ABOD)
  - Clustering-Based Local Outlier Factor (CBLOF)
  - Histogram-Based Outlier Score (HBOS)
  - Minimum Covariance Determinant (MCD)
  - Subspace Outlier Detection (SOD)
- **Interactive Web UI**: Upload JSON files and visualize results with ECharts
- **Configurable Parameters**: Fine-tune algorithms with exposed parameters

## Requirements

- Python 3.12.0+
- uv package manager

## Installation

```bash
# Install dependencies
uv sync

# Or using pip and requirements.txt
pip install -r requirements.txt
```

## Usage

```bash
# Run the web service
uv run python main.py

# Or activate virtual environment first
source .venv/bin/activate
python main.py
```

Access the web interface at http://localhost:9999

## JSON Data Format

Upload a JSON file with the following format:

```json
{
  "units": "",
  "series": [
    {
      "counter": "pg_cpu_used",
      "endpoint": "rase-prd-123-p-szf",
      "data": [
        [1769563495000, 80.21],
        [1769563555000, 64.28],
        [1769563615000, 33.71]
      ]
    }
  ]
}
```

Where `data` is an array of `[timestamp_ms, value]` pairs.

## Project Structure

```
time-art/
├── app/
│   ├── __init__.py
│   └── utils/
│       ├── __init__.py
│       ├── predictor.py      # Prophet prediction module
│       └── detector.py       # Anomaly detection module
├── static/
├── templates/
│   └── index.html            # Web UI
├── main.py                   # Flask application
├── requirements.txt
├── pyproject.toml
└── sample_data.json          # Example data file
```

## API Endpoints

- `GET /` - Web interface
- `POST /analyze` - Analyze time series data
- `GET /algorithms` - List available anomaly detection algorithms

## Prediction Parameters (Prophet)

- Growth Type: linear, logistic, flat
- Forecast Periods: Number of future points to predict
- Changepoints: Number of potential changepoints
- Seasonality Mode: additive or multiplicative
- Seasonality Prior Scale: Strength of seasonality
- And more...

## Anomaly Detection Parameters

- Algorithm: iforest, lof, ocsvm, dbscan, pca
- Contamination: Expected proportion of outliers
- Rolling Window: Window size for rolling statistics
- Feature Engineering: Time features, lag features

## License

MIT License

## Changelog

### 2026-02-05 - Interactive Chart Initialization Fix
- Fixed issue where interactive chart data was compressed on the left side after initial analysis
- Added `requestAnimationFrame` in `switchView()` to ensure DOM is fully updated before rendering
- Enhanced `dataZoom` configuration with `minSpan`, `maxSpan`, and `filterMode` parameters
- Chart now displays full data range immediately after analysis, matching the "reset" button behavior
