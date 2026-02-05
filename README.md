# Time Series Analysis Web Service

A web-based time series analysis tool featuring Prophet forecasting and anomaly detection using scikit-learn.

## Features

- **Time Series Prediction**: Uses Facebook Prophet to forecast future values
- **Anomaly Detection**: Multiple algorithms for detecting outliers:
  - Isolation Forest
  - Local Outlier Factor (LOF)
  - One-Class SVM
  - DBSCAN Clustering
  - PCA-based detection
- **Interactive Web UI**: Upload JSON files and visualize results
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

Access the web interface at http://localhost:5000

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
