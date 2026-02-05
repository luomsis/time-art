# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Application
```bash
# Preferred: using uv
uv run python main.py

# Or using pip/venv
source .venv/bin/activate  # if using venv
python main.py
```

The Flask app runs on **port 9999** (not 5000) at http://localhost:9999

### Installing Dependencies
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

## Architecture Overview

**Time-Art** is a Flask-based web service for time series analysis with two main capabilities:

1. **Time Series Prediction** ([`app/utils/predictor.py`](app/utils/predictor.py)) - Uses Facebook Prophet for forecasting
2. **Anomaly Detection** ([`app/utils/detector.py`](app/utils/detector.py)) - Uses PyOD with 9 different algorithms

### Request Flow

```
Flask (main.py) → Validates JSON → Routes to predictor or detector module → Returns chart_data + Base64 image + metrics
```

- **Input**: JSON file upload with format `{ "units": "", "series": [{ "counter": "", "endpoint": "", "data": [[timestamp_ms, value], ...] }] }`
- **Output**: JSON response with:
  - `chart_data`: Raw data for interactive ECharts visualization
  - `image`: Base64-encoded PNG (fallback/static view)
  - `metrics`: Analysis metrics
  - `anomalies`: Detected anomalies list (for detection)

### Key Components

| File | Purpose |
|------|---------|
| [`main.py`](main.py) | Flask routes, file upload handling, parameter validation |
| [`app/utils/predictor.py`](app/utils/predictor.py) | `ProphetPredictor` class - forecasting with configurable growth/seasonality |
| [`app/utils/detector.py`](app/utils/detector.py) | `AnomalyDetector` class - 9 PyOD algorithms with feature engineering |
| [`templates/index.html`](templates/index.html) | Single-page web UI with ECharts interactive charts |
| [`static/echarts.min.js`](static/echarts.min.js) | ECharts library (local, 1.1MB) |

### Architecture Patterns

- **Strategy Pattern**: `AnomalyDetector.ALGORITHMS` dict maps algorithm names to PyOD classes
- **Factory Pattern**: Algorithm instances created dynamically based on form parameters
- **Modular Design**: ML logic separated from web layer in `app/utils/`
- **Dual Output**: Both static images (matplotlib) and interactive data (ECharts) are generated

### Supported Anomaly Detection Algorithms

The `ALGORITHMS` dict in [`detector.py`](app/utils/detector.py#L40-L50) maps:
- `iforest`, `lof`, `ocsvm`, `knn`, `abod`, `cblof`, `hbos`, `mcd`, `sod`

Each has algorithm-specific parameters handled in `main.py:_run_detection()`.

### Interactive Chart Features

The web UI supports interactive visualization using ECharts:

1. **View Toggle**: Switch between interactive chart and static image
2. **Zoom & Pan**: Mouse wheel zoom, drag to pan (via ECharts dataZoom)
3. **Time Filter**: Set start/end time range to filter displayed data
4. **Dual Y-Axis** (anomaly detection): Values + anomaly scores on separate axes

### Chart Data Format

**Prediction** returns:
```javascript
{
  actual: [[timestamp_ms, value], ...],
  forecast: [[timestamp_ms, value], ...],
  upper: [[timestamp_ms, value], ...],
  lower: [[timestamp_ms, value], ...],
  forecast_start: timestamp_ms,
  counter: string,
  endpoint: string
}
```

**Detection** returns:
```javascript
{
  normal: [[timestamp_ms, value], ...],
  anomaly: [[timestamp_ms, value], ...],
  scores: [[timestamp_ms, score], ...],
  threshold: number,
  counter: string,
  endpoint: string
}
```

### Chinese Localization

The UI is in Chinese. Matplotlib fonts are configured with fallback priority:
1. macOS: Arial Unicode MS, PingFang SC, STHeiti
2. Windows: SimHei, Microsoft YaHei
3. Linux/CentOS: WenQuanYi Zen Hei, Noto Sans CJK SC

This configuration appears in both [`predictor.py`](app/utils/predictor.py#L14-L20) and [`detector.py`](app/utils/detector.py#L13-L19).

### Important Implementation Details

- **Matplotlib Backend**: Uses `Agg` backend (non-display) for server-side image generation
- **Feature Engineering**: Anomaly detector creates time features, lag features, and rolling statistics automatically
- **Data Handling**: Timestamps are milliseconds, converted to pandas datetime with `unit='ms'`
- **Logistic Growth**: Prophet adds `cap` column (max value + 10%) when growth='logistic'
- **Port**: Server runs on port 9999 (not 5000)
- **Max File Size**: 16MB limit configured in Flask
- **ECharts Loading**: Local file at `/static/echarts.min.js` (loaded from Gitee mirror)
- **No Testing**: No test framework is currently implemented

### Adding a New Algorithm

1. Import the PyOD model class in [`detector.py`](app/utils/detector.py)
2. Add to `ALGORITHMS` dict
3. Add parameter parsing in `main.py:_run_detection()`
4. Add `elif` branch in `detector.py:build_model()`
