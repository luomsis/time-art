"""Flask web application for time series analysis.

支持 Prophet 预测和 PyOD 异常检测的 Web 服务。
- 端口: 9999
- 最大文件上传: 16MB
- 支持格式: JSON
"""

import json
import os
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from app.utils.predictor import ProphetPredictor
from app.utils.detector import AnomalyDetector

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = Path('uploads')
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['ALLOWED_EXTENSIONS'] = {'json'}


def allowed_file(filename):
    """检查文件扩展名是否允许。"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """渲染主页面。"""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """处理时间序列分析请求。

    支持两种分析类型:
    - prediction: Prophet 预测
    - detection: PyOD 异常检测
    """
    try:
        # 检查文件是否存在
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a JSON file.'}), 400

        # 读取并解析 JSON
        try:
            content = file.read().decode('utf-8')
            data = json.loads(content)
        except json.JSONDecodeError as e:
            return jsonify({'error': f'Invalid JSON format: {str(e)}'}), 400
        except Exception as e:
            return jsonify({'error': f'Error reading file: {str(e)}'}), 400

        # 验证 JSON 结构
        if 'series' not in data or not data['series']:
            return jsonify({'error': 'Invalid JSON format: "series" field is required'}), 400

        series_data = data['series'][0]
        if 'data' not in series_data or not series_data['data']:
            return jsonify({'error': 'Invalid JSON format: "data" field is required'}), 400

        counter = series_data.get('counter', 'unknown')
        endpoint = series_data.get('endpoint', 'unknown')
        time_series = series_data['data']

        # 获取分析类型
        analysis_type = request.form.get('analysis_type', 'prediction')

        if analysis_type == 'prediction':
            return _run_prediction(time_series, counter, endpoint)
        else:
            return _run_detection(time_series, counter, endpoint)

    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


def _run_prediction(time_series, counter, endpoint):
    """运行 Prophet 预测。

    Args:
        time_series: 时间序列数据 [[timestamp, value], ...]
        counter: 指标名称
        endpoint: 端点名称

    Returns:
        JSON 响应，包含预测结果、图表数据和指标
    """
    try:
        # 获取 Prophet 参数
        params = {
            'growth': request.form.get('growth', 'linear'),
            'n_changepoints': int(request.form.get('n_changepoints', 25)),
            'changepoint_range': float(request.form.get('changepoint_range', 0.8)),
            'yearly_seasonality': request.form.get('yearly_seasonality', 'auto'),
            'weekly_seasonality': request.form.get('weekly_seasonality', 'auto'),
            'daily_seasonality': request.form.get('daily_seasonality', 'auto'),
            'seasonality_mode': request.form.get('seasonality_mode', 'additive'),
            'seasonality_prior_scale': float(request.form.get('seasonality_prior_scale', 10.0)),
            'changepoint_prior_scale': float(request.form.get('changepoint_prior_scale', 0.05)),
            'interval_width': float(request.form.get('interval_width', 0.8)),
            'forecast_periods': int(request.form.get('forecast_periods', 30)),
            'add_monthly_seasonality': request.form.get('add_monthly_seasonality') == 'true',
        }

        # 处理季节性布尔值
        for key in ['yearly_seasonality', 'weekly_seasonality', 'daily_seasonality']:
            val = params[key]
            if val == 'true':
                params[key] = True
            elif val == 'false':
                params[key] = False

        # 运行预测
        predictor = ProphetPredictor(params)
        img_base64, metrics = predictor.run(time_series, counter, endpoint)

        # 获取交互式图表数据
        chart_data = predictor.get_chart_data(counter, endpoint)

        return jsonify({
            'success': True,
            'image': img_base64,
            'metrics': metrics,
            'counter': counter,
            'endpoint': endpoint,
            'type': 'prediction',
            'chart_data': chart_data
        })

    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


def _run_detection(time_series, counter, endpoint):
    """运行 PyOD 异常检测。

    Args:
        time_series: 时间序列数据 [[timestamp, value], ...]
        counter: 指标名称
        endpoint: 端点名称

    Returns:
        JSON 响应，包含检测结果、异常详情和图表数据
    """
    try:
        # 获取检测参数
        params = {
            'algorithm': request.form.get('algorithm', 'iforest'),
            'contamination': float(request.form.get('contamination', 0.1)),
            'use_time_features': request.form.get('use_time_features', 'true') == 'true',
            'use_lag_features': request.form.get('use_lag_features', 'true') == 'true',
            'n_lags': int(request.form.get('n_lags', 3)),
            'rolling_window': int(request.form.get('rolling_window', 5)),
        }

        # 算法特定参数
        algorithm = params['algorithm']

        if algorithm == 'iforest':
            max_samples_val = request.form.get('max_samples', 'auto')
            if max_samples_val != 'auto':
                max_samples_val = int(max_samples_val)
            params.update({
                'n_estimators': int(request.form.get('n_estimators', 100)),
                'max_samples': max_samples_val,
                'max_features': float(request.form.get('max_features', 1.0)),
                'bootstrap': request.form.get('bootstrap', 'false') == 'true',
            })
        elif algorithm == 'lof':
            params.update({
                'n_neighbors': int(request.form.get('n_neighbors', 20)),
                'algorithm_type': request.form.get('algorithm_type', 'auto'),
                'leaf_size': int(request.form.get('leaf_size', 30)),
            })
        elif algorithm == 'ocsvm':
            params.update({
                'kernel': request.form.get('kernel', 'rbf'),
                'degree': int(request.form.get('degree', 3)),
                'gamma': request.form.get('gamma', 'scale'),
                'nu': float(request.form.get('nu', 0.5)),
            })
        elif algorithm == 'knn':
            params.update({
                'n_neighbors': int(request.form.get('n_neighbors', 5)),
                'method': request.form.get('knn_method', 'largest'),
                'radius': float(request.form.get('radius', 1.0)),
                'algorithm_type': request.form.get('algorithm_type', 'auto'),
                'leaf_size': int(request.form.get('leaf_size', 30)),
            })
        elif algorithm == 'abod':
            params.update({
                'n_neighbors': int(request.form.get('abod_neighbors', 10)),
                'method': request.form.get('abod_method', 'fast'),
            })
        elif algorithm == 'cblof':
            params.update({
                'n_clusters': int(request.form.get('n_clusters', 8)),
                'alpha': float(request.form.get('alpha', 0.9)),
                'beta': int(request.form.get('beta', 10)),
                'use_weights': request.form.get('use_weights', 'false') == 'true',
            })
        elif algorithm == 'hbos':
            params.update({
                'n_bins': int(request.form.get('n_bins', 10)),
                'alpha': float(request.form.get('hbos_alpha', 0.1)),
                'tol': float(request.form.get('tol', 0.5)),
            })
        elif algorithm == 'mcd':
            params.update({
                'store_precision': request.form.get('store_precision', 'true') == 'true',
                'assume_centered': request.form.get('assume_centered', 'false') == 'true',
            })
        elif algorithm == 'sod':
            params.update({
                'n_neighbors': int(request.form.get('n_neighbors', 20)),
                'ref_set': int(request.form.get('ref_set', 10)),
                'alpha': float(request.form.get('sod_alpha', 0.8)),
            })

        # 运行检测
        detector = AnomalyDetector(params)
        img_base64, metrics = detector.run(time_series, counter, endpoint)

        # 获取异常详情
        anomaly_details = detector.get_anomaly_details()

        # 获取交互式图表数据
        chart_data = detector.get_chart_data(counter, endpoint)

        return jsonify({
            'success': True,
            'image': img_base64,
            'metrics': metrics,
            'anomalies': anomaly_details[:50],  # 限制返回 50 条异常
            'counter': counter,
            'endpoint': endpoint,
            'type': 'detection',
            'chart_data': chart_data
        })

    except Exception as e:
        return jsonify({'error': f'Detection error: {str(e)}'}), 500


@app.route('/algorithms')
def get_algorithms():
    """获取可用的异常检测算法列表。"""
    return jsonify({
        'algorithms': list(AnomalyDetector.ALGORITHMS.keys())
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9999)
