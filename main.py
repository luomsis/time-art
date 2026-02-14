"""Flask web application for time series analysis.

支持 Prophet 预测和 PyOD 异常检测的 Web 服务。
- 端口: 9999
- 最大文件上传: 16MB
- 支持格式: JSON
- 日志目录: logs/
"""

import argparse
import json
import logging
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from app.utils.predictor import ProphetPredictor
from app.utils.detector import AnomalyDetector
from app.utils.sarima_predictor import SARIMAPredictor

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 配置日志
def setup_logging(debug=False):
    """配置日志系统。

    Args:
        debug: 是否启用调试模式
    """
    # 创建日志目录
    log_dir = PROJECT_ROOT / 'logs'
    log_dir.mkdir(exist_ok=True)

    # 日志格式
    log_format = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s [%(name)s:%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 简化的控制台格式
    console_format = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s %(message)s',
        datefmt='%H:%M:%S'
    )

    # 获取根日志记录器
    logger = logging.getLogger()

    # 设置日志级别
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # 清除现有处理器
    logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # 文件处理器 - 按大小轮转 (10MB, 保留5个备份)
    file_handler = RotatingFileHandler(
        log_dir / 'app.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    # 错误日志处理器 - 按时间轮转 (每天)
    error_handler = TimedRotatingFileHandler(
        log_dir / 'error.log',
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(log_format)
    error_handler.suffix = '%Y-%m-%d'
    logger.addHandler(error_handler)

    return logger

# 初始化日志
logger = setup_logging(debug=False)

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
    logger.info("访问主页")
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
            logger.warning("请求中未包含文件")
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            logger.warning("未选择文件")
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            logger.warning(f"不支持的文件类型: {file.filename}")
            return jsonify({'error': 'Invalid file type. Please upload a JSON file.'}), 400

        # 读取并解析 JSON
        try:
            content = file.read().decode('utf-8')
            data = json.loads(content)
            logger.info(f"成功解析 JSON 文件: {file.filename}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON 格式错误: {str(e)}")
            return jsonify({'error': f'Invalid JSON format: {str(e)}'}), 400
        except Exception as e:
            logger.error(f"读取文件错误: {str(e)}")
            return jsonify({'error': f'Error reading file: {str(e)}'}), 400

        # 验证 JSON 结构
        if 'series' not in data or not data['series']:
            logger.error("JSON 缺少 'series' 字段")
            return jsonify({'error': 'Invalid JSON format: "series" field is required'}), 400

        series_data = data['series'][0]
        if 'data' not in series_data or not series_data['data']:
            logger.error("JSON 缺少 'data' 字段")
            return jsonify({'error': 'Invalid JSON format: "data" field is required'}), 400

        counter = series_data.get('counter', 'unknown')
        endpoint = series_data.get('endpoint', 'unknown')
        raw_time_series = series_data['data']

        # 过滤掉包含 null 值的数据点
        time_series = [
            point for point in raw_time_series
            if point is not None and len(point) == 2 and point[0] is not None and point[1] is not None
        ]

        original_count = len(raw_time_series)
        filtered_count = len(time_series)
        if original_count > filtered_count:
            logger.info(f"过滤 null 值: 原始 {original_count} 个点, 有效 {filtered_count} 个点")

        logger.info(f"分析请求 - 类型: 预测, 指标: {counter}, 端点: {endpoint}, 数据点数: {len(time_series)}")

        # 获取分析类型
        analysis_type = request.form.get('analysis_type', 'prediction')

        if analysis_type == 'prediction':
            return _run_prediction(time_series, counter, endpoint)
        else:
            return _run_detection(time_series, counter, endpoint)

    except Exception as e:
        logger.exception(f"分析请求异常: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


def _run_prediction(time_series, counter, endpoint):
    """运行时间序列预测 (Prophet 或 SARIMA)。

    Args:
        time_series: 时间序列数据 [[timestamp, value], ...]
        counter: 指标名称
        endpoint: 端点名称

    Returns:
        JSON 响应，包含预测结果、图表数据和指标
    """
    try:
        # 获取预测模型类型
        prediction_model = request.form.get('prediction_model', 'prophet')

        if prediction_model == 'sarima':
            logger.info(f"开始 SARIMA 预测 - 指标: {counter}, 端点: {endpoint}")
            return _run_sarima_prediction(time_series, counter, endpoint)
        else:
            logger.info(f"开始 Prophet 预测 - 指标: {counter}, 端点: {endpoint}")
            return _run_prophet_prediction(time_series, counter, endpoint)

    except Exception as e:
        logger.exception(f"预测错误: {str(e)}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


def _run_prophet_prediction(time_series, counter, endpoint):
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
            'enforce_non_negative': request.form.get('enforce_non_negative', 'true') == 'true',
            # 数据清洗参数
            'clean_handle_missing': request.form.get('clean_handle_missing', 'interpolate'),
            'clean_handle_inf': request.form.get('clean_handle_inf', 'true') == 'true',
            'clean_smooth_outliers': request.form.get('clean_smooth_outliers', 'none'),
            'clean_outlier_threshold': float(request.form.get('clean_outlier_threshold', 1.5)),
            'clean_filter_zero': request.form.get('clean_filter_zero', 'false') == 'true',
            'clean_min_data_points': int(request.form.get('clean_min_data_points', 2)),
            'clean_min_time_span': int(request.form.get('clean_min_time_span', 0)),
        }

        logger.debug(f"Prophet 参数: {params}")

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

        logger.info(f"Prophet 预测完成 - 数据点: {metrics.get('data_points', 'N/A')}, "
                   f"预测周期: {metrics.get('forecast_periods', 'N/A')}, "
                   f"MAPE: {metrics.get('mape', 'N/A')}%")

        # 获取交互式图表数据
        chart_data = predictor.get_chart_data(counter, endpoint)

        return jsonify({
            'success': True,
            'image': img_base64,
            'metrics': metrics,
            'counter': counter,
            'endpoint': endpoint,
            'type': 'prediction',
            'model': 'prophet',
            'chart_data': chart_data
        })

    except Exception as e:
        logger.exception(f"Prophet 预测错误: {str(e)}")
        raise


def _run_sarima_prediction(time_series, counter, endpoint):
    """运行 SARIMA 预测。

    Args:
        time_series: 时间序列数据 [[timestamp, value], ...]
        counter: 指标名称
        endpoint: 端点名称

    Returns:
        JSON 响应，包含预测结果、图表数据和指标
    """
    try:
        # 获取 SARIMA 参数
        params = {
            # ARIMA 阶数
            'p': int(request.form.get('sarima_p', 1)),
            'd': int(request.form.get('sarima_d', 1)),
            'q': int(request.form.get('sarima_q', 1)),
            # 季节性阶数
            'P': int(request.form.get('sarima_P', 1)),
            'D': int(request.form.get('sarima_D', 1)),
            'Q': int(request.form.get('sarima_Q', 1)),
            # 季节性周期
            'seasonal_period': request.form.get('sarima_s', 'none'),
            # 预测参数
            'forecast_periods': int(request.form.get('forecast_periods', 30)),
            'confidence_level': float(request.form.get('confidence_level', 0.8)),
            # 模型选项
            'auto_order': request.form.get('auto_order', 'false') == 'true',
            'enforce_stationarity': request.form.get('enforce_stationarity', 'true') == 'true',
            'enforce_invertibility': request.form.get('enforce_invertibility', 'true') == 'true',
            'enforce_non_negative': request.form.get('enforce_non_negative', 'true') == 'true',
            # 数据清洗参数
            'clean_handle_missing': request.form.get('clean_handle_missing', 'interpolate'),
            'clean_handle_inf': request.form.get('clean_handle_inf', 'true') == 'true',
            'clean_smooth_outliers': request.form.get('clean_smooth_outliers', 'none'),
            'clean_outlier_threshold': float(request.form.get('clean_outlier_threshold', 1.5)),
            'clean_filter_zero': request.form.get('clean_filter_zero', 'false') == 'true',
            'clean_min_data_points': int(request.form.get('clean_min_data_points', 2)),
            'clean_min_time_span': int(request.form.get('clean_min_time_span', 0)),
        }

        logger.debug(f"SARIMA 参数: p={params['p']}, d={params['d']}, q={params['q']}, "
                   f"P={params['P']}, D={params['D']}, Q={params['Q']}, s={params['seasonal_period']}")

        # 运行预测
        predictor = SARIMAPredictor(params)
        img_base64, metrics = predictor.run(time_series, counter, endpoint)

        logger.info(f"SARIMA 预测完成 - 数据点: {metrics.get('data_points', 'N/A')}, "
                   f"预测周期: {metrics.get('forecast_periods', 'N/A')}, "
                   f"MAPE: {metrics.get('mape', 'N/A')}%")

        # 获取交互式图表数据
        chart_data = predictor.get_chart_data(counter, endpoint)

        return jsonify({
            'success': True,
            'image': img_base64,
            'metrics': metrics,
            'counter': counter,
            'endpoint': endpoint,
            'type': 'prediction',
            'model': 'SARIMA',
            'chart_data': chart_data
        })

    except Exception as e:
        logger.exception(f"SARIMA 预测错误: {str(e)}")
        raise


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
        logger.info(f"开始异常检测 - 指标: {counter}, 端点: {endpoint}")

        # 获取检测参数
        params = {
            'algorithm': request.form.get('algorithm', 'iforest'),
            'contamination': float(request.form.get('contamination', 0.1)),
            'use_time_features': request.form.get('use_time_features', 'true') == 'true',
            'use_lag_features': request.form.get('use_lag_features', 'true') == 'true',
            'n_lags': int(request.form.get('n_lags', 3)),
            'rolling_window': int(request.form.get('rolling_window', 5)),
            # 数据清洗参数
            'clean_handle_missing': request.form.get('clean_handle_missing', 'interpolate'),
            'clean_handle_inf': request.form.get('clean_handle_inf', 'true') == 'true',
            'clean_smooth_outliers': request.form.get('clean_smooth_outliers', 'none'),
            'clean_outlier_threshold': float(request.form.get('clean_outlier_threshold', 1.5)),
            'clean_filter_zero': request.form.get('clean_filter_zero', 'false') == 'true',
            'clean_min_data_points': int(request.form.get('clean_min_data_points', 2)),
            'clean_min_time_span': int(request.form.get('clean_min_time_span', 0)),
        }

        logger.debug(f"检测参数: algorithm={params['algorithm']}, "
                    f"contamination={params['contamination']}")

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

        logger.info(f"检测完成 - 总数据点: {metrics.get('total_points', 'N/A')}, "
                   f"异常数: {metrics.get('anomaly_count', 'N/A')}, "
                   f"异常占比: {metrics.get('anomaly_percentage', 'N/A')}%")

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
        logger.exception(f"检测错误: {str(e)}")
        return jsonify({'error': f'Detection error: {str(e)}'}), 500


@app.route('/algorithms')
def get_algorithms():
    """获取可用的异常检测算法列表。"""
    logger.debug("获取算法列表")
    return jsonify({
        'algorithms': list(AnomalyDetector.ALGORITHMS.keys())
    })


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description='Time-Art 时间序列分析服务',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
    python main.py              # 默认配置启动
    python main.py --port 8888 # 指定端口
    python main.py --debug      # 调试模式
        '''
    )
    parser.add_argument('--port', type=int, default=9999,
                        help='服务监听端口 (默认: 9999)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='服务监听地址 (默认: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')
    parser.add_argument('--no-reload', action='store_true',
                        help='禁用自动重载')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # 重新配置日志（根据调试模式）
    logger = setup_logging(debug=args.debug)

    logger.info('=' * 60)
    logger.info('Time-Art 时间序列分析服务启动')
    logger.info('=' * 60)
    logger.info(f"监听地址: {args.host}:{args.port}")
    logger.info(f"调试模式: {'启用' if args.debug else '禁用'}")
    logger.info(f"自动重载: {'禁用' if args.no_reload else '启用'}")
    logger.info(f"日志目录: {PROJECT_ROOT / 'logs'}")
    logger.info('=' * 60)

    # 启动服务
    app.run(host=args.host, port=args.port, debug=args.debug,
             use_reloader=not args.no_reload)
