"""SARIMA 时间序列预测模块。

实现基于 statsmodels SARIMA 的时间序列预测功能，支持:
- 自回归差分移动平均模型 (ARIMA)
- 季节性 SARIMA (p,d,q)(P,D,Q,s)
- 自动参数搜索 (auto-ARIMA)
- 置信区间预测
"""

import io
import base64
import warnings
from datetime import datetime
from typing import Dict, Any, List, Tuple

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
# 字体优先级: macOS -> Windows -> Linux/CentOS
plt.rcParams['font.sans-serif'] = [
    'Arial Unicode MS', 'PingFang SC', 'STHeiti',  # macOS
    'SimHei', 'Microsoft YaHei',  # Windows
    'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',  # Linux/CentOS wqy
    'Noto Sans CJK SC', 'Noto Sans CJK',  # Linux/CentOS noto
    'DejaVu Sans'  # Fallback
]
plt.rcParams['axes.unicode_minus'] = False

import pandas as pd
import numpy as np
from matplotlib.dates import DateFormatter

from app.utils.data_cleaner import DataCleaner

# 抑制 statsmodels 的警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class SARIMAPredictor:
    """基于 SARIMA 的时间序列预测器。

    支持功能:
    - ARIMA/SARIMA 时间序列预测
    - 季节性建模
    - 自动参数选择
    - 置信区间预测
    - 交互式图表数据输出
    """

    # 常见的季节性周期 (以5分钟为单位的周期数)
    SEASONAL_PERIODS = {
        'hourly': 12,          # 每小时 (5分钟间隔)
        'daily': 288,          # 每天 (5分钟间隔)
        'weekly': 2016,        # 每周 (5分钟间隔)
        'none': 0,             # 无季节性
    }

    def __init__(self, params: Dict[str, Any]):
        """初始化 SARIMA 预测器。

        Args:
            params: SARIMA 参数字典，包含:
                - p: AR阶数 (0-5)
                - d: 差分阶数 (0-2)
                - q: MA阶数 (0-5)
                - P: 季节性AR阶数 (0-3)
                - D: 季节性差分阶数 (0-2)
                - Q: 季节性MA阶数 (0-3)
                - s: 季节性周期 (hourly/daily/weekly/none)
                - forecast_periods: 预测周期数
                - confidence_level: 置信区间 (0.7-0.99)
                - method: 参数估计方法
                - auto_order: 是否自动选择参数
                - enforce_stationary: 是否强制平稳
                - enforce_invertibility: 是否强制可逆
                - 数据清洗参数 (clean_*)
        """
        self.params = params
        self.model = None
        self.df = None
        self.forecast = None
        self.cleaning_report = {}
        self.order = None
        self.seasonal_order = None

    def prepare_data(self, data: List[List[float]]) -> pd.DataFrame:
        """准备 SARIMA 模型所需的数据格式。

        Args:
            data: [[timestamp_ms, value], ...] 格式的时间序列数据

        Returns:
            DataFrame with 'ds' (datetime) and 'y' (value) columns
        """
        # 数据清洗
        clean_params = {
            'handle_missing': self.params.get('clean_handle_missing', 'interpolate'),
            'handle_inf': self.params.get('clean_handle_inf', True),
            'smooth_outliers': self.params.get('clean_smooth_outliers', 'none'),
            'outlier_threshold': self.params.get('clean_outlier_threshold', 1.5),
            'filter_zero': self.params.get('clean_filter_zero', False),
            'min_data_points': self.params.get('clean_min_data_points', 2),
            'min_time_span_seconds': self.params.get('clean_min_time_span', 0),
        }

        cleaner = DataCleaner(clean_params)
        df = cleaner.clean(data)

        # 保存清洗报告
        self.cleaning_report = cleaner.get_report()

        self.df = df
        return df

    def _determine_seasonal_period(self) -> int:
        """确定季节性周期。

        Returns:
            季节性周期s (0表示无季节性)
        """
        s_param = self.params.get('seasonal_period', 'none')

        # 如果是数字，直接使用
        if isinstance(s_param, int):
            return s_param

        # 如果是预定义名称，查找对应的周期
        return self.SEASONAL_PERIODS.get(s_param, 0)

    def _auto_select_order(self, df: pd.DataFrame) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
        """自动选择 ARIMA 阶数。

        使用简化的启发式方法选择参数:
        - ADF检验确定差分阶数 d
        - ACF/PACF 分析确定 p, q

        Args:
            df: 输入数据

        Returns:
            (order, seasonal_order) 元组
        """
        from statsmodels.tsa.stattools import adfuller
        from statsmodels.tsa.stattools import acf, pacf

        # 准备时间序列
        y = df['y'].values

        # 1. 确定差分阶数 d (ADF检验)
        def get_d_order(series, max_d=2):
            for d in range(max_d + 1):
                if d == 0:
                    test_series = series
                else:
                    test_series = np.diff(series, n=d)

                # 处理可能的常数序列
                if len(np.unique(test_series)) <= 1:
                    return d

                try:
                    result = adfuller(test_series, regression='ct', autolag='AIC')
                    if result[1] < 0.05:  # p-value < 0.05 拒绝单位根假设
                        return d
                except:
                    return d
            return max_d

        d = get_d_order(y)

        # 2. 对差分后的序列进行 ACF/PACF 分析
        diff_y = np.diff(y, n=d) if d > 0 else y

        try:
            # 计算ACF和PACF
            acf_values = acf(diff_y, nlags=min(20, len(diff_y) // 4), fft=True)
            pacf_values = pacf(diff_y, nlags=min(20, len(diff_y) // 4))

            # 简单的阈值方法确定p和q
            # 寻找最后一个显著滞后期
            significance = 1.96 / np.sqrt(len(diff_y))

            # p (AR阶数) - 基于PACF
            p = 0
            for i in range(1, len(pacf_values)):
                if abs(pacf_values[i]) > significance:
                    p = i
                else:
                    break
            p = min(p, 5)  # 限制最大值

            # q (MA阶数) - 基于ACF
            q = 0
            for i in range(1, len(acf_values)):
                if abs(acf_values[i]) > significance:
                    q = i
                else:
                    break
            q = min(q, 5)  # 限制最大值

        except Exception:
            # 默认值
            p, q = 1, 1

        # 3. 季节性参数 (简化)
        s = self._determine_seasonal_period()
        if s > 0:
            P, D, Q = 1, 1, 1
        else:
            P, D, Q = 0, 0, 0

        order = (p, d, q)
        seasonal_order = (P, D, Q, s)

        return order, seasonal_order

    def build_model(self, df: pd.DataFrame = None) -> None:
        """构建 SARIMA 模型。

        Args:
            df: 输入数据，如果为None则使用self.df
        """
        if df is None:
            df = self.df

        # 动态导入 statsmodels (不强制安装)
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            raise ImportError(
                "statsmodels 库未安装。请运行: pip install statsmodels"
            )

        # 获取或自动选择参数
        if self.params.get('auto_order', False):
            order, seasonal_order = self._auto_select_order(df)
        else:
            p = self.params.get('p', 1)
            d = self.params.get('d', 1)
            q = self.params.get('q', 1)
            order = (p, d, q)

            s = self._determine_seasonal_period()
            if s > 0:
                P = self.params.get('P', 1)
                D = self.params.get('D', 1)
                Q = self.params.get('Q', 1)
                seasonal_order = (P, D, Q, s)
            else:
                seasonal_order = (0, 0, 0, 0)

        self.order = order
        self.seasonal_order = seasonal_order

        # 构建模型
        model_kwargs = {
            'order': order,
            'seasonal_order': seasonal_order,
            'enforce_stationarity': self.params.get('enforce_stationarity', True),
            'enforce_invertibility': self.params.get('enforce_invertibility', True),
        }

        # 添加可选参数
        if 'method' in self.params:
            model_kwargs['method'] = self.params['method']

        self.model = SARIMAX(df['y'].values, **model_kwargs)

    def fit(self, df: pd.DataFrame = None) -> None:
        """拟合 SARIMA 模型。

        Args:
            df: 输入数据，如果为None则使用self.df
        """
        if self.model is None:
            self.build_model(df)

        # 拟合模型，不显示收敛信息
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.results = self.model.fit(disp=False, maxiter=100)

    def predict(self, periods: int = None) -> np.ndarray:
        """进行预测。

        Args:
            periods: 预测周期数

        Returns:
            预测结果数组
        """
        if periods is None:
            periods = self.params.get('forecast_periods', 30)

        # 置信水平
        confidence_level = self.params.get('confidence_level', 0.8)
        alpha = 1 - confidence_level

        # 进行预测
        forecast_result = self.results.get_forecast(steps=periods)
        predicted_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=alpha)

        # 创建预测DataFrame
        last_timestamp = self.df['ds'].max()

        # 生成未来时间戳 (假设5分钟间隔)
        time_delta = pd.Timedelta(minutes=5)
        future_dates = [last_timestamp + time_delta * (i + 1) for i in range(periods)]

        self.forecast = pd.DataFrame({
            'ds': future_dates,
            'yhat': predicted_mean.values,
            'yhat_lower': conf_int.iloc[:, 0].values,
            'yhat_upper': conf_int.iloc[:, 1].values,
        })

        # 对非负数据进行截断
        if self.params.get('enforce_non_negative', False):
            self.forecast['yhat'] = self.forecast['yhat'].clip(lower=0)
            self.forecast['yhat_lower'] = self.forecast['yhat_lower'].clip(lower=0)
            self.forecast['yhat_upper'] = self.forecast['yhat_upper'].clip(lower=0)

        return self.forecast

    def generate_plot(self, counter: str, endpoint: str) -> str:
        """生成预测图表。

        Args:
            counter: 指标名称
            endpoint: 端点名称

        Returns:
            Base64编码的图像字符串
        """
        fig, ax = plt.subplots(figsize=(16, 8))

        # 绘制置信区间
        ax.fill_between(self.forecast['ds'],
                        self.forecast['yhat_lower'],
                        self.forecast['yhat_upper'],
                        alpha=0.25, color='#10B981',
                        label=f'置信区间 ({int(self.params.get("confidence_level", 0.8)*100)}%)')

        # 绘制预测趋势线
        ax.plot(self.forecast['ds'], self.forecast['yhat'], '-',
                label='预测趋势', linewidth=2.5, color='#059669')

        # 绘制实际值
        ax.plot(self.df['ds'], self.df['y'], 'o', label='实际值',
                markersize=3, alpha=0.6, color='#0EA5E9',
                markeredgecolor='white', markeredgewidth=0.5)

        # 标记预测起点
        forecast_start = self.df['ds'].max()
        ax.axvline(x=forecast_start, color='#6366F1', linestyle='-',
                   alpha=0.7, linewidth=2, label='预测起点')

        # 添加预测区域阴影
        ax.axvspan(forecast_start, self.forecast['ds'].max(),
                  alpha=0.05, color='#6366F1')

        # 标题和标签
        order_str = f"({self.order[0]},{self.order[1]},{self.order[2]})"
        seasonal_str = f"({self.seasonal_order[0]},{self.seasonal_order[1]},{self.seasonal_order[2]},{self.seasonal_order[3]})"
        if self.seasonal_order[3] == 0:
            title_order = f"ARIMA{order_str}"
        else:
            title_order = f"SARIMA{order_str}×{seasonal_str}"

        ax.set_xlabel('时间', fontsize=13, fontweight='medium')
        ax.set_ylabel('数值', fontsize=13, fontweight='medium')
        ax.set_title(f'SARIMA 时间序列预测\n{title_order} - {endpoint} - {counter}',
                     fontsize=16, fontweight='bold', pad=20)

        # 图例
        ax.legend(loc='upper left', fontsize=11, framealpha=0.95,
                 shadow=True, fancybox=True, borderpad=0.8)

        # 网格
        ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        # 背景色
        ax.set_facecolor('#FAFAFA')
        fig.patch.set_facecolor('#FFFFFF')

        # 格式化x轴
        ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M'))
        plt.xticks(rotation=30, ha='right')

        plt.tight_layout()

        # 转换为base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                   facecolor=fig.get_facecolor())
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return img_base64

    def run(self, data: List[List[float]], counter: str,
            endpoint: str) -> Tuple[str, Dict[str, Any]]:
        """运行完整的预测流程。

        Args:
            data: 时间序列数据
            counter: 指标名称
            endpoint: 端点名称

        Returns:
            (base64图像, 指标字典) 元组
        """
        # 准备数据
        df = self.prepare_data(data)

        # 构建和拟合模型
        self.build_model()
        self.fit()

        # 预测
        self.predict()

        # 生成图表
        img_base64 = self.generate_plot(counter, endpoint)

        # 计算指标
        metrics = self._calculate_metrics()

        return img_base64, metrics

    def get_chart_data(self, counter: str = '', endpoint: str = '') -> Dict[str, Any]:
        """获取交互式图表的原始数据。

        Args:
            counter: 指标名称
            endpoint: 端点名称

        Returns:
            图表数据字典
        """
        # 实际数据
        actual_data = [
            [int(ds.timestamp() * 1000), float(y)]
            for ds, y in zip(self.df['ds'], self.df['y'])
        ]

        # 预测数据
        forecast_data = [
            [int(ds.timestamp() * 1000), float(yhat)]
            for ds, yhat in zip(self.forecast['ds'], self.forecast['yhat'])
        ]

        # 上界
        upper_data = [
            [int(ds.timestamp() * 1000), float(yhat_upper)]
            for ds, yhat_upper in zip(self.forecast['ds'], self.forecast['yhat_upper'])
        ]

        # 下界
        lower_data = [
            [int(ds.timestamp() * 1000), float(yhat_lower)]
            for ds, yhat_lower in zip(self.forecast['ds'], self.forecast['yhat_lower'])
        ]

        forecast_start = int(self.df['ds'].max().timestamp() * 1000)

        return {
            'actual': actual_data,
            'forecast': forecast_data,
            'upper': upper_data,
            'lower': lower_data,
            'forecast_start': forecast_start,
            'counter': counter,
            'endpoint': endpoint,
            'model': 'SARIMA',
            'order': str(self.order),
            'seasonal_order': str(self.seasonal_order),
        }

    def _calculate_metrics(self) -> Dict[str, Any]:
        """计算预测指标。

        Returns:
            指标字典
        """
        # 获取拟合值 (用于计算误差指标)
        try:
            fitted_values = self.results.fittedvalues

            # 计算误差指标
            y_true = self.df['y'].values

            # 对齐数据长度
            min_len = min(len(y_true), len(fitted_values))
            y_true = y_true[:min_len]
            y_pred = fitted_values[:min_len].values if hasattr(fitted_values, 'values') else fitted_values[:min_len]

            # 处理NaN值
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true = y_true[mask]
            y_pred = y_pred[mask]

            metrics = {
                'forecast_periods': len(self.forecast),
                'data_points': len(self.df),
                'forecast_start': self.df['ds'].max().strftime('%Y-%m-%d %H:%M:%S'),
                'forecast_end': self.forecast['ds'].max().strftime('%Y-%m-%d %H:%M:%S'),
                'model': 'SARIMA',
                'order': str(self.order),
                'seasonal_order': str(self.seasonal_order),
            }

            if len(y_true) > 0:
                mse = np.mean((y_true - y_pred) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(y_true - y_pred))

                # MAPE
                non_zero_mask = y_true != 0
                if np.any(non_zero_mask):
                    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) /
                                          y_true[non_zero_mask])) * 100
                else:
                    mape = 0

                # AIC 和 BIC
                metrics.update({
                    'mse': round(mse, 4),
                    'rmse': round(rmse, 4),
                    'mae': round(mae, 4),
                    'mape': round(mape, 2),
                    'aic': round(self.results.aic, 2),
                    'bic': round(self.results.bic, 2),
                })
            else:
                metrics.update({
                    'aic': round(self.results.aic, 2),
                    'bic': round(self.results.bic, 2),
                })
        except Exception as e:
            metrics = {
                'forecast_periods': len(self.forecast),
                'data_points': len(self.df),
                'forecast_start': self.df['ds'].max().strftime('%Y-%m-%d %H:%M:%S'),
                'forecast_end': self.forecast['ds'].max().strftime('%Y-%m-%d %H:%M:%S'),
                'model': 'SARIMA',
                'order': str(self.order),
                'seasonal_order': str(self.seasonal_order),
            }

        # 添加清洗报告
        if self.cleaning_report:
            metrics['cleaning_report'] = self.cleaning_report

        return metrics
