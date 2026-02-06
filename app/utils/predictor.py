"""Prophet 时间序列预测模块。

实现基于 Facebook Prophet 的时间序列预测功能，支持:
- 趋势预测 (linear/logistic/flat)
- 多级季节性 (日/周/月/年)
- 动态阈值 (置信区间)
- 异常点检测 (超出阈值的历史数据)
"""

import io
import base64
from datetime import datetime
from typing import Dict, Any, List, Tuple

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 配置中文字体支持
import matplotlib.pyplot as plt
# 字体优先级: macOS -> Windows -> Linux/CentOS
plt.rcParams['font.sans-serif'] = [
    'Arial Unicode MS', 'PingFang SC', 'STHeiti',  # macOS
    'SimHei', 'Microsoft YaHei',  # Windows
    'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',  # Linux/CentOS wqy
    'Noto Sans CJK SC', 'Noto Sans CJK',  # Linux/CentOS noto
    'DejaVu Sans'  # Fallback
]
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示

import pandas as pd
import numpy as np
from prophet import Prophet
from matplotlib.dates import DateFormatter


class ProphetPredictor:
    """基于 Prophet 的时间序列预测器。

    支持功能:
    - 时间序列预测与拟合
    - 动态阈值区间 (置信区间)
    - 历史异常点检测
    - 交互式图表数据输出
    """

    def __init__(self, params: Dict[str, Any]):
        """初始化 Prophet 预测器。

        Args:
            params: Prophet 参数字典，包含:
                - growth: 增长类型 (linear/logistic/flat)
                - seasonality_mode: 季节性模式 (additive/multiplicative)
                - seasonality_prior_scale: 季节性强度
                - interval_width: 置信区间宽度
                - daily/weekly/yearly_seasonality: 季节性开关
        """
        self.params = params
        self.model = None
        self.df = None
        self.forecast = None

    def prepare_data(self, data: List[List[float]]) -> pd.DataFrame:
        """准备 Prophet 模型所需的数据格式。

        Args:
            data: [[timestamp_ms, value], ...] 格式的时间序列数据

        Returns:
            DataFrame with 'ds' and 'y' columns (and 'cap' for logistic growth)
        """
        df = pd.DataFrame(data, columns=['ds', 'y'])
        df['ds'] = pd.to_datetime(df['ds'], unit='ms')
        df = df.sort_values('ds').drop_duplicates(subset=['ds'])
        df = df.reset_index(drop=True)

        # Add capacity column for logistic growth
        growth = self.params.get('growth', 'linear')
        if growth == 'logistic':
            # Set cap to max value + 10% margin for logistic growth
            cap_value = df['y'].max() * 1.1
            df['cap'] = cap_value

        self.df = df
        return df

    def build_model(self) -> Prophet:
        """Build and configure Prophet model with parameters."""
        model_params = {
            'growth': self.params.get('growth', 'linear'),
            'changepoints': self.params.get('changepoints', None),
            'n_changepoints': self.params.get('n_changepoints', 25),
            'changepoint_range': self.params.get('changepoint_range', 0.8),
            'yearly_seasonality': self.params.get('yearly_seasonality', 'auto'),
            'weekly_seasonality': self.params.get('weekly_seasonality', 'auto'),
            'daily_seasonality': self.params.get('daily_seasonality', 'auto'),
            'seasonality_mode': self.params.get('seasonality_mode', 'additive'),
            'seasonality_prior_scale': self.params.get('seasonality_prior_scale', 10.0),
            'holidays_prior_scale': self.params.get('holidays_prior_scale', 10.0),
            'changepoint_prior_scale': self.params.get('changepoint_prior_scale', 0.05),
            'mcmc_samples': self.params.get('mcmc_samples', 0),
            'interval_width': self.params.get('interval_width', 0.8),
            'uncertainty_samples': self.params.get('uncertainty_samples', 1000),
        }

        # Remove None values
        model_params = {k: v for k, v in model_params.items() if v is not None}

        self.model = Prophet(**model_params)

        # Add custom seasonality if specified
        if self.params.get('add_monthly_seasonality'):
            self.model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        return self.model

    def fit(self, df: pd.DataFrame) -> None:
        """Fit the Prophet model."""
        if self.model is None:
            self.build_model()
        self.model.fit(df)

    def predict(self, periods: int = None) -> pd.DataFrame:
        """
        Make predictions.

        Args:
            periods: Number of periods to forecast into the future

        Returns:
            DataFrame with predictions
        """
        if periods is None:
            periods = self.params.get('forecast_periods', 30)

        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq='5min')

        # Add cap column for logistic growth
        growth = self.params.get('growth', 'linear')
        if growth == 'logistic':
            cap_value = self.df['y'].max() * 1.1
            future['cap'] = cap_value

        # Make predictions
        self.forecast = self.model.predict(future)
        return self.forecast

    def generate_plot(self, counter: str, endpoint: str) -> str:
        """
        Generate comparison plot of actual vs predicted values.

        Args:
            counter: Counter name for title
            endpoint: Endpoint name for title

        Returns:
            Base64 encoded image string
        """
        fig, ax = plt.subplots(figsize=(16, 8))

        # Plot confidence interval (uncertainty band) - draw first to be behind other elements
        ax.fill_between(self.forecast['ds'],
                        self.forecast['yhat_lower'],
                        self.forecast['yhat_upper'],
                        alpha=0.25, color='#F59E0B',
                        label=f'动态阈值区间 ({int(self.params.get("interval_width", 0.8)*100)}% 置信度)')

        # Plot threshold lines for better visibility
        ax.plot(self.forecast['ds'], self.forecast['yhat_upper'],
                '-', linewidth=1, alpha=0.5, color='#F59E0B',
                linestyle='--', label='_nolegend_')
        ax.plot(self.forecast['ds'], self.forecast['yhat_lower'],
                '-', linewidth=1, alpha=0.5, color='#F59E0B',
                linestyle='--', label='_nolegend_')

        # Plot predicted values (fitted + forecast)
        ax.plot(self.forecast['ds'], self.forecast['yhat'], '-',
                label='预测趋势', linewidth=2.5, color='#7C3AED')

        # Plot actual values from original data
        ax.plot(self.df['ds'], self.df['y'], 'o', label='实际值',
                markersize=3, alpha=0.6, color='#0EA5E9', markeredgecolor='white',
                markeredgewidth=0.5)

        # Mark points that exceed the threshold (anomalies in historical data)
        historical_forecast = self.forecast[self.forecast['ds'] <= self.df['ds'].max()]
        outliers = self.df[(self.df['y'] > historical_forecast['yhat_upper'].values[:len(self.df)]) |
                          (self.df['y'] < historical_forecast['yhat_lower'].values[:len(self.df)])]
        if len(outliers) > 0:
            ax.scatter(outliers['ds'], outliers['y'], s=80,
                      facecolors='none', edgecolors='#EF4444',
                      linewidths=2, label='超出阈值', zorder=5)

        # Mark the forecast start point with a styled vertical line
        forecast_start = self.df['ds'].max()
        ax.axvline(x=forecast_start, color='#6366F1', linestyle='-',
                   alpha=0.7, linewidth=2, label='预测起点')

        # Add shaded region for forecast period
        ax.axvspan(forecast_start, self.forecast['ds'].max(),
                  alpha=0.05, color='#6366F1')

        ax.set_xlabel('时间', fontsize=13, fontweight='medium')
        ax.set_ylabel('数值', fontsize=13, fontweight='medium')
        ax.set_title(f'Prophet 动态阈值预测\n{endpoint} - {counter}',
                     fontsize=16, fontweight='bold', pad=20)

        # Enhanced legend
        ax.legend(loc='upper left', fontsize=11, framealpha=0.95,
                 shadow=True, fancybox=True, borderpad=0.8)

        # Enhanced grid
        ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        # Light background
        ax.set_facecolor('#FAFAFA')
        fig.patch.set_facecolor('#FFFFFF')

        # Format x-axis
        ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M'))
        plt.xticks(rotation=30, ha='right')

        plt.tight_layout()

        # Convert to base64 with higher quality
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                   facecolor=fig.get_facecolor())
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return img_base64

    def run(self, data: List[List[float]], counter: str,
            endpoint: str) -> Tuple[str, Dict[str, Any]]:
        """
        Run the complete prediction pipeline.

        Args:
            data: Time series data
            counter: Counter name
            endpoint: Endpoint name

        Returns:
            Tuple of (base64_image, metrics_dict)
        """
        # Prepare data
        df = self.prepare_data(data)

        # Build and fit model
        self.build_model()
        self.fit(df)

        # Make predictions
        self.predict()

        # Generate plot
        img_base64 = self.generate_plot(counter, endpoint)

        # Calculate metrics
        metrics = self._calculate_metrics()

        return img_base64, metrics

    def get_chart_data(self, counter: str = '', endpoint: str = '') -> Dict[str, Any]:
        """Get raw data for interactive chart rendering."""
        # Convert timestamps to milliseconds for JavaScript
        actual_data = [
            [int(ds.timestamp() * 1000), float(y)]
            for ds, y in zip(self.df['ds'], self.df['y'])
        ]

        forecast_data = [
            [int(ds.timestamp() * 1000), float(yhat)]
            for ds, yhat in zip(self.forecast['ds'], self.forecast['yhat'])
        ]

        upper_data = [
            [int(ds.timestamp() * 1000), float(yhat_upper)]
            for ds, yhat_upper in zip(self.forecast['ds'], self.forecast['yhat_upper'])
        ]

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
            'endpoint': endpoint
        }

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate prediction metrics."""
        # Merge forecast with original data to get actual values
        merged = self.forecast.merge(self.df[['ds', 'y']], on='ds', how='left', suffixes=('', '_actual'))
        forecast_hist = merged[merged['ds'] <= self.df['ds'].max()]

        metrics = {
            'forecast_periods': len(self.forecast) - len(self.df),
            'data_points': len(self.df),
            'forecast_start': self.df['ds'].max().strftime('%Y-%m-%d %H:%M:%S'),
            'forecast_end': self.forecast['ds'].max().strftime('%Y-%m-%d %H:%M:%S'),
        }

        if len(forecast_hist) > 0 and 'y_actual' in forecast_hist.columns:
            y_true = forecast_hist['y_actual'].values
            y_pred = forecast_hist['yhat'].values

            # Handle NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true = y_true[mask]
            y_pred = y_pred[mask]

            if len(y_true) > 0:
                mse = np.mean((y_true - y_pred) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(y_true - y_pred))

                # MAPE (avoiding division by zero)
                non_zero_mask = y_true != 0
                if np.any(non_zero_mask):
                    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) /
                                          y_true[non_zero_mask])) * 100
                else:
                    mape = 0

                metrics.update({
                    'mse': round(mse, 4),
                    'rmse': round(rmse, 4),
                    'mae': round(mae, 4),
                    'mape': round(mape, 2),
                })

        return metrics
