"""Anomaly detection module using PyOD."""

import io
import base64
from typing import Dict, Any, List, Tuple

import matplotlib
matplotlib.use('Agg')

# Configure Chinese font
import matplotlib.pyplot as plt
# Font priority: macOS fonts -> Windows fonts -> Linux/CentOS fonts
plt.rcParams['font.sans-serif'] = [
    'Arial Unicode MS', 'PingFang SC', 'STHeiti',  # macOS
    'SimHei', 'Microsoft YaHei',  # Windows
    'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',  # Linux/CentOS wqy
    'Noto Sans CJK SC', 'Noto Sans CJK',  # Linux/CentOS noto
    'DejaVu Sans'  # Fallback
]
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

import pandas as pd
import numpy as np
from matplotlib.dates import DateFormatter
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.knn import KNN
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.mcd import MCD
from pyod.models.sod import SOD


class AnomalyDetector:
    """Anomaly detector using PyOD."""

    # Available algorithms
    ALGORITHMS = {
        'iforest': IForest,
        'lof': LOF,
        'ocsvm': OCSVM,
        'knn': KNN,
        'abod': ABOD,
        'cblof': CBLOF,
        'hbos': HBOS,
        'mcd': MCD,
        'sod': SOD,
    }

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize anomaly detector with parameters.

        Args:
            params: Dictionary containing detection parameters
        """
        self.params = params
        self.model = None
        self.df = None
        self.predictions = None
        self.anomalies = None
        self.decision_scores = None

    def prepare_data(self, data: List[List[float]]) -> pd.DataFrame:
        """
        Prepare data for anomaly detection.

        Args:
            data: List of [timestamp, value] pairs

        Returns:
            DataFrame with prepared features
        """
        df = pd.DataFrame(data, columns=['timestamp', 'value'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('datetime').drop_duplicates(subset=['datetime'])
        df = df.reset_index(drop=True)

        # Create additional features for better detection
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['rolling_mean'] = df['value'].rolling(
            window=self.params.get('rolling_window', 5), min_periods=1).mean()
        df['rolling_std'] = df['value'].rolling(
            window=self.params.get('rolling_window', 5), min_periods=1).std()
        df['rolling_std'] = df['rolling_std'].fillna(df['rolling_std'].mean())

        # Create feature matrix
        feature_cols = ['value', 'rolling_mean', 'rolling_std']

        # Add time-based features if enabled
        if self.params.get('use_time_features', True):
            feature_cols.extend(['hour', 'day_of_week'])

        # Add lag features if enabled
        if self.params.get('use_lag_features', True):
            for lag in range(1, self.params.get('n_lags', 3) + 1):
                df[f'lag_{lag}'] = df['value'].shift(lag)
                df[f'lag_{lag}'] = df[f'lag_{lag}'].bfill()
                feature_cols.append(f'lag_{lag}')

        self.df = df
        return df

    def build_model(self):
        """Build the anomaly detection model with specified algorithm."""
        algorithm = self.params.get('algorithm', 'iforest')
        algorithm_class = self.ALGORITHMS.get(algorithm, IForest)

        # Common parameters
        contamination = self.params.get('contamination', 0.1)

        # Algorithm-specific parameters
        if algorithm == 'iforest':
            n_estimators = self.params.get('n_estimators', 100)
            max_samples = self.params.get('max_samples', 'auto')
            max_features = self.params.get('max_features', 1.0)
            bootstrap = self.params.get('bootstrap', False)

            self.model = IForest(
                n_estimators=n_estimators,
                max_samples=max_samples,
                max_features=max_features,
                bootstrap=bootstrap,
                contamination=contamination,
                random_state=42,
            )

        elif algorithm == 'lof':
            n_neighbors = self.params.get('n_neighbors', 20)
            algorithm_type = self.params.get('algorithm_type', 'auto')
            leaf_size = self.params.get('leaf_size', 30)

            self.model = LOF(
                n_neighbors=n_neighbors,
                algorithm=algorithm_type,
                leaf_size=leaf_size,
                contamination=contamination,
            )

        elif algorithm == 'ocsvm':
            kernel = self.params.get('kernel', 'rbf')
            degree = self.params.get('degree', 3)
            gamma = self.params.get('gamma', 'scale')
            nu = self.params.get('nu', 0.5)

            self.model = OCSVM(
                kernel=kernel,
                degree=degree,
                gamma=gamma,
                nu=nu,
                contamination=contamination,
            )

        elif algorithm == 'knn':
            n_neighbors = self.params.get('n_neighbors', 5)
            method = self.params.get('method', 'largest')
            radius = self.params.get('radius', 1.0)
            algorithm_type = self.params.get('algorithm_type', 'auto')
            leaf_size = self.params.get('leaf_size', 30)

            self.model = KNN(
                n_neighbors=n_neighbors,
                method=method,
                radius=radius,
                algorithm=algorithm_type,
                leaf_size=leaf_size,
                contamination=contamination,
            )

        elif algorithm == 'abod':
            n_neighbors = self.params.get('n_neighbors', 10)
            method = self.params.get('method', 'fast')

            self.model = ABOD(
                n_neighbors=n_neighbors,
                method=method,
                contamination=contamination,
            )

        elif algorithm == 'cblof':
            n_clusters = self.params.get('n_clusters', 8)
            alpha = self.params.get('alpha', 0.9)
            beta = self.params.get('beta', 10)
            use_weights = self.params.get('use_weights', False)

            self.model = CBLOF(
                n_clusters=n_clusters,
                alpha=alpha,
                beta=beta,
                use_weights=use_weights,
                check_estimator=False,
                random_state=42,
                contamination=contamination,
            )

        elif algorithm == 'hbos':
            n_bins = self.params.get('n_bins', 10)
            alpha = self.params.get('alpha', 0.1)
            tol = self.params.get('tol', 0.5)

            self.model = HBOS(
                n_bins=n_bins,
                alpha=alpha,
                tol=tol,
                contamination=contamination,
            )

        elif algorithm == 'mcd':
            store_precision = self.params.get('store_precision', True)
            assume_centered = self.params.get('assume_centered', False)
            support_fraction = self.params.get('support_fraction', None)

            self.model = MCD(
                store_precision=store_precision,
                assume_centered=assume_centered,
                support_fraction=support_fraction,
                contamination=contamination,
            )

        elif algorithm == 'sod':
            n_neighbors = self.params.get('n_neighbors', 20)
            ref_set = self.params.get('ref_set', 10)
            alpha = self.params.get('alpha', 0.8)

            self.model = SOD(
                n_neighbors=n_neighbors,
                ref_set=ref_set,
                alpha=alpha,
                contamination=contamination,
            )

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        return self.model

    def fit_predict(self) -> np.ndarray:
        """Fit the model and make predictions."""
        if self.model is None:
            self.build_model()

        # Prepare feature matrix
        feature_cols = ['value', 'rolling_mean', 'rolling_std']

        if self.params.get('use_time_features', True):
            feature_cols.extend(['hour', 'day_of_week'])

        if self.params.get('use_lag_features', True):
            for lag in range(1, self.params.get('n_lags', 3) + 1):
                feature_cols.append(f'lag_{lag}')

        X = self.df[feature_cols].values

        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)

        # Fit and predict
        self.model.fit(X)
        self.predictions = self.model.labels_
        self.decision_scores = self.model.decision_scores_

        # Get anomaly indices (PyOD uses 1 for anomalies)
        self.anomalies = self.df[self.predictions == 1].copy()

        return self.predictions

    def generate_plot(self, counter: str, endpoint: str) -> str:
        """
        Generate plot showing original data with marked anomalies.

        Args:
            counter: Counter name for title
            endpoint: Endpoint name for title

        Returns:
            Base64 encoded image string
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                        gridspec_kw={'height_ratios': [2, 1]})

        # Main plot - time series with anomalies
        ax1.plot(self.df['datetime'], self.df['value'], 'o-',
                 label='正常', markersize=3, linewidth=1, alpha=0.7, color='#2E86AB')

        # Highlight anomalies
        if len(self.anomalies) > 0:
            ax1.scatter(self.anomalies['datetime'], self.anomalies['value'],
                       color='red', s=100, label='异常', zorder=5,
                       edgecolors='darkred', linewidths=2, alpha=0.8)

        ax1.set_xlabel('时间', fontsize=12)
        ax1.set_ylabel('数值', fontsize=12)
        ax1.set_title(f'异常检测结果\n{endpoint} - {counter}',
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Format x-axis
        ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Decision scores plot
        if self.decision_scores is not None:
            ax2.plot(self.df['datetime'], self.decision_scores,
                    color='purple', linewidth=1.5, alpha=0.7)

            # Calculate threshold
            threshold = np.percentile(self.decision_scores,
                                     (1 - self.params.get('contamination', 0.1)) * 100)
            ax2.axhline(y=threshold, color='red', linestyle='--',
                       linewidth=2, label=f'阈值 ({threshold:.4f})')
            ax2.fill_between(self.df['datetime'], 0, self.decision_scores,
                            where=self.decision_scores >= threshold,
                            color='red', alpha=0.3, label='异常区域')
            ax2.set_xlabel('时间', fontsize=12)
            ax2.set_ylabel('异常分数', fontsize=12)
            ax2.set_title('异常分数', fontsize=12)
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return img_base64

    def run(self, data: List[List[float]], counter: str,
            endpoint: str) -> Tuple[str, Dict[str, Any]]:
        """
        Run the complete anomaly detection pipeline.

        Args:
            data: Time series data
            counter: Counter name
            endpoint: Endpoint name

        Returns:
            Tuple of (base64_image, metrics_dict)
        """
        # Prepare data
        self.prepare_data(data)

        # Build model and detect anomalies
        self.build_model()
        self.fit_predict()

        # Generate plot
        img_base64 = self.generate_plot(counter, endpoint)

        # Calculate metrics
        metrics = self._calculate_metrics()

        return img_base64, metrics

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate detection metrics."""
        n_anomalies = len(self.anomalies)
        n_total = len(self.df)

        metrics = {
            'algorithm': self.params.get('algorithm', 'iforest'),
            'total_points': n_total,
            'anomaly_count': n_anomalies,
            'normal_count': n_total - n_anomalies,
            'anomaly_percentage': round((n_anomalies / n_total * 100) if n_total > 0 else 0, 2),
            'contamination': self.params.get('contamination', 0.1),
        }

        # Add threshold if available
        if self.decision_scores is not None:
            threshold = np.percentile(self.decision_scores,
                                     (1 - self.params.get('contamination', 0.1)) * 100)
            metrics['threshold'] = round(float(threshold), 4)

        # Add anomaly details if any
        if n_anomalies > 0:
            anomaly_values = self.anomalies['value'].values
            metrics.update({
                'min_anomaly_value': round(float(np.min(anomaly_values)), 4),
                'max_anomaly_value': round(float(np.max(anomaly_values)), 4),
                'mean_anomaly_value': round(float(np.mean(anomaly_values)), 4),
            })

        return metrics

    def get_anomaly_details(self) -> List[Dict[str, Any]]:
        """Get detailed information about detected anomalies."""
        if self.anomalies is None or len(self.anomalies) == 0:
            return []

        anomalies = self.anomalies[['datetime', 'value']].copy()
        anomalies['score'] = self.decision_scores[self.predictions == 1]
        anomalies = anomalies.sort_values('score', ascending=False)

        return anomalies.to_dict('records')
