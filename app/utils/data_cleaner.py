"""时间序列数据清洗模块。

提供多种数据清洗功能:
- 缺失值处理（删除/插值/填充）
- 无穷值处理
- 异常值平滑处理（IQR/3σ方法）
- 零值过滤
- 数据质量验证
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


class DataCleaner:
    """时间序列数据清洗器。

    支持功能:
    - 缺失值处理
    - 无穷值处理
    - 异常值平滑
    - 零值过滤
    - 数据质量验证
    """

    def __init__(self, params: Dict[str, Any]):
        """初始化数据清洗器。

        Args:
            params: 清洗参数字典，包含:
                - handle_missing: 缺失值处理方式 ('delete', 'interpolate', 'ffill', 'bfill', 'mean')
                - handle_inf: 是否处理无穷值
                - smooth_outliers: 异常值平滑方法 ('none', 'iqr', 'sigma3')
                - outlier_threshold: 异常值阈值倍数 (默认 IQR用1.5, 3σ用3)
                - filter_zero: 是否过滤零值
                - min_data_points: 最小数据点数量
                - min_time_span_seconds: 最小时间跨度（秒）
        """
        self.params = params
        self.cleaning_report = {}

    def clean(self, data: List[List[float]], value_col: str = 'y') -> pd.DataFrame:
        """执行完整的数据清洗流程。

        Args:
            data: [[timestamp_ms, value], ...] 格式的时间序列数据
            value_col: 值列的列名

        Returns:
            清洗后的 DataFrame

        Raises:
            ValueError: 当数据质量不符合要求时
        """
        # 转换为 DataFrame
        df = pd.DataFrame(data, columns=['ds', value_col])
        df['ds'] = pd.to_datetime(df['ds'], unit='ms')

        # 记录初始状态
        initial_count = len(df)
        self.cleaning_report = {
            'initial_count': initial_count,
            'deleted_duplicates': 0,
            'deleted_missing': 0,
            'deleted_zero': 0,
            'deleted_inf': 0,
            'smoothed_outliers': 0,
            'interpolated_values': 0,
            'final_count': 0
        }

        # 1. 排序和去重
        df = df.sort_values('ds').drop_duplicates(subset=['ds'])
        duplicates_removed = initial_count - len(df)
        self.cleaning_report['deleted_duplicates'] = duplicates_removed

        # 2. 处理无穷值
        if self.params.get('handle_inf', True):
            before_count = len(df)
            inf_mask = np.isinf(df[value_col])
            if inf_mask.any():
                df.loc[inf_mask, value_col] = np.nan
            self.cleaning_report['deleted_inf'] = int(inf_mask.sum())

        # 3. 处理缺失值
        df = self._handle_missing_values(df, value_col)

        # 4. 过滤零值
        if self.params.get('filter_zero', False):
            before_count = len(df)
            df = df[df[value_col] != 0]
            self.cleaning_report['deleted_zero'] = before_count - len(df)

        # 5. 异常值平滑处理
        df = self._smooth_outliers(df, value_col)

        # 6. 重置索引
        df = df.reset_index(drop=True)

        # 7. 数据质量验证
        self._validate_data(df)

        # 记录最终状态
        self.cleaning_report['final_count'] = len(df)
        self.cleaning_report['total_removed'] = initial_count - len(df)
        self.cleaning_report['retention_rate'] = round(len(df) / initial_count * 100, 2) if initial_count > 0 else 0

        return df

    def _handle_missing_values(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """处理缺失值。"""
        method = self.params.get('handle_missing', 'interpolate')

        missing_count = int(df[value_col].isna().sum())

        if missing_count == 0:
            return df

        if method == 'delete':
            df = df.dropna(subset=[value_col])
            self.cleaning_report['deleted_missing'] = missing_count

        elif method == 'interpolate':
            before_missing = int(df[value_col].isna().sum())
            df[value_col] = df[value_col].interpolate(method='linear', limit_direction='both')
            # 如果首尾还有NaN，用前后值填充（使用新语法）
            df[value_col] = df[value_col].bfill().ffill()
            self.cleaning_report['interpolated_values'] = before_missing

        elif method == 'ffill':
            before_missing = int(df[value_col].isna().sum())
            df[value_col] = df[value_col].ffill()
            # 如果开头是NaN，用第一个非NaN值填充
            if df[value_col].iloc[0] != df[value_col].iloc[0]:  # 检查是否为NaN
                df[value_col] = df[value_col].bfill()
            self.cleaning_report['interpolated_values'] = before_missing

        elif method == 'bfill':
            before_missing = int(df[value_col].isna().sum())
            df[value_col] = df[value_col].bfill()
            # 如果结尾是NaN，用最后一个非NaN值填充
            if df[value_col].iloc[-1] != df[value_col].iloc[-1]:  # 检查是否为NaN
                df[value_col] = df[value_col].ffill()
            self.cleaning_report['interpolated_values'] = before_missing

        elif method == 'mean':
            before_missing = int(df[value_col].isna().sum())
            mean_value = df[value_col].mean()
            if not np.isnan(mean_value):
                df[value_col] = df[value_col].fillna(mean_value)
                self.cleaning_report['interpolated_values'] = before_missing
            else:
                # 如果均值也是NaN，则删除
                df = df.dropna(subset=[value_col])
                self.cleaning_report['deleted_missing'] = before_missing

        return df

    def _smooth_outliers(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """异常值平滑处理。"""
        method = self.params.get('smooth_outliers', 'none')

        if method == 'none' or len(df) == 0:
            return df

        outlier_mask = pd.Series([False] * len(df), index=df.index)

        if method == 'iqr':
            threshold = self.params.get('outlier_threshold', 1.5)
            Q1 = df[value_col].quantile(0.25)
            Q3 = df[value_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = (df[value_col] < lower_bound) | (df[value_col] > upper_bound)

        elif method == 'sigma3':
            threshold = self.params.get('outlier_threshold', 3)
            mean = df[value_col].mean()
            std = df[value_col].std()
            if std > 0:
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                outlier_mask = (df[value_col] < lower_bound) | (df[value_col] > upper_bound)

        # 对异常值进行平滑处理（用滚动均值替换）
        if outlier_mask.any():
            self.cleaning_report['smoothed_outliers'] = int(outlier_mask.sum())
            # 使用滚动窗口均值替换异常值
            window = min(5, len(df) // 2)  # 窗口大小为5或数据长度的一半
            if window >= 2:
                rolling_mean = df[value_col].rolling(window=window, center=True, min_periods=1).mean()
                df.loc[outlier_mask, value_col] = rolling_mean[outlier_mask]

        return df

    def _validate_data(self, df: pd.DataFrame) -> None:
        """验证数据质量。"""
        # 检查数据点数量
        min_points = self.params.get('min_data_points', 2)
        if len(df) < min_points:
            raise ValueError(f"数据点数量不足：清洗后仅剩 {len(df)} 个点，最少需要 {min_points} 个点")

        # 检查时间跨度
        min_span = self.params.get('min_time_span_seconds', 0)
        if min_span > 0 and len(df) >= 2:
            time_span = (df['ds'].max() - df['ds'].min()).total_seconds()
            if time_span < min_span:
                raise ValueError(f"时间跨度过短：{time_span} 秒，最少需要 {min_span} 秒")

    def get_report(self) -> Dict[str, Any]:
        """获取清洗报告。"""
        return self.cleaning_report

    def get_report_summary(self) -> str:
        """获取清洗报告摘要（用于显示）。"""
        r = self.cleaning_report
        summary_parts = []

        if r.get('deleted_duplicates', 0) > 0:
            summary_parts.append(f"删除重复: {r['deleted_duplicates']}")
        if r.get('deleted_missing', 0) > 0:
            summary_parts.append(f"删除缺失: {r['deleted_missing']}")
        if r.get('interpolated_values', 0) > 0:
            summary_parts.append(f"插值填充: {r['interpolated_values']}")
        if r.get('deleted_inf', 0) > 0:
            summary_parts.append(f"处理无穷值: {r['deleted_inf']}")
        if r.get('deleted_zero', 0) > 0:
            summary_parts.append(f"过滤零值: {r['deleted_zero']}")
        if r.get('smoothed_outliers', 0) > 0:
            summary_parts.append(f"平滑异常值: {r['smoothed_outliers']}")

        if summary_parts:
            return "数据清洗: " + ", ".join(summary_parts) + f" | 保留率: {r.get('retention_rate', 100)}%"
        else:
            return "数据清洗: 无需处理"
