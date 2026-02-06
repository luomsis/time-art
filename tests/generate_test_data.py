"""
测试数据生成器 - 用于展示Prophet动态阈值特性

生成具有明显周期性和波动性变化的时间序列数据，
使Prophet的动态置信区间能够清晰展示。
"""

import json
import numpy as np
from datetime import datetime, timedelta


def generate_dynamic_threshold_test_data():
    """
    生成测试数据，展示动态阈值特性。

    数据特点：
    1. 基础趋势：缓慢上升的CPU使用率
    2. 日季节性：工作时间(9-18点)CPU较高
    3. 周季节性：工作日vs周末差异
    4. 随机波动：高负载期波动大，低负载期波动小
    """

    # 生成时间范围：30天，每5分钟一个点
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    num_points = 30 * 24 * 12  # 30天 * 24小时 * 12点/小时

    data = []
    base_cpu = 30  # 基础CPU使用率

    for i in range(num_points):
        # 当前时间
        current_time = start_time + timedelta(minutes=5 * i)
        hour = current_time.hour
        day_of_week = current_time.weekday()  # 0=Monday, 6=Sunday

        # 1. 基础趋势：缓慢上升 (每天增加0.5%)
        trend = (i / num_points) * 15

        # 2. 日季节性：工作时间(9-18点)CPU较高
        if 9 <= hour <= 18:
            # 工作时间：高负载，波动大
            hour_pattern = 25 + 10 * np.sin((hour - 9) * np.pi / 9)
            noise_level = 8  # 高波动
        elif 0 <= hour <= 6:
            # 深夜：低负载，波动小
            hour_pattern = -15
            noise_level = 2  # 低波动
        else:
            # 其他时间：中等
            hour_pattern = 5
            noise_level = 4

        # 3. 周季节性：周末CPU较低
        if day_of_week >= 5:  # 周末
            weekly_pattern = -10
            noise_level *= 0.7  # 周末波动更小
        else:
            weekly_pattern = 0

        # 4. 添加随机波动（模拟真实数据的噪声）
        # 波动大小随负载水平变化
        noise = np.random.normal(0, noise_level)

        # 5. 偶尔的尖峰（模拟异常事件）
        spike = 0
        if np.random.random() < 0.005:  # 0.5%概率出现尖峰
            spike = np.random.uniform(15, 30)

        # 计算最终CPU值，并限制在合理范围
        cpu_value = base_cpu + trend + hour_pattern + weekly_pattern + noise + spike
        cpu_value = max(5, min(95, cpu_value))  # 限制在5%-95%

        # 转换为毫秒时间戳
        timestamp_ms = int(current_time.timestamp() * 1000)

        data.append([timestamp_ms, round(cpu_value, 2)])

    return {
        "units": "CPU使用率 (%)",
        "series": [{
            "counter": "cpu_usage_percent",
            "endpoint": "database_server_primary",
            "data": data
        }]
    }


def generate_multi_pattern_test_data():
    """
    生成多模式测试数据，展示不同负载模式下的动态阈值。
    """

    start_time = datetime(2025, 1, 1, 0, 0, 0)
    num_points = 14 * 24 * 12  # 14天
    data = []

    for i in range(num_points):
        current_time = start_time + timedelta(minutes=5 * i)
        hour = current_time.hour
        day = current_time.day

        # 每天有不同的负载模式
        day_pattern = (day % 7) / 7 * 20  # 0-20的日间变化

        # 小时模式：三个高峰期
        if 9 <= hour < 12:
            hour_value = 30 + np.sin((hour - 9) * np.pi / 3) * 15
        elif 14 <= hour < 17:
            hour_value = 25 + np.sin((hour - 14) * np.pi / 3) * 12
        elif 20 <= hour < 23:
            hour_value = 20 + np.sin((hour - 20) * np.pi / 3) * 10
        else:
            hour_value = 10

        # 波动随值变化
        noise_level = 3 + hour_value * 0.15
        noise = np.random.normal(0, noise_level)

        value = 20 + day_pattern + hour_value + noise
        value = max(5, min(95, value))

        timestamp_ms = int(current_time.timestamp() * 1000)
        data.append([timestamp_ms, round(value, 2)])

    return {
        "units": "CPU使用率 (%)",
        "series": [{
            "counter": "cpu_multi_peak",
            "endpoint": "app_server_01",
            "data": data
        }]
    }


def main():
    """生成并保存测试数据文件"""

    print("生成测试数据...")

    # 生成主测试数据（30天，展示完整动态阈值特性）
    test_data1 = generate_dynamic_threshold_test_data()
    with open('test_data_dynamic_threshold.json', 'w') as f:
        json.dump(test_data1, f, indent=2)
    print(f"✓ 生成 test_data_dynamic_threshold.json ({len(test_data1['series'][0]['data'])} 数据点)")

    # 生成多模式测试数据（14天，展示多峰值模式）
    test_data2 = generate_multi_pattern_test_data()
    with open('test_data_multi_peak.json', 'w') as f:
        json.dump(test_data2, f, indent=2)
    print(f"✓ 生成 test_data_multi_peak.json ({len(test_data2['series'][0]['data'])} 数据点)")

    print("\n数据特点:")
    print("- 基础趋势：缓慢上升")
    print("- 日季节性：工作时间高负载，深夜低负载")
    print("- 周季节性：工作日vs周末差异")
    print("- 动态波动：高负载期波动大，低负载期波动小")
    print("- 随机尖峰：模拟偶发异常事件")
    print("\n建议参数设置:")
    print("  - daily_seasonality: true")
    print("  - weekly_seasonality: true")
    print("  - seasonality_mode: multiplicative")
    print("  - seasonality_prior_scale: 15-20")
    print("  - interval_width: 0.85-0.95")


if __name__ == '__main__':
    main()
