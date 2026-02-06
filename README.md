# Time-Art: 时间序列分析服务

基于 Flask 的 Web 时间序列分析服务，支持 Prophet 预测和 PyOD 异常检测，具备动态阈值特性。

## 特性

### 预测分析 (Prophet)
- **动态阈值**：置信区间随数据波动自适应调整
  - 高负载期阈值变宽（容纳更大波动）
  - 低负载期阈值收窄（更灵敏检测异常）
- **多级季节性**：支持日、周、月、年季节性模式
- **趋势分析**：自动检测趋势变化点

### 异常检测 (PyOD)
支持 9 种异常检测算法：
- IForest (孤立森林)
- LOF (局部离群因子)
- OCSVM (单类 SVM)
- KNN (K 近邻)
- ABOD (基于角度的异常检测)
- CBLOF (基于聚类的 LOF)
- HBOS (直方图异常检测)
- MCD (最小协方差行列式)
- SOD (随机子空间方法)

### 交互式可视化
- **ECharts 交互图表**：支持缩放、平移、时间筛选
- **实时同步**：时间过滤器与图表范围自动同步
- **智能标记**：自动高亮超出阈值的历史数据点
- **双 Y 轴**：异常检测时同时显示数值和异常分数

## 快速开始

### 安装依赖

```bash
# 推荐：使用 uv
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 启动服务

```bash
# 使用 uv 启动
uv run python main.py

# 或激活虚拟环境后启动
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python main.py
```

访问 http://localhost:9999

### 生成测试数据

```bash
# 生成展示动态阈值特性的测试数据
uv run python tests/generate_test_data.py
```

生成的测试文件：
- `test_data_dynamic_threshold.json` - 30天数据，完整动态阈值展示
- `test_data_multi_peak.json` - 14天数据，多峰值模式

## JSON 数据格式

```json
{
  "units": "CPU使用率 (%)",
  "series": [
    {
      "counter": "cpu_usage",
      "endpoint": "database_server",
      "data": [
        [1736160000000, 45.2],
        [1736160060000, 78.5],
        [1736160120000, 62.1]
      ]
    }
  ]
}
```

`data` 格式：`[时间戳(毫秒), 数值]`

## 项目结构

```
time-art/
├── app/
│   └── utils/
│       ├── predictor.py    # Prophet 预测模块
│       └── detector.py     # 异常检测模块
├── static/
│   └── echarts.min.js      # ECharts 图表库
├── templates/
│   └── index.html          # Web UI
├── tests/
│   └── generate_test_data.py  # 测试数据生成器
├── main.py                 # Flask 应用入口
└── requirements.txt
```

## Prophet 预测参数推荐

针对有周期性波动的 CPU 指标推荐以下参数：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `growth` | linear | 线性趋势 |
| `forecast_periods` | 100+ | 预测周期数 |
| `daily_seasonality` | true | 启用日季节性 |
| `weekly_seasonality` | true | 启用周季节性 |
| `seasonality_mode` | multiplicative | 乘法季节性 |
| `seasonality_prior_scale` | 18 | 季节性强度 |
| `interval_width` | 0.9 | 90% 置信区间 |

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 界面 |
| `/analyze` | POST | 分析时间序列数据 |
| `/algorithms` | GET | 获取可用算法列表 |

## 依赖

- Python 3.12+
- Flask - Web 框架
- Prophet - 时间序列预测
- PyOD - 异常检测
- Matplotlib - 静态图表生成
- ECharts - 交互式图表

## 更新日志

### 2026-02-06 - 动态阈值与交互优化
- 添加 Prophet 动态阈值可视化功能
- 优化图表样式，突出阈值区间
- 自动标记超出阈值的历史数据点
- 时间过滤器与图表范围实时同步
- 取消鼠标悬停高亮效果
- 更新 Prohet 参数默认值为推荐配置

### 2026-02-05 - 交互式图表修复
- 修复图表初始化时数据压缩问题
- 优化 dataZoom 配置
- 添加时间筛选功能

## License

MIT License
