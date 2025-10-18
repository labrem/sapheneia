---
title: Sapheneia TimesFM Forecasting
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
tags:
  - time-series
  - forecasting
  - timesfm
  - finance
  - machine-learning
---

# Sapheneia TimesFM: Time Series Forecasting with Google's Foundation Model

A professional web application for time series forecasting powered by Google's TimesFM (Time Series Foundation Model). Built for financial forecasting and analysis with advanced covariates support.

## 🚀 Quick Start

1. **Upload Your Data**: Upload a CSV file with time series data
2. **Configure Model**: Set context length, horizon, and other parameters
3. **Add Covariates** (Optional): Enhance forecasts with exogenous variables
4. **Generate Forecast**: Get point forecasts with prediction intervals
5. **Visualize & Download**: Interactive charts and downloadable results

## ✨ Features

### Core Capabilities
- **TimesFM 2.0-500m Model**: State-of-the-art foundation model for time series
- **Quantile Forecasting**: Prediction intervals for uncertainty quantification
- **Covariates Support**: Dynamic and static, numerical and categorical variables
- **Professional Visualizations**: Publication-quality charts with Plotly
- **Interactive Interface**: User-friendly web application

### Advanced Features
- **Multi-Series Forecasting**: Process multiple time series simultaneously
- **Flexible Horizons**: Forecast from 1 to 128 periods ahead
- **Customizable Context**: Use 64 to 2048 historical data points
- **Real-time Processing**: Fast inference on CPU
- **Export Options**: Download forecasts as CSV or HTML charts

## 📊 Data Format

Your CSV file should have:
- **Date column** as the first column
- **Time series columns** with numerical values
- **Data definition JSON** specifying column types:

```json
{
  "price": "target",
  "temperature": "dynamic_numerical",
  "day_of_week": "dynamic_categorical",
  "store_id": "static_categorical",
  "base_sales": "static_numerical"
}
```

### Column Types
- **target**: Main time series to forecast
- **dynamic_numerical**: Time-varying numerical covariates
- **dynamic_categorical**: Time-varying categorical covariates
- **static_numerical**: Series-level numerical features
- **static_categorical**: Series-level categorical features

## 🎯 Use Cases

### Financial Forecasting
- Stock price prediction
- Revenue forecasting
- Trading volume estimation
- Risk analysis

### Business Analytics
- Sales forecasting
- Demand planning
- Inventory optimization
- Customer behavior prediction

### Research & Academia
- Economic indicators
- Climate data analysis
- Experimental time series
- Comparative studies

## 🔧 Model Configuration

### Recommended Settings

**Quick Testing**:
- Context Length: 64
- Horizon Length: 24
- Backend: CPU

**Production Use**:
- Context Length: 512-2048
- Horizon Length: 24-128
- Backend: CPU (or GPU for faster inference)

### Covariates Configuration
When using covariates:
- Dynamic covariates must cover context + horizon periods
- Use `xreg_mode="xreg + timesfm"` for best results
- Enable normalization for stability
- Start with small ridge values (0.0-0.01)

## 📚 About TimesFM

TimesFM is a decoder-only foundation model for time-series forecasting, pre-trained on 100 billion real-world time points. Key features:

- **Foundation Model**: Pre-trained on diverse time series data
- **Zero-Shot Forecasting**: Works on new data without retraining
- **Attention-Based**: Leverages transformer architecture
- **Production-Ready**: Developed and tested by Google Research

Learn more: [TimesFM Research Paper](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)

## 🛠️ Technical Stack

- **Model**: Google TimesFM 2.0-500m (PyTorch)
- **Backend**: Flask + Python 3.11
- **Visualization**: Plotly + Matplotlib
- **ML Libraries**: JAX, NumPy, Pandas, scikit-learn
- **Deployment**: Docker on Hugging Face Spaces

## 📖 Documentation

- [GitHub Repository](https://github.com/labrem/sapheneia)
- [Full Documentation](https://github.com/labrem/sapheneia/blob/main/README.md)
- [TimesFM GitHub](https://github.com/google-research/timesfm)

## 🤝 Contributing

Contributions are welcome! This is a research project focused on advancing TimesFM capabilities for practical applications.

## 📄 License

MIT License - See [LICENSE](https://github.com/labrem/sapheneia/blob/main/LICENSE) for details.

## 🙏 Acknowledgments

- Google Research for the TimesFM foundation model
- Hugging Face for Spaces infrastructure
- The open-source time series forecasting community

---

**Note**: This application runs on CPU by default. For faster inference on large datasets, consider using GPU-enabled Spaces.
