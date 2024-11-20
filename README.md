# SentimentSeer

**SentimentSeer** is a machine learning-powered stock price prediction tool that combines historical price data with sentiment analysis from news sources. Using LSTM (Long Short-Term Memory) neural networks, it forecasts future stock trends by leveraging time-series data and market sentiment.

---

## Features

- 📈 **Stock Price Prediction**: Predicts stock prices based on historical data.
- 📰 **Sentiment Analysis**: Incorporates market sentiment from news articles to enhance accuracy.
- 🧠 **Deep Learning Architecture**: Utilizes a 3-layer LSTM model optimized for performance.
- ⚡ **Optimized Pipeline**: Includes mixed-precision training, multi-threaded CPU preprocessing, and GPU acceleration.

---

## Installation

### Prerequisites

1. **Python 3.10+**
2. **TensorFlow 2.13+**
3. **CUDA (for GPU support)**: Ensure compatibility with TensorFlow and your GPU.

### Clone the Repository

```bash
git clone https://github.com/rektshadow/sentimentseer.git
cd sentimentseer
pip install -r requirements.txt

Create a .env file with your API keys:

STOCK=your_stock_api_key
NEWS_SENTIMENT=your_news_sentiment_api_key
