import os
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import json
# Machine Learning imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import set_global_policy

# Set GPU flags
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/cuda/" #your path maybe different, depending upon the os you are using

# Enable mixed precision to optimize GPU computations
set_global_policy('mixed_float16')

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Configure CPU threading for 16 thread cpu
tf.config.threading.set_intra_op_parallelism_threads(16)  # Number of intra-op threads (per operation)
tf.config.threading.set_inter_op_parallelism_threads(16)  # Number of inter-op threads (parallel ops)

# Load environment variables
load_dotenv()

# API keys
STOCK_API_KEY = os.getenv("STOCK")
NEWS_API_KEY = os.getenv("NEWS_SENTIMENT")

# Seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def fetch_stock_data(stock_name="ADANIENT", period="1yr", filter="default"):
    """
    Fetch historical stock data from API
    """
    url = f'https://stock.indianapi.in/historical_data?stock_name={stock_name}&period={period}&filter={filter}'
    headers = {'X-Api-Key': STOCK_API_KEY}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        values = data["datasets"][0]["values"]
        
        # Convert to DataFrame
        historical_data = pd.DataFrame(values, columns=["Date", "Price"])
        historical_data['Date'] = pd.to_datetime(historical_data['Date'])
        historical_data['Price'] = historical_data['Price'].astype(float)
        historical_data.set_index('Date', inplace=True)
        
        # Sort index to ensure chronological order
        historical_data.sort_index(inplace=True)
        
        return historical_data
    
    except requests.RequestException as e:
        print(f"Error fetching stock data: {e}")
        return None

def fetch_sentiment_score():
    """
    Fetch sentiment score for the stock with detailed error handling
    """
    url = 'https://api.marketaux.com/v1/news/all'
    params = {
        'countries': 'in',
        'filter_entities': 'true',
        'limit': 3,  # API max limit in free plan
        'published_after': '2024-11-13T05:30',
        'api_token': NEWS_API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        
        # Print full response for debugging
        print("Full API Response Status:", response.status_code)
        print("Response Content:", response.text)
        
        if response.status_code != 200:
            print(f"API Error: {response.status_code} - {response.text}")
            return 0
        
        data = response.json()
        
        # Print raw data for debugging
        print("Raw API Data:", json.dumps(data, indent=2))
        
        sentiment_scores = []
        for item in data.get('data', []):
            entities = item.get('entities', [])
            for entity in entities:
                # Symbol matching with extended keywords
                if entity['symbol'] in ['ADANIENT.NS', 'ADANIENT', 'ADANI']:
                    sentiment_scores.append(entity.get('sentiment_score', 0))
        
        # More detailed logging
        print("Found Sentiment Scores:", sentiment_scores)
        
        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            print(f"Average Sentiment: {avg_sentiment}")
            return avg_sentiment
        else:
            print("No sentiment scores found for the stock")
            return 0
    
    except Exception as e:
        print(f"Comprehensive Error in fetch_sentiment_score: {e}")
        import traceback
        traceback.print_exc()
        return 0

def create_features(df):
    """
    Create additional features for the model - adjusted for 1-year timeframe
    """
    df = df.copy()
    
    # Calculate returns
    df['Returns'] = df['Price'].pct_change()
    
    # Adjusted moving averages for shorter timeframe
    df['MA10'] = df['Price'].rolling(window=10).mean()  # Shorter MA
    df['MA20'] = df['Price'].rolling(window=20).mean()
    df['MA50'] = df['Price'].rolling(window=50).mean()  # Removed MA200
    
    # Exponential Moving Average
    df['EMA10'] = df['Price'].ewm(span=10, adjust=False).mean()  # Shorter EMA
    
    # Price momentum - shorter period
    df['Momentum'] = df['Price'].pct_change(periods=10)  # Reduced from 20
    
    # Volatility measure
    df['Volatility'] = df['Returns'].rolling(window=10).std()  # New feature
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    return df

def create_sequences(data, seq_length):
    """
    Create sequences for LSTM input
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])
    
    return np.array(X), np.array(y)

def preprocess_data(historical_data, market_sentiment, seq_length=60):  # Reduced sequence length
    """
    Preprocess data for LSTM model
    """
    # Add features
    df = create_features(historical_data)
    
    # Add market sentiment
    df['Market_Sentiment'] = market_sentiment
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    
    # Create sequences
    X, y = create_sequences(scaled_data, seq_length)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler, df.columns

def build_lstm_model(input_shape):
    """
    Build LSTM model - adjusted architecture for 1-year data
    """
    model = Sequential([
        Input(shape=input_shape),  # Explicit input layer

        # First LSTM Layer
        LSTM(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),

        # Second LSTM Layer
        LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),

        # Third LSTM Layer
        LSTM(32, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),

        # Dense Layers
        Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.2),
        Dense(1)  # Single output for regression
    ])

    # Compile model with mixed precision optimizer
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Optimized learning rate
        loss='mean_squared_error',
        metrics=['mae']
    )
    return model

def preprocess_dataset(dataset):
    dataset = dataset.shuffle(buffer_size=10000)  # Shuffle data
    dataset = dataset.cache()  # Cache in memory
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)  # Prefetch batches for training
    return dataset

def train_model(X_train, X_test, y_train, y_test):
    """
    Train the LSTM model with adjusted parameters for 1-year data
    """
    model = build_lstm_model(X_train.shape[1:])
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10,  # Reduced patience
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=5, # Reduced patience
        min_lr=0.00001
    )
    
    history = model.fit(
        X_train, y_train, 
        epochs=100,  # Reduced epochs
        batch_size=64,  # Reduced batch size
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return model, history

def predict_next_day(model, last_sequence, scaler, market_sentiment):
    """
    Predict next day's price
    """
    # Add market sentiment to the last sequence
    last_sequence_with_sentiment = np.column_stack([
        last_sequence, 
        np.full((last_sequence.shape[0], 1), market_sentiment)
    ])
    
    last_sequence_with_sentiment = last_sequence_with_sentiment.reshape(
        1, *last_sequence_with_sentiment.shape
    )
    
    predicted_normalized = model.predict(last_sequence_with_sentiment)
    
    predicted_price = scaler.inverse_transform(
        np.concatenate([predicted_normalized, 
                       np.zeros((predicted_normalized.shape[0], 
                               scaler.scale_.shape[0] - 1))], 
                      axis=1)
    )[0, 0]
    
    return predicted_price

def plot_training_history(history):
    """
    Visualize model training history
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.show()

def main():
    try:
        print("Fetching historical stock data...")
        historical_data = fetch_stock_data("ADANIENT", period="1yr", filter="default")
        
        if historical_data is None or len(historical_data) == 0:
            print("No historical data fetched. Check API connection.")
            return
        
        print(f"Data points: {len(historical_data)}")
        print(f"Date range: {historical_data.index.min()} to {historical_data.index.max()}")
        
        print("Fetching market sentiment...")
        market_sentiment = fetch_sentiment_score()
        print(f"Market Sentiment Score: {market_sentiment}")
        
        market_sentiment_series = pd.Series(
            [market_sentiment] * len(historical_data), 
            index=historical_data.index
        )
        
        print("Preprocessing data...")
        X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(
            historical_data, 
            market_sentiment_series, 
            seq_length=60  # Using 60 days for sequence length
        )
        
        print("Training model...")
        model, history = train_model(X_train, X_test, y_train, y_test)
        
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {loss}")
        print(f"Mean Absolute Error: {mae}")
        
        # Prepare last sequence with all features
        last_features = create_features(historical_data).iloc[-60:]
        last_features['Market_Sentiment'] = market_sentiment
        
        last_sequence = scaler.transform(last_features)
        predicted_price = predict_next_day(model, last_sequence, scaler, market_sentiment)
        print(f"Predicted Next Day Price: {predicted_price}")
        
        plot_training_history(history)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
