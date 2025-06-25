import requests
import pandas as pd
import numpy as np
import xgboost as xgb
import ta

# === TELEGRAM CONFIG ===
BOT_TOKEN = '8171180329:AAHGYhyURBK4wi5vP8y29BfhkYq9IHQnqEk'
CHAT_ID = '6899457606'
MODEL_PATH = 'btc_xgb_model.bin'

# === TELEGRAM SEND ===
def send_telegram_message(text):
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    payload = {'chat_id': CHAT_ID, 'text': text}
    response = requests.post(url, data=payload)
    print(f"Telegram response: {response.status_code}, {response.json()}")

# === FETCH BTC/USDT OHLCV DATA ===
def fetch_binance_ohlcv(symbol="BTCUSDT", interval="1m", limit=361):
    url = "https://api.binance.com/api/v3/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    r = requests.get(url, params=params)
    r.raise_for_status()
    raw = r.json()

    df = pd.DataFrame(raw, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "_", "_", "_", "_", "_", "_"
    ])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df.columns = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    df["Datetime"] = pd.to_datetime(df["Timestamp"], unit='ms')
    df.set_index("Datetime", inplace=True)
    df = df.astype(float)
    return df

# === BUILD FEATURES (MUST MATCH TRAINING) ===
WINDOWS = [5, 15, 30, 60, 120, 180, 360]

def build_features(window, current):
    features = {}
    for lag in [1, 2, 3, 5, 10, 30, 60, 120, 180, 360]:
        features[f"lag_{lag}"] = window["Close"].iloc[-lag]
    for w in WINDOWS:
        features[f"roll_mean_{w}"] = window["Close"].rolling(w).mean().iloc[-1]
        features[f"roll_std_{w}"] = window["Close"].rolling(w).std().iloc[-1]
        features[f"roll_min_{w}"] = window["Close"].rolling(w).min().iloc[-1]
        features[f"roll_max_{w}"] = window["Close"].rolling(w).max().iloc[-1]
        features[f"roll_median_{w}"] = window["Close"].rolling(w).median().iloc[-1]
        features[f"roll_skew_{w}"] = window["Close"].rolling(w).skew().iloc[-1]
        features[f"roll_kurt_{w}"] = window["Close"].rolling(w).kurt().iloc[-1]
    for p in WINDOWS:
        features[f"pct_change_{p}"] = window["Close"].pct_change(p).iloc[-1]
    for w in WINDOWS:
        v_mean = window["Volume"].rolling(w).mean().iloc[-1]
        features[f"vol_mean_{w}"] = v_mean
        features[f"vol_std_{w}"] = window["Volume"].rolling(w).std().iloc[-1]
        features[f"vol_spike_{w}"] = window["Volume"].iloc[-1] / v_mean if v_mean != 0 else 0
    features["rsi"] = ta.momentum.RSIIndicator(window["Close"], window=14).rsi().iloc[-1]
    macd = ta.trend.MACD(window["Close"])
    features["macd"] = macd.macd().iloc[-1]
    features["macd_signal"] = macd.macd_signal().iloc[-1]
    features["macd_diff"] = macd.macd_diff().iloc[-1]
    boll = ta.volatility.BollingerBands(close=window["Close"], window=20)
    features["bollinger_mavg"] = boll.bollinger_mavg().iloc[-1]
    features["bollinger_hband"] = boll.bollinger_hband().iloc[-1]
    features["bollinger_lband"] = boll.bollinger_lband().iloc[-1]
    return features

# === MAIN EXECUTION ===
def main():
    LOOKBACK = 360
    df = fetch_binance_ohlcv()
    if len(df) < LOOKBACK + 1:
        print("Not enough data.")
        return

    # Use second to last row to avoid using an in-progress candle
    window = df.iloc[-(LOOKBACK + 1):-1].copy()
    current = df.iloc[-2]
    features = build_features(window, current)

    X = pd.DataFrame([features])
    dmatrix = xgb.DMatrix(X)

    model = xgb.Booster()
    model.load_model(MODEL_PATH)

    raw_pred = model.predict(dmatrix)[0]
    pred = np.sign(raw_pred) * (raw_pred ** 2)
    shap_strength = np.sum(np.abs(model.predict(dmatrix, pred_contribs=True)), axis=1)[0]

    # Format output
    price_change_pct = 2 * pred
    confidence_pct = 20 * shap_strength

    msg = (
        f"Signal (BTC/USDT)\n"
        f"Time: {current.name}\n"
        f"Current price: {current['Close']:.2f} USDT\n"
        f"Predicted price change: {price_change_pct:+.2f}%\n"
        f"Confidence: {confidence_pct:.2f}%"
    )
    send_telegram_message(msg)

# === RUN ONCE ===
if __name__ == "__main__":
    main()
