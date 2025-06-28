import pandas as pd
import numpy as np
import xgboost as xgb
import ta

# === USER CONFIGURABLE PARAMETERS ===
TEST_BEGIN = 0
TEST_END = 200000
INITIAL_BALANCE = 100.0
LEVERAGE = 4.0
TP_PROFIT_USD = 1.0
SL_SCALE = 0.5
PRED_THRESHOLD = 0.5
SHAP_THRESHOLD = 3.0
TP_PERCENTAGE = 0.01
MIN_TRADE_INTERVAL_MINUTES = 15

# === PERMANENT PARAMETERS ===
FEE_PCT = 0.000315
MAX_TRADE_DURATION_MINUTES = 60
LOOKBACK = 360
WINDOWS = [5, 15, 30, 60, 120, 180, 360]

# === LOAD MODEL ===
model = xgb.Booster()
model.load_model("btc_xgb_model.bin")

# === LOAD DATA ===
df = pd.read_csv("btc_data_test.csv", parse_dates=["Datetime"])
df = df.sort_values("Timestamp").reset_index(drop=True)
df = df.iloc[TEST_BEGIN:min(TEST_END, len(df))].copy()

balance = INITIAL_BALANCE
open_margin = 0.0
trades = []
open_trades = []
caution_mode = False
last_trade_time = None
trade_counter = 1

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
    dt = current["Datetime"]
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

for t in range(LOOKBACK, len(df) - 1):
    current = df.iloc[t]
    dt = current["Datetime"]

    # --- Close open trades ---
    to_close = []
    for trade in open_trades:
        idx_now = t
        entry_time = trade["entry_time"]
        entry_price = trade["entry_price"]
        position_size = trade["position_size"]
        direction = trade["direction"]
        entry_type = trade["type"]
        tp_pct = trade["tp_pct"]
        sl_pct = trade["sl_pct"]
        trade_id = trade["trade_id"]

        high = df.iloc[idx_now]["High"]
        low = df.iloc[idx_now]["Low"]
        close = df.iloc[idx_now]["Close"]
        now_time = df.iloc[idx_now]["Datetime"]

        duration = (now_time - entry_time).total_seconds() / 60
        resolved = False
        pnl_pct = None
        result_type = None

        if direction == 1:
            if high >= entry_price * (1 + tp_pct + FEE_PCT):
                pnl_pct = tp_pct - FEE_PCT
                result_type = f"LONG TP {trade_id}"
                resolved = True
            elif low <= entry_price * (1 + sl_pct - FEE_PCT):
                pnl_pct = sl_pct - FEE_PCT
                result_type = f"LONG SL {trade_id}"
                resolved = True
        else:
            if low <= entry_price * (1 - tp_pct - FEE_PCT):
                pnl_pct = tp_pct - FEE_PCT
                result_type = f"SHORT TP {trade_id}"
                resolved = True
            elif high >= entry_price * (1 - sl_pct + FEE_PCT):
                pnl_pct = sl_pct - FEE_PCT
                result_type = f"SHORT SL {trade_id}"
                resolved = True

        if not resolved and duration > MAX_TRADE_DURATION_MINUTES:
            exit_price = close
            exit_price_adj = exit_price * (1 - FEE_PCT) if direction == 1 else exit_price * (1 + FEE_PCT)
            pnl_pct = direction * ((exit_price_adj - entry_price) / entry_price)
            result_type = f"{entry_type.upper()} TIMEOUT {trade_id}"
            resolved = True

        if resolved:
            profit = position_size * pnl_pct * LEVERAGE
            is_hypothetical = trade.get("hypothetical", False)
            if not is_hypothetical:
                balance += profit
                open_margin -= position_size

            trades.append({
                "type": entry_type,
                "pnl": pnl_pct,
                "profit": profit,
                "size": position_size,
                "leverage": LEVERAGE,
                "entry_shap": trade["entry_shap"],
                "entry_pred": trade["entry_pred"],
                "entry_time": entry_time,
                "exit_time": now_time,
                "balance": balance if not is_hypothetical else "HYPOTHETICAL",
                "result": result_type + (" (HYPOTHETICAL)" if is_hypothetical else "")
            })

            if 'SL' in result_type and not is_hypothetical:
                print(f"[{now_time}] {result_type} | PnL: {pnl_pct:.4f} | Loss: {profit:.2f} | New Balance: ${balance:.2f}")
                caution_mode = True
            elif is_hypothetical and profit > 0 and caution_mode:
                print(f"[{now_time}] EXITING CAUTION MODE after profitable hypothetical trade {trade_id}")
                caution_mode = False
            else:
                print(f"[{now_time}] {result_type} | PnL: {pnl_pct:.4f} | Profit: {profit:.2f} | New Balance: ${balance:.2f}")
            to_close.append(trade)

    open_trades = [tr for tr in open_trades if tr not in to_close]

    # --- Feature Extraction ---
    window = df.iloc[t - LOOKBACK:t].copy()
    features = build_features(window, current)
    X_point = pd.DataFrame([features])
    dmatrix = xgb.DMatrix(X_point)
    raw_pred = model.predict(dmatrix)[0]
    pred = np.sign(raw_pred) * (raw_pred ** 2)

    # === Prediction Threshold ===
    if not (PRED_THRESHOLD <= abs(pred)):
        continue

    shap_strength = np.sum(np.abs(model.predict(dmatrix, pred_contribs=True)), axis=1)[0]

    # === Trade Entry ===
    if shap_strength >= SHAP_THRESHOLD:
        tp_pct = TP_PERCENTAGE
        direction = 1 if pred > 0 else -1
        entry_type = "long" if direction == 1 else "short"
        entry_price = current["Close"]
        entry_time = dt

        net_tp_pct = tp_pct - FEE_PCT
        if net_tp_pct <= 0:
            continue

        position_size = TP_PROFIT_USD / (net_tp_pct * LEVERAGE)
        target_loss = -TP_PROFIT_USD * SL_SCALE
        sl_pct = (target_loss / (position_size * LEVERAGE)) + FEE_PCT

        available_balance = balance - open_margin
        if not caution_mode and position_size > available_balance:
            continue

        can_trade = last_trade_time is None or (dt - last_trade_time).total_seconds() / 60 >= MIN_TRADE_INTERVAL_MINUTES
        filter_ok = True

        if can_trade and filter_ok:
            trade_id = trade_counter
            print(f"[{entry_time}] OPENING {'HYPOTHETICAL ' if caution_mode else ''}{entry_type.upper()} {trade_id} | Pred: {pred:.4f} | SHAP: {shap_strength:.4f}")
            open_trades.append({
                "type": entry_type,
                "direction": direction,
                "entry_price": entry_price,
                "entry_time": entry_time,
                "entry_idx": t,
                "position_size": position_size,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "entry_shap": shap_strength,
                "entry_pred": pred,
                "trade_id": trade_id,
                "hypothetical": caution_mode
            })
            if not caution_mode:
                open_margin += position_size
            last_trade_time = dt
            trade_counter += 1

# === Final forced exits ===
last_dt = df.iloc[-1]["Datetime"]
for trade in open_trades:
    idx_now = len(df) - 1
    entry_time = trade["entry_time"]
    entry_price = trade["entry_price"]
    position_size = trade["position_size"]
    direction = trade["direction"]
    entry_type = trade["type"]
    trade_id = trade["trade_id"]
    is_hypothetical = trade.get("hypothetical", False)
    close = df.iloc[idx_now]["Close"]
    now_time = last_dt
    exit_price_adj = close * (1 - FEE_PCT) if direction == 1 else close * (1 + FEE_PCT)
    pnl_pct = direction * ((exit_price_adj - entry_price) / entry_price)
    profit = position_size * pnl_pct * LEVERAGE
    if not is_hypothetical:
        balance += profit
        open_margin -= position_size
    trades.append({
        "type": entry_type,
        "pnl": pnl_pct,
        "profit": profit,
        "size": position_size,
        "leverage": LEVERAGE,
        "entry_shap": trade["entry_shap"],
        "entry_pred": trade["entry_pred"],
        "entry_time": entry_time,
        "exit_time": now_time,
        "balance": balance if not is_hypothetical else "HYPOTHETICAL",
        "result": f"{entry_type.upper()} FORCED EXIT {trade_id}" + (" (HYPOTHETICAL)" if is_hypothetical else "")
    })
    print(f"[{now_time}] {entry_type.upper()} FORCED EXIT {trade_id} | PnL: {pnl_pct:.4f} | Profit: {profit:.2f} | New Balance: ${balance:.2f}")

# === Summary ===
if trades:
    total_return_pct = (balance - INITIAL_BALANCE) / INITIAL_BALANCE
    avg_profit = np.mean([t['profit'] for t in trades if isinstance(t['balance'], float)])
    win_rate = sum(1 for t in trades if t['profit'] > 0 and isinstance(t['balance'], float)) / \
               sum(1 for t in trades if isinstance(t['balance'], float))
    longs = [t for t in trades if t['type'] == 'long' and isinstance(t['balance'], float)]
    shorts = [t for t in trades if t['type'] == 'short' and isinstance(t['balance'], float)]
    long_winrate = sum(1 for t in longs if t['profit'] > 0) / len(longs) if longs else 0
    short_winrate = sum(1 for t in shorts if t['profit'] > 0) / len(shorts) if shorts else 0
    print("\n--- Trading Summary ---")
    print(f"Initial Balance: ${INITIAL_BALANCE:.2f}")
    print(f"Final Balance: ${balance:.2f}")
    print(f"Total Return: {total_return_pct:.2%}")
    print(f"Total Trades: {len(trades)}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Long Win Rate: {long_winrate:.2%} | Short Win Rate: {short_winrate:.2%}")
else:
    print("No trades executed.")
