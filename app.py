import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from vnstock import stock_historical_data
import ta
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="VN30F Terminal PRO MAX", page_icon="⚡", layout="wide")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; background-color: #0a0e1a; color: #e2e8f0; }
    .stApp { background: #0a0e1a; }
    .score-card { padding: 15px; border-radius: 8px; font-weight: bold; text-align: center; border: 2px solid; margin-bottom: 10px; }
    .box { background: #111827; border: 1px solid #1e2d4a; border-radius: 8px; padding: 12px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA ENGINE (vnstock 0.2.8.2)
# ─────────────────────────────────────────────
@st.cache_data(ttl=20)
def fetch_data(symbol, tf_minutes, days=5):
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        df = stock_historical_data(symbol=symbol, start_date=start_date, end_date=end_date, resolution=str(tf_minutes), type='derivative')
        if df is not None and not df.empty:
            df['time'] = pd.to_datetime(df['time'])
            return df.sort_values('time').reset_index(drop=True)
    except: pass
    return pd.DataFrame()

# ─────────────────────────────────────────────
# INDICATORS & PATTERNS ENGINE
# ─────────────────────────────────────────────
def apply_all_logic(df):
    if df.empty: return df
    
    # --- 1. Chỉ báo xu hướng & Momentum (Thư viện ta) ---
    df['ema9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
    df['ema50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    
    macd = ta.trend.MACD(df['close'])
    df['macd_hist'] = macd.macd_diff()
    
    rsi = ta.momentum.RSIIndicator(df['close'])
    df['rsi'] = rsi.rsi()
    
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
    df['adx'] = adx.adx()
    df['di_pos'] = adx.adx_pos()
    df['di_neg'] = adx.adx_neg()
    
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = bb.bollinger_wband()
    
    df['vol_ma'] = df['volume'].rolling(20).mean()
    
    # --- 2. Nâng cấp 4: VWAP ---
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (df['tp'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # --- 3. Nâng cấp 3: 6 Mẫu nến quan trọng ---
    body = abs(df['close'] - df['open'])
    rng = df['high'] - df['low']
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
    
    # Doji & Marubozu
    df['is_doji'] = body <= (rng * 0.1)
    df['is_marubozu'] = body >= (rng * 0.9)
    
    # Hammer & Shooting Star
    df['is_hammer'] = (lower_shadow >= 2 * body) & (upper_shadow <= 0.1 * rng) & (rng > 0)
    df['is_shooting_star'] = (upper_shadow >= 2 * body) & (lower_shadow <= 0.1 * rng) & (rng > 0)
    
    # Engulfing
    df['is_bull_engulfing'] = (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1)) & (df['close'].shift(1) < df['open'].shift(1))
    df['is_bear_engulfing'] = (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1)) & (df['close'].shift(1) > df['open'].shift(1))
    
    return df

# ─────────────────────────────────────────────
# SCORING & PREDICTION (Nâng cấp 1 & 2)
# ─────────────────────────────────────────────
def analyze_confluence_and_predict(df1, df5):
    l1 = df1.iloc[-1]
    l5 = df5.iloc[-1]
    score = 0
    breakdown = []
    
    # ==========================================
    # NÂNG CẤP 1: BẢNG TRỌNG SỐ CONFLUENCE
    # ==========================================
    # 1. ADX > 35 & DI
    if l1['adx'] > 35:
        if l1['di_pos'] > l1['di_neg']: score += 25; breakdown.append("🟢 ADX > 35 & DI+ cắt lên (+25)")
        else: score -= 25; breakdown.append("🔴 ADX > 35 & DI- cắt lên (-25)")
            
    # 2. EMA9 > 21 > 50
    if l1['ema9'] > l1['ema21'] > l1['ema50']: score += 15; breakdown.append("🟢 EMA Hướng lên (+15)")
    elif l1['ema9'] < l1['ema21'] < l1['ema50']: score -= 15; breakdown.append("🔴 EMA Hướng xuống (-15)")
        
    # 3. MACD Hist
    if l1['macd_hist'] > 0 and l1['macd_hist'] > df1['macd_hist'].iloc[-2]: score += 15; breakdown.append("🟢 MACD Hist dương tăng (+15)")
    elif l1['macd_hist'] < 0 and l1['macd_hist'] < df1['macd_hist'].iloc[-2]: score -= 15; breakdown.append("🔴 MACD Hist âm giảm (-15)")
        
    # 4. RSI (40-60 trong Uptrend/Downtrend)
    if (l1['ema9'] > l1['ema50']) and (40 <= l1['rsi'] <= 60): score += 10; breakdown.append("🟢 RSI 40-60 Uptrend (+10)")
    elif (l1['ema9'] < l1['ema50']) and (40 <= l1['rsi'] <= 60): score -= 10; breakdown.append("🔴 RSI 40-60 Downtrend (-10)")
        
    # 5. BB Breakout + Volume
    if (l1['close'] > l1['bb_upper']) and (l1['volume'] > l1['vol_ma']): score += 20; breakdown.append("🟢 BB Breakout Lên + Vol lớn (+20)")
    elif (l1['close'] < l1['bb_lower']) and (l1['volume'] > l1['vol_ma']): score -= 20; breakdown.append("🔴 BB Breakout Xuống + Vol lớn (-20)")
        
    # 6. Mẫu Nến Engulfing
    if l1['is_bull_engulfing']: score += 15; breakdown.append("🟢 Nến Bullish Engulfing (+15)")
    if l1['is_bear_engulfing']: score -= 15; breakdown.append("🔴 Nến Bearish Engulfing (-15)")
        
    # 7. Volume > 2x MA
    if l1['volume'] > 2 * l1['vol_ma']:
        score += 10 if l1['close'] > l1['open'] else -10
        breakdown.append(f"{'🟢' if l1['close'] > l1['open'] else '🔴'} Volume > 2x MA ({'+10' if l1['close'] > l1['open'] else '-10'})")
        
    # 8. Đồng thuận 1P & 5P
    ema_1p_bull = l1['ema9'] > l1['ema21']
    ema_5p_bull = l5['ema9'] > l5['ema21']
    if ema_1p_bull == ema_5p_bull:
        score += 20 if ema_1p_bull else -20
        breakdown.append(f"{'🟢' if ema_1p_bull else '🔴'} Đồng thuận khung 5P & 1P ({'+20' if ema_1p_bull else '-20'})")
        
    # 9. Phân kỳ RSI (5 nến gần nhất)
    # Check nhanh: Giá tạo đáy thấp hơn nhưng RSI tạo đáy cao hơn (Bullish)
    prices = df1['close'].tail(5).values
    rsis = df1['rsi'].tail(5).values
    if prices[-1] < prices[0] and rsis[-1] > rsis[0]:
        score += 20; breakdown.append("🟢 Phân kỳ RSI Dương (+20)")
    elif prices[-1] > prices[0] and rsis[-1] < rsis[0]:
        score -= 20; breakdown.append("🔴 Phân kỳ RSI Âm (-20)")
        
    # Giới hạn điểm -100 đến 100
    score = max(min(score, 100), -100)

    # ==========================================
    # NÂNG CẤP 2: DỰ BÁO 3-5 PHIÊN TỚI
    # ==========================================
    predictions = []
    
    # Yếu tố 1: Momentum tích lũy (ADX Trend)
    adx_trend = df5['adx'].diff().tail(3)
    if (adx_trend > 0).all(): predictions.append("⚡ Momentum (ADX): Xu hướng đang hình thành mạnh, sắp có bứt phá lớn.")
    
    # Yếu tố 2: Phân kỳ
    if prices[-1] < prices[0] and rsis[-1] > rsis[0]: predictions.append("✨ RSI Divergence: Năng lượng giảm cạn kiệt, xác suất cao đảo chiều TĂNG.")
    elif prices[-1] > prices[0] and rsis[-1] < rsis[0]: predictions.append("⚠️ RSI Divergence: Năng lượng tăng cạn kiệt, xác suất cao đảo chiều GIẢM.")
        
    # Yếu tố 3: BB Squeeze + EMA
    sqz_threshold = df5['bb_width'].tail(100).quantile(0.15)
    if l5['bb_width'] < sqz_threshold:
        dir_sqz = "LÊN" if l5['ema9'] > l5['ema21'] else "XUỐNG"
        predictions.append(f"🌊 BB Squeeze: Đang nén giá mạnh. Xác suất cao bung {dir_sqz} (theo EMA).")
        
    # Yếu tố 4: Volume accumulation (5 nến)
    last_5 = df5.tail(5)
    green_vol = last_5[last_5['close'] >= last_5['open']]['volume'].sum()
    red_vol = last_5[last_5['close'] < last_5['open']]['volume'].sum()
    if green_vol > red_vol * 1.5: predictions.append("📈 Volume: Lực mua đang tích lũy (Gom hàng).")
    elif red_vol > green_vol * 1.5: predictions.append("📉 Volume: Lực bán đang áp đảo (Xả hàng).")
        
    # Yếu tố 5: MACD Slope
    macd_slope = df5['macd_hist'].diff().iloc[-1]
    if l5['macd_hist'] < 0 and macd_slope > 0: predictions.append("🔄 MACD Slope: Động lượng đang đảo chiều TĂNG từ dưới lên.")
    elif l5['macd_hist'] > 0 and macd_slope < 0: predictions.append("🔄 MACD Slope: Động lượng đang suy yếu, chuẩn bị GIẢM.")

    return score, breakdown, predictions

# ─────────────────────────────────────────────
# UI TỔNG HỢP
# ─────────────────────────────────────────────
st.sidebar.title("⚡ VN30F ULTRA")
symbol = st.sidebar.selectbox("Mã hợp đồng", ["VN30F1M", "VN30F2M"])
auto_refresh = st.sidebar.checkbox("Tự động làm mới", value=True)

df1 = fetch_data(symbol, 1)
df5 = fetch_data(symbol, 5)

if not df1.empty and not df5.empty:
    df1 = apply_all_logic(df1)
    df5 = apply_all_logic(df5)
    
    score, breakdown, predictions = analyze_confluence_and_predict(df1, df5)
    
    c1, c2 = st.columns([1.2, 2])
    
    with c1:
        # Gauge Chart Thể hiện Confluence Score
        fig_score = go.Figure(go.Indicator(
            mode = "gauge+number", value = score,
            gauge = {'axis': {'range': [-100, 100]},
                     'bar': {'color': "#e2e8f0"},
                     'steps': [
                         {'range': [-100, -70], 'color': "#ff5252"},
                         {'range': [-70, -30], 'color': "#7f1d1d"},
                         {'range': [-30, 30], 'color': "#1e293b"},
                         {'range': [30, 70], 'color': "#064e3b"},
                         {'range': [70, 100], 'color': "#00e676"}]},
            title = {'text': "CONFLUENCE SCORE", 'font': {'size': 14, 'color': '#38bdf8'}}
        ))
        fig_score.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=280, margin=dict(t=40, b=0, l=10, r=10))
        st.plotly_chart(fig_score, use_container_width=True)
        
        # Signal Box
        if score >= 70: st.markdown('<div class="score-card" style="color:#00e676; border-color:#00e676; background:rgba(0,230,118,0.1)">🚀 KHUYẾN NGHỊ LONG MẠNH</div>', unsafe_allow_html=True)
        elif score <= -70: st.markdown('<div class="score-card" style="color:#ff5252; border-color:#ff5252; background:rgba(255,82,82,0.1)">💥 KHUYẾN NGHỊ SHORT MẠNH</div>', unsafe_allow_html=True)
        else: st.markdown('<div class="score-card" style="color:#ffd600; border-color:#ffd600; background:rgba(255,214,0,0.1)">🔄 SIDEWAY / KHÔNG RÕ HƯỚNG</div>', unsafe_allow_html=True)

    with c2:
        tab_pred, tab_breakdown = st.tabs(["🔮 Dự báo 3-5 phiên tới", "⚙️ Chi tiết cộng điểm (Score)"])
        with tab_pred:
            st.markdown("<div class='box'>", unsafe_allow_html=True)
            if predictions:
                for p in predictions: st.markdown(f"**{p}**")
            else:
                st.write("Thị trường đang đi ngang, chưa có tín hiệu gom hàng hay nén giá.")
            
            # Cảnh báo nến ở nến cuối cùng
            l1 = df1.iloc[-1]
            st.markdown("---")
            st.write(f"**📍 VWAP hiện tại:** {l1['vwap']:.1f} (Giá đóng cửa: {l1['close']:.1f})")
            if l1['is_hammer']: st.write("🔨 **Nến hiện tại:** Đang hình thành Hammer (Hỗ trợ tốt)")
            elif l1['is_shooting_star']: st.write("🌠 **Nến hiện tại:** Đang hình thành Shooting Star (Kháng cự mạnh)")
            elif l1['is_doji']: st.write("⚖️ **Nến hiện tại:** Doji (Dấu hiệu do dự)")
            elif l1['is_marubozu']: st.write("🚀 **Nến hiện tại:** Marubozu (Động lượng rất mạnh)")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with tab_breakdown:
            st.markdown("<div class='box'>", unsafe_allow_html=True)
            if breakdown:
                for b in breakdown: st.markdown(f"{b}")
            else: st.write("Chưa có chỉ báo nào thỏa mãn điều kiện cộng/trừ điểm.")
            st.markdown("</div>", unsafe_allow_html=True)

    # Biểu đồ chính
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=df1['time'], open=df1['open'], high=df1['high'], low=df1['low'], close=df1['close'], name="VN30F1M"), row=1, col=1)
    # Thêm VWAP vào chart
    fig.add_trace(go.Scatter(x=df1['time'], y=df1['vwap'], line=dict(color='#ffd600', width=2), name="VWAP"), row=1, col=1)
    # Thêm RSI
    fig.add_trace(go.Scatter(x=df1['time'], y=df1['rsi'], line=dict(color='#38bdf8'), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
    
    fig.update_layout(template="plotly_dark", height=550, margin=dict(t=20, b=0, l=0, r=0), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Lỗi dữ liệu. Vui lòng đợi API phản hồi...")

if auto_refresh:
    time.sleep(15)
    st.rerun()
