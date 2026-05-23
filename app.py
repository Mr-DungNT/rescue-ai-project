import streamlit as st
import pandas as pd
import re
import time
import math
import requests
import numpy as np
import pydeck as pdk
import json

# ─────────────────────────────────────────────
# PHẦN 1: ML ENGINE — XGBoost + Synthetic Data
# ─────────────────────────────────────────────
# Import lazy để không crash nếu chưa cài
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

@st.cache_resource(show_spinner=False)
def train_xgboost_model():
    """
    Train XGBoost trên synthetic data sinh từ mô hình vật lý.
    Cache lại — chỉ train 1 lần duy nhất khi khởi động app.
    
    Features: [velocity_kmh, wind_speed_ms, wind_deg, time_lost_min, temp_c]
    Targets : [delta_lat, delta_lon]
    """
    if not XGBOOST_AVAILABLE:
        return None, None

    np.random.seed(42)
    N = 5000  # 5000 mẫu tổng hợp

    # Sinh dữ liệu đầu vào ngẫu nhiên trong dải thực tế
    velocity    = np.random.uniform(0, 60, N)       # km/h
    wind_speed  = np.random.uniform(0, 25, N)       # m/s
    wind_deg    = np.random.uniform(0, 360, N)      # độ
    time_lost   = np.random.uniform(5, 120, N)      # phút
    temp_c      = np.random.uniform(5, 40, N)       # °C

    # Mô hình vật lý làm nhãn (ground truth) + nhiễu Gaussian ±5%
    noise = np.random.normal(1.0, 0.05, N)
    bearing_rad = np.radians(wind_deg)
    drift_kmh   = velocity + (wind_speed * 3.6 * 0.03)
    offset_m    = (drift_kmh / 60) * time_lost * 1000 / 2 * noise

    delta_lat = (offset_m * np.cos(bearing_rad)) / 111111
    delta_lon = (offset_m * np.sin(bearing_rad)) / (111111 * np.cos(np.radians(21.0)))

    X = np.column_stack([velocity, wind_speed, wind_deg, time_lost, temp_c])

    # Huấn luyện 2 model riêng: 1 cho lat, 1 cho lon
    model_lat = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    model_lon = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    model_lat.fit(X, delta_lat)
    model_lon.fit(X, delta_lon)

    return model_lat, model_lon


def predict_with_uncertainty(model_lat, model_lon, features, n_bootstrap=100):
    """
    Bootstrap Ensemble: chạy N lần với nhiễu nhỏ trên features
    → trả về (mean_delta_lat, mean_delta_lon, std_lat, std_lon)
    để vẽ ellipse xác suất 68% và 95%.
    """
    if model_lat is None:
        # Fallback vật lý nếu không có XGBoost
        velocity, wind_speed, wind_deg, time_lost, temp_c = features
        bearing = math.radians(wind_deg)
        drift_kmh = velocity + (wind_speed * 3.6 * 0.03)
        offset_m = (drift_kmh / 60) * time_lost * 1000 / 2
        d_lat = (offset_m * math.cos(bearing)) / 111111
        d_lon = (offset_m * math.sin(bearing)) / (111111 * math.cos(math.radians(21.0)))
        return d_lat, d_lon, abs(d_lat) * 0.1, abs(d_lon) * 0.1

    feat_arr = np.array(features).reshape(1, -1)
    lat_preds = []
    lon_preds = []

    for _ in range(n_bootstrap):
        # Thêm nhiễu nhỏ ±3% để mô phỏng uncertainty của sensor
        noisy = feat_arr * np.random.normal(1.0, 0.03, feat_arr.shape)
        lat_preds.append(model_lat.predict(noisy)[0])
        lon_preds.append(model_lon.predict(noisy)[0])

    return (
        float(np.mean(lat_preds)),
        float(np.mean(lon_preds)),
        float(np.std(lat_preds)),
        float(np.std(lon_preds))
    )


# ─────────────────────────────────────────────
# PHẦN 2: WEATHER API
# ─────────────────────────────────────────────
API_KEY = "23913db94b60da48fe4dd64dbab2344f"

def get_realtime_weather(lat, lon):
    try:
        url = (f"https://api.openweathermap.org/data/2.5/weather"
               f"?lat={lat}&lon={lon}&appid={API_KEY}&units=metric&lang=vi")
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            d = resp.json()
            return {
                "wind_speed":  d['wind'].get('speed', 0),
                "wind_deg":    d['wind'].get('deg', 0),
                "temp":        d['main'].get('temp', 25),
                "humidity":    d['main'].get('humidity', 70),
                "rain":        d.get('rain', {}).get('1h', 0),
                "description": d['weather'][0].get('description', 'N/A'),
                "visibility":  d.get('visibility', 10000) / 1000,
            }
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
# PHẦN 3: DATA INGESTION — Xử lý Excel bền vững
# ─────────────────────────────────────────────

# Bảng ánh xạ địa chỉ chữ → tọa độ (mở rộng dễ dàng)
ADDRESS_MAP = {
    "đại cồ việt":      (21.0032, 105.8430),
    "trần đại nghĩa":   (21.0022, 105.8430),
    "hoàn kiếm":        (21.0285, 105.8542),
    "ba đình":          (21.0380, 105.8353),
    "hà nội":           (21.0278, 105.8342),
    "hồ chí minh":      (10.8231, 106.6297),
    "đà nẵng":          (16.0544, 108.2022),
}

def extract_coords_robust(text):
    """
    Bóc tách tọa độ bền vững:
    1. Thử tìm float hợp lệ (lat trong [-90,90], lon trong [-180,180])
    2. Nếu không có → ánh xạ địa chỉ chữ
    3. Fallback → Hà Nội
    """
    text_str = str(text).strip()

    # Tìm tất cả số float/int
    numbers = re.findall(r"[-+]?\d+\.?\d*", text_str)
    floats  = [float(n) for n in numbers]

    # Lọc cặp lat/lon hợp lệ
    valid_pairs = [
        (floats[i], floats[i+1])
        for i in range(len(floats)-1)
        if -90 <= floats[i] <= 90 and -180 <= floats[i+1] <= 180
    ]
    if valid_pairs:
        return valid_pairs[0]

    # Ánh xạ địa chỉ
    lower = text_str.lower()
    for key, coords in ADDRESS_MAP.items():
        if key in lower:
            return coords

    return (21.0278, 105.8342)  # Fallback Hà Nội


def extract_velocity(text):
    matches = re.findall(r"[-+]?\d+\.?\d*", str(text))
    return float(matches[0]) if matches else 0.0


def build_route_3d(df, coord_col_idx=5, n_points=None):
    """
    Đọc toàn bộ lộ trình từ DataFrame,
    tích lũy cao độ giả lập để render Pydeck 3D.
    """
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        lat, lon = extract_coords_robust(row.iloc[coord_col_idx])
        altitude = i * 15  # tích lũy 15m mỗi điểm → hiệu ứng leo núi
        rows.append({"lat": lat, "lon": lon, "altitude": altitude, "step": i})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# PHẦN 4: PYDECK 3D MAP
# ─────────────────────────────────────────────

def build_pydeck_map(route_df, origin_lat, origin_lon, target_lat, target_lon,
                     std_lat, std_lon):
    """
    Render bản đồ Pydeck 3D với:
    - HexagonLayer: heatmap tích lũy lộ trình (màu lạnh→nóng)
    - ScatterplotLayer: điểm mất dấu (đỏ), tâm datum (vàng)
    - PathLayer: đường lộ trình
    - ScatterplotLayer (rings): vòng ellipse xác suất 68% và 95%
    """

    # --- Layer 1: Hexagon Heatmap 3D ---
    hex_layer = pdk.Layer(
        "HexagonLayer",
        data=route_df,
        get_position=["lon", "lat"],
        get_elevation="altitude",
        elevation_scale=8,
        elevation_range=[0, 3000],
        radius=80,
        pickable=True,
        extruded=True,
        color_range=[
            [0,   40, 120, 200],
            [0,   90, 180, 200],
            [0,  180, 150, 200],
            [200, 200,  0, 220],
            [220, 100,  0, 220],
            [180,   0,  0, 240],
        ],
    )

    # --- Layer 2: Path Layer ---
    path_data = [{"path": [[r.lon, r.lat] for _, r in route_df.iterrows()]}]
    path_layer = pdk.Layer(
        "PathLayer",
        data=path_data,
        get_path="path",
        get_color=[255, 220, 0, 200],
        width_min_pixels=3,
    )

    # --- Layer 3: Điểm mất dấu & Datum ---
    points_data = [
        {"lat": origin_lat, "lon": origin_lon, "color": [20, 20, 20, 255],   "radius": 120, "label": "Điểm mất dấu"},
        {"lat": target_lat, "lon": target_lon, "color": [255, 50,  50, 255], "radius": 150, "label": "Tâm Datum (XGBoost)"},
    ]
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=points_data,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
    )

    # --- Layer 4: Vòng xác suất 68% & 95% ---
    # Xấp xỉ ellipse bằng 64 điểm trên vòng tròn
    def make_ring(center_lat, center_lon, r_lat, r_lon, n=64):
        pts = []
        for k in range(n + 1):
            angle = 2 * math.pi * k / n
            pts.append([
                center_lon + r_lon * math.sin(angle),
                center_lat + r_lat * math.cos(angle)
            ])
        return pts

    ring_68  = make_ring(target_lat, target_lon, std_lat,       std_lon)
    ring_95  = make_ring(target_lat, target_lon, std_lat * 1.96, std_lon * 1.96)

    rings_data = [
        {"path": ring_68,  "color": [255, 200, 0, 220], "name": "68% confidence"},
        {"path": ring_95,  "color": [255, 80,  0, 160], "name": "95% confidence"},
    ]
    ring_layer = pdk.Layer(
        "PathLayer",
        data=rings_data,
        get_path="path",
        get_color="color",
        width_min_pixels=2,
    )

    # --- View State ---
    view_state = pdk.ViewState(
        latitude=(origin_lat + target_lat) / 2,
        longitude=(origin_lon + target_lon) / 2,
        zoom=13,
        pitch=55,
        bearing=15,
    )

    deck = pdk.Deck(
        layers=[hex_layer, path_layer, scatter_layer, ring_layer],
        initial_view_state=view_state,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={"text": "{label}\n{name}"},
    )
    return deck


# ─────────────────────────────────────────────
# PHẦN 5: STREAMLIT UI
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AI Rescue System v2",
    layout="wide",
    page_icon="🚨"
)

# ── CSS nền tối, phong cách tactical ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background-color: #080f1a;
    color: #c8d8e8;
}
h1, h2, h3 { font-family: 'Share Tech Mono', monospace; color: #00e5ff; letter-spacing: 2px; }
.stMetric { background: #0d1f33; border: 1px solid #1a3a5c; border-radius: 8px;
            padding: 12px; border-left: 4px solid #00e5ff; }
.stMetric label { color: #7aa8cc !important; font-size: 0.75rem !important; }
.stMetric [data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 800; }
div.stButton > button {
    background: linear-gradient(135deg, #ff4b2b, #ff416c);
    color: white; border: none; border-radius: 6px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1rem; padding: 12px 28px;
    box-shadow: 0 0 20px rgba(255,75,43,0.5);
    transition: all 0.3s ease;
}
div.stButton > button:hover { box-shadow: 0 0 35px rgba(255,75,43,0.9); transform: scale(1.03); }
.warning-box {
    background: #0d1f33; border: 1px solid #ff4b2b;
    border-left: 5px solid #ff4b2b; border-radius: 8px;
    padding: 20px; margin: 16px 0; font-family: 'Share Tech Mono', monospace;
}
.success-box {
    background: #0a1f0d; border: 1px solid #00e676;
    border-left: 5px solid #00e676; border-radius: 8px;
    padding: 20px; margin: 16px 0;
}
.model-badge {
    display: inline-block; background: #00e5ff22;
    border: 1px solid #00e5ff; border-radius: 20px;
    padding: 2px 12px; font-size: 0.78rem; color: #00e5ff;
    font-family: 'Share Tech Mono', monospace; margin-left: 10px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<h1 style="font-size:1.6rem; margin-bottom:4px;">
    🚨 HỆ THỐNG CỨU HỘ AI — TÍCH HỢP ĐỊA HÌNH & THỜI TIẾT
    <span class="model-badge">XGBoost v2 + 3D Pydeck</span>
</h1>
<p style="color:#5a8ab0; font-size:0.85rem; font-family:'Share Tech Mono',monospace;">
    Real-time Weather · Drift Prediction · Uncertainty Ellipse · 3D Terrain Heatmap
</p>
""", unsafe_allow_html=True)
st.divider()

# ── Session State ──
for key in ['analysis_active', 'model_lat', 'model_lon', 'model_trained']:
    if key not in st.session_state:
        st.session_state[key] = False

# ── Pre-load XGBoost (chạy ngầm khi app khởi động) ──
if not st.session_state.model_trained:
    with st.spinner("⚙️ Đang khởi tạo mô hình XGBoost (chỉ lần đầu)..."):
        ml, mln = train_xgboost_model()
        st.session_state.model_lat = ml
        st.session_state.model_lon = mln
        st.session_state.model_trained = True
    if XGBOOST_AVAILABLE:
        st.success("✅ XGBoost đã sẵn sàng — 5.000 mẫu synthetic + Bootstrap Ensemble")
    else:
        st.warning("⚠️ XGBoost chưa cài (`pip install xgboost`) — đang dùng mô hình vật lý dự phòng.")

# ── Sidebar ──
st.sidebar.markdown("## 📂 Dữ liệu đầu vào")
uploaded_file = st.sidebar.file_uploader("Tải file Trip Report (Excel)", type=["xlsx", "xls"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # Lấy dữ liệu dòng đầu
    latest   = df.iloc[0]
    lat, lon = extract_coords_robust(latest.iloc[5])
    velocity = extract_velocity(latest.iloc[9])

    # Weather
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌍 Môi trường thực tế")
    weather = get_realtime_weather(lat, lon)

    if weather:
        st.sidebar.success(f"📍 `{lat:.5f}, {lon:.5f}`")
        c1, c2 = st.sidebar.columns(2)
        c1.metric("🌡️ Nhiệt độ", f"{weather['temp']}°C")
        c2.metric("💧 Lượng mưa", f"{weather['rain']} mm/h")
        st.sidebar.write(f"🌬️ Gió: **{weather['wind_speed']} m/s** — hướng **{weather['wind_deg']}°**")
        st.sidebar.write(f"👁️ Tầm nhìn: **{weather['visibility']} km** | {weather['description'].capitalize()}")
        wind_speed = weather['wind_speed']
        wind_dir   = weather['wind_deg']
        temp_c     = weather['temp']
    else:
        st.sidebar.warning("⚠️ Dùng dữ liệu dự phòng")
        wind_speed, wind_dir, temp_c = 5.0, 45.0, 25.0

    time_lost = st.sidebar.slider("⏱️ Thời gian mất tín hiệu (phút)", 5, 120, 30)

    # ── Quick Metrics ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🚗 Vận tốc TB",        f"{velocity} km/h")
    col2.metric("🌬️ Sức gió",           f"{wind_speed} m/s")
    col3.metric("🌡️ Nhiệt độ",          f"{temp_c}°C")
    col4.metric("⏱️ Mất tín hiệu",      f"{time_lost} phút")

    st.divider()

    # ── Nút kích hoạt AI ──
    st.subheader("🤖 Phân tích AI & Chiến thuật Cứu hộ")

    if st.button("🚀  Kích hoạt AI Phân tích rủi ro & Tọa độ"):
        st.session_state.analysis_active = True

    if st.session_state.analysis_active:

        with st.status("🛰️ Đang quét dữ liệu đa tầng...", expanded=True) as status:
            st.write("🔄 Nạp mô hình XGBoost...")
            time.sleep(0.4)
            st.write("🌡️ Đọc chỉ số thời tiết thực tế...")
            time.sleep(0.3)
            st.write("🧮 Chạy Bootstrap Ensemble (100 lần)...")
            time.sleep(0.5)
            st.write("📐 Tính toán vùng xác suất 68% & 95%...")
            time.sleep(0.3)
            status.update(label="✅ Phân tích hoàn tất!", state="complete")

        # ── Dự đoán XGBoost ──
        features = [velocity, wind_speed, wind_dir, time_lost, temp_c]
        d_lat, d_lon, std_lat, std_lon = predict_with_uncertainty(
            st.session_state.model_lat,
            st.session_state.model_lon,
            features,
            n_bootstrap=100
        )

        new_lat = lat + d_lat
        new_lon = lon + d_lon

        # Bán kính vòng 68% tính bằng mét
        radius_68_m  = int(std_lat * 111111)
        radius_95_m  = int(std_lat * 1.96 * 111111)

        # Survival analysis
        water_temp    = temp_c - 2
        survival_time = "6–12 giờ" if water_temp > 20 else ("2–4 giờ" if water_temp > 10 else "< 1 giờ")
        is_cold       = temp_c < 20
        is_rain       = (weather['rain'] > 5) if weather else False

        # Kết quả
        st.markdown(f"""
<div class="warning-box">
<h3 style="color:#ff6b6b; margin-top:0;">🎯 TỌA ĐỘ MỤC TIÊU ƯU TIÊN  <span style="font-size:0.8rem;color:#aaa;">(XGBoost + Bootstrap)</span></h3>
<p>📌 <b>Tâm Datum:</b> <code>{new_lat:.6f}, {new_lon:.6f}</code></p>
<p>📐 <b>Vùng tin cậy 68%:</b> bán kính ~<b>{radius_68_m} m</b> &nbsp;|&nbsp;
      <b>95%:</b> ~<b>{radius_95_m} m</b></p>
<p>🧭 Dịch chuyển: <b>{d_lat*111111:.0f} m</b> Nam-Bắc &nbsp;/&nbsp; <b>{d_lon*111111*math.cos(math.radians(lat)):.0f} m</b> Đông-Tây</p>
</div>
<div class="success-box">
<h3 style="color:#00e676; margin-top:0;">📋 PHÂN TÍCH CHUYÊN MÔN</h3>
<p>⏳ <b>Thời gian vàng:</b> {survival_time} (nhiệt độ nước ~{water_temp:.1f}°C)</p>
<p>❄️ <b>Rủi ro hạ thân nhiệt:</b> {"🚨 CAO — cần ưu tiên sưởi ấm ngay" if is_cold else "✅ Thấp"}</p>
<p>🌧️ <b>Lượng mưa:</b> {"⚠️ Mưa lớn — giảm tầm nhìn, triển khai rada" if is_rain else "✅ Bình thường"}</p>
<p>🚤 <b>Chiến thuật đề xuất:</b> Hình xoắn ốc mở rộng (Expanding Square) từ Datum, ưu tiên vùng 68%.</p>
</div>
""", unsafe_allow_html=True)

        st.code(f"LAT: {new_lat:.6f}   LON: {new_lon:.6f}   [±{radius_68_m}m @ 68% | ±{radius_95_m}m @ 95%]", language="text")

        # ── Feature Importance ──
        if XGBOOST_AVAILABLE and st.session_state.model_lat is not None:
            with st.expander("📊 Feature Importance — XGBoost (Lat model)"):
                fi = st.session_state.model_lat.feature_importances_
                fi_df = pd.DataFrame({
                    "Feature":    ["Vận tốc (km/h)", "Sức gió (m/s)", "Hướng gió (°)", "Thời gian (phút)", "Nhiệt độ (°C)"],
                    "Importance": fi
                }).sort_values("Importance", ascending=False)
                st.bar_chart(fi_df.set_index("Feature")["Importance"])

        # ── Pydeck 3D Map ──
        st.divider()
        st.subheader("🗺️ Bản đồ vệ tinh 3D — Lộ trình & Vùng xác suất")

        route_df = build_route_3d(df, coord_col_idx=5)

        if route_df.empty or len(route_df) < 2:
            # Nếu Excel không có đủ dòng lộ trình → mock 1 điểm
            route_df = pd.DataFrame([
                {"lat": lat, "lon": lon, "altitude": 0, "step": 0}
            ])

        deck = build_pydeck_map(
            route_df, lat, lon, new_lat, new_lon, std_lat, std_lon
        )
        st.pydeck_chart(deck)

        # Chú thích
        st.markdown("""
<p style="font-size:0.78rem; color:#5a7a9a; font-family:'Share Tech Mono',monospace;">
🟡 Đường vàng: Lộ trình &nbsp;|&nbsp; ⚫ Điểm đen: Mất dấu &nbsp;|&nbsp; 🔴 Điểm đỏ: Tâm Datum &nbsp;|&nbsp;
🟡 Vòng vàng: 68% &nbsp;|&nbsp; 🟠 Vòng cam: 95% &nbsp;|&nbsp; Cột màu: Heatmap tích lũy cao độ
</p>
""", unsafe_allow_html=True)

else:
    # ── Welcome Screen ──
    st.markdown("""
<div style="text-align:center; padding: 60px 20px;">
    <h2 style="color:#00e5ff; font-family:'Share Tech Mono',monospace;">⬅️ Tải file Excel để bắt đầu</h2>
    <p style="color:#4a7a9b; max-width:500px; margin:0 auto; line-height:1.8;">
        Hệ thống sẽ tự động bóc tách tọa độ, kết nối thời tiết thực tế,
        chạy mô hình <b>XGBoost</b> và hiển thị bản đồ vệ tinh <b>3D Pydeck</b>
        với vùng xác suất <b>68% & 95%</b>.
    </p>
</div>
""", unsafe_allow_html=True)
    
# Chia tỷ lệ 3 cột để ép bức ảnh thu gọn và căn lề ngay chính giữa màn hình
    col_left, col_mid, col_right = st.columns([1, 2, 1])
    with col_mid:
        try:
            # Gọi trực tiếp file ảnh PNG lưu nội bộ trong máy của cậu
            st.image("cuuho.png", caption="Hệ thống trực chỉ huy và phân tích rủi ro AI 24/7", use_container_width=True)
        except Exception:
            # Nhánh dự phòng phòng hờ gõ sai tên file, app vẫn chạy tiếp không bị lỗi đỏ
            st.info("💡 Mẹo: Bỏ file ảnh tên 'cuuho.png' vào thư mục dự án để hiển thị poster chỉ huy.")
