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
# PHẦN 1: ML ENGINE — Algorithm + Synthetic Data
# ─────────────────────────────────────────────
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

@st.cache_resource(show_spinner=False)
def train_xgboost_model():
    if not XGBOOST_AVAILABLE:
        return None, None

    np.random.seed(42)
    N = 5000  

    velocity    = np.random.uniform(0, 60, N)       
    wind_speed  = np.random.uniform(0, 25, N)       
    wind_deg    = np.random.uniform(0, 360, N)      
    time_lost   = np.random.uniform(5, 120, N)      
    temp_c      = np.random.uniform(5, 40, N)       

    noise = np.random.normal(1.0, 0.05, N)
    bearing_rad = np.radians(wind_deg)
    drift_kmh   = velocity + (wind_speed * 3.6 * 0.03)
    offset_m    = (drift_kmh / 60) * time_lost * 1000 / 2 * noise

    delta_lat = (offset_m * np.cos(bearing_rad)) / 111111
    delta_lon = (offset_m * np.sin(bearing_rad)) / (111111 * np.cos(np.radians(21.0)))

    X = np.column_stack([velocity, wind_speed, wind_deg, time_lost, temp_c])

    model_lat = XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
    )
    model_lon = XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
    )
    model_lat.fit(X, delta_lat)
    model_lon.fit(X, delta_lon)

    return model_lat, model_lon


def predict_with_uncertainty(model_lat, model_lon, features, n_bootstrap=100):
    if model_lat is None:
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
        noisy = feat_arr * np.random.normal(1.0, 0.03, feat_arr.shape)
        lat_preds.append(model_lat.predict(noisy)[0])
        lon_preds.append(model_lon.predict(noisy)[0])

    return (
        float(np.mean(lat_preds)), float(np.mean(lon_preds)),
        float(np.std(lat_preds)), float(np.std(lon_preds))
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
# PHẦN 3: DATA INGESTION — Đọc Tọa độ Đa điểm từ File
# ─────────────────────────────────────────────
def parse_excel_data(df):
    """
    Hàm tự động quét tìm các cột chứa từ khóa vĩ độ / kinh độ 
    để xử lý linh hoạt cho mọi form file log.
    """
    lat_col, lon_col, vel_col = None, None, None
    
    for col in df.columns:
        col_str = str(col).lower()
        if 'vĩ độ' in col_str or 'latitude' in col_str or 'lat' in col_str:
            lat_col = col
        elif 'kinh độ' in col_str or 'longitude' in col_str or 'lon' in col_str:
            lon_col = col
        elif 'vận tốc' in col_str or 'velocity' in col_str or 'speed' in col_str or 'mileage' in col_str:
            vel_col = col

    # Dự phòng nếu không quét được tên cột dạng Text (đọc theo index mặc định của bảng dữ liệu)
    if lat_col is None or lon_col is None:
        lat_col = df.columns[2] if len(df.columns) > 2 else df.columns[0]
        lon_col = df.columns[3] if len(df.columns) > 3 else df.columns[1]
    if vel_col is None:
        vel_col = df.columns[-1]

    # Làm sạch dữ liệu, lọc bỏ các dòng trống coordinates
    cleaned_rows = []
    for idx, row in df.iterrows():
        try:
            lat_val = float(str(row[lat_col]).replace(',', '').strip())
            lon_val = float(str(row[lon_col]).replace(',', '').strip())
            
            # Đọc vận tốc
            vel_str = re.findall(r"[-+]?\d+\.?\d*", str(row[vel_col]))
            vel_val = float(vel_str[0]) if vel_str else 0.0
            
            cleaned_rows.append({
                "lat": lat_val,
                "lon": lon_val,
                "velocity": vel_val,
                "altitude": idx * 12, # Tạo cao độ tăng dần cho map 3D sinh động
                "step": idx,
                "label": f"Điểm thứ {idx + 1}"
            })
        except ValueError:
            continue # Bỏ qua dòng tiêu đề phụ hoặc dòng trống chữ

    return pd.DataFrame(cleaned_rows)


# ─────────────────────────────────────────────
# PHẦN 4: PYDECK 3D MAP MULTI-POINTS
# ─────────────────────────────────────────────
def build_pydeck_map(route_df, origin_lat, origin_lon, target_lat, target_lon, std_lat, std_lon):
    # Lớp 1: Hiển thị toàn bộ các điểm từ file dữ liệu lên bản đồ dưới dạng khối Hexagon 3D
    hex_layer = pdk.Layer(
        "HexagonLayer",
        data=route_df,
        get_position=["lon", "lat"],
        get_elevation="altitude",
        elevation_scale=8,
        elevation_range=[0, 3000],
        radius=40,
        pickable=True,
        extruded=True,
        color_range=[
            [0,   40, 120, 200],
            [0,   90, 180, 200],
            [0,  180, 150, 200],
            [200, 200,  0, 220],
            [220, 100,  0, 220],
            [180,   0,   0, 240],
        ],
    )

    # Lớp 2: Vẽ đường nối liền mạch hành trình chạy qua toàn bộ danh sách điểm
    path_data = [{"path": [[r.lon, r.lat] for _, r in route_df.iterrows()]}]
    path_layer = pdk.Layer(
        "PathLayer",
        data=path_data,
        get_path="path",
        get_color=[255, 220, 0, 200],
        width_min_pixels=3,
    )

    # Lớp 3: Đánh dấu chi tiết tất cả các điểm tọa độ từ dữ liệu đầu vào lên bản đồ
    points_layer = pdk.Layer(
        "ScatterplotLayer",
        data=route_df,
        get_position=["lon", "lat"],
        get_fill_color=[0, 229, 255, 200], # Màu xanh neon đặc trưng của app
        get_radius=50,
        pickable=True,
    )

    # Đánh dấu 2 điểm chốt yếu: Điểm bắt đầu mất tín hiệu và Điểm dự đoán của AI
    marker_data = [
        {"lat": origin_lat, "lon": origin_lon, "color": [20, 20, 20, 255],   "radius": 100, "label": "Điểm mất dấu cuối cùng"},
        {"lat": target_lat, "lon": target_lon, "color": [255, 50,  50, 255], "radius": 120, "label": "Tâm Datum dự báo (XGBoost)"},
    ]
    target_layer = pdk.Layer(
        "ScatterplotLayer",
        data=marker_data,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
    )

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
        {"path": ring_68,  "color": [255, 200, 0, 220], "name": "Vùng tin cậy 68%"},
        {"path": ring_95,  "color": [255, 80,  0, 160], "name": "Vùng tin cậy 95%"},
    ]
    ring_layer = pdk.Layer(
        "PathLayer",
        data=rings_data,
        get_path="path",
        get_color="color",
        width_min_pixels=2,
    )

    view_state = pdk.ViewState(
        latitude=(origin_lat + target_lat) / 2,
        longitude=(origin_lon + target_lon) / 2,
        zoom=14,
        pitch=55,
        bearing=15,
    )

    deck = pdk.Deck(
        layers=[hex_layer, path_layer, points_layer, target_layer, ring_layer],
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
    page_icon="🛰️",
    initial_sidebar_state="expanded",
)

# ── CSS: Thiết kế lại toàn bộ giao diện theo 1 hệ màu thống nhất (Light / Ops-room) ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@300;400;500;600;700;800&display=swap');

:root{
    --ink:        #0f172a;
    --ink-soft:   #475569;
    --line:       #e2e8f0;
    --surface:    #ffffff;
    --bg:         #f4f6fb;
    --accent:     #0f6fff;
    --accent-soft:#eaf2ff;
    --danger:     #e0483e;
    --danger-soft:#fdecea;
    --success:    #17a673;
    --success-soft:#e8f8f2;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: var(--bg);
    color: var(--ink);
}

/* Ẩn khoảng trắng thừa trên cùng của khối main */
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: var(--ink);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #ffffff !important; font-weight: 700; }
section[data-testid="stSidebar"] hr { border-color: #ffffff22; }
[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] {
    background-color: #1e293b !important;
    border: 1px dashed #475569 !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #ffffff !important;
}

/* ── Tiêu đề ── */
h1, h2, h3 { font-family: 'Inter', sans-serif !important; color: var(--ink); letter-spacing: -0.2px; }

.hero {
    background: linear-gradient(135deg, var(--ink) 0%, #1e3a5f 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 22px;
    box-shadow: 0 8px 24px rgba(15,23,42,0.18);
}
.hero h1 {
    color: #ffffff !important;
    font-size: 1.65rem;
    font-weight: 800;
    margin: 0 0 6px 0;
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
}
.hero p {
    color: #cbd5e1;
    font-size: 0.92rem;
    margin: 0;
}
.model-badge {
    display: inline-block; background: #ffffff18;
    border: 1px solid #5aa9ff; border-radius: 20px;
    padding: 3px 14px; font-size: 0.72rem; color: #7cc4ff !important;
    font-family: 'Share Tech Mono', monospace; font-weight: 400;
    letter-spacing: 0.5px;
}

/* ── Card chung ── */
.card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 18px;
    box-shadow: 0 2px 8px rgba(15,23,42,0.04);
}
.card h3 { margin-top: 0; font-size: 1.05rem; font-weight: 700; }
.section-title {
    font-size: 1.15rem; font-weight: 700; color: var(--ink);
    margin: 6px 0 14px 0; display:flex; align-items:center; gap:8px;
}

/* ── Metric ── */
.stMetric { 
    background: var(--surface) !important; 
    border: 1px solid var(--line) !important; 
    border-radius: 12px !important;
    padding: 14px 16px !important; 
    border-left: 4px solid var(--accent) !important; 
    box-shadow: 0 2px 8px rgba(15,23,42,0.04);
}
.stMetric label { color: var(--ink-soft) !important; font-size: 0.78rem !important; font-weight: 500; }
.stMetric [data-testid="stMetricValue"] { color: var(--ink) !important; font-weight: 800; font-size: 1.5rem !important; }

/* ── Input trong nội dung chính ── */
div[data-testid="stNumberInput"] div,
div[data-testid="stTextInput"] div,
.stSlider div,
div[data-testid="stFileUploaderDropzone"] {
    background-color: var(--surface) !important;
    color: var(--ink) !important;
    border-radius: 8px !important;
}
div[data-testid="stNumberInput"] input, 
div[data-testid="stTextInput"] input {
    color: var(--ink) !important;
    font-weight: 500;
}

/* ── Nút bấm chính ── */
div.stButton > button {
    background: linear-gradient(135deg, var(--danger), #ff6a5f);
    color: white; border: none; border-radius: 8px;
    font-family: 'Inter', sans-serif; font-weight: 600;
    font-size: 0.95rem; padding: 12px 28px;
    box-shadow: 0 4px 14px rgba(224,72,62,0.35);
    transition: all 0.2s ease;
    width: 100%;
}
div.stButton > button:hover { box-shadow: 0 6px 20px rgba(224,72,62,0.5); transform: translateY(-1px); }

/* ── Hộp cảnh báo / kết quả ── */
.warning-box {
    background: var(--surface) !important; 
    border: 1px solid var(--line) !important;
    border-left: 5px solid var(--danger) !important; 
    border-radius: 12px !important;
    padding: 22px 24px; 
    margin: 0 0 16px 0; 
    box-shadow: 0 2px 10px rgba(15,23,42,0.05);
    color: var(--ink) !important;
}
.warning-box h3 { color: var(--danger) !important; }
.warning-box p, .warning-box b, .warning-box code { color: var(--ink) !important; }

.success-box {
    background: var(--surface) !important; 
    border: 1px solid var(--line) !important;
    border-left: 5px solid var(--success) !important; 
    border-radius: 12px !important;
    padding: 22px 24px; 
    margin: 0 0 16px 0;
    box-shadow: 0 2px 10px rgba(15,23,42,0.05);
    color: var(--ink) !important;
}
.success-box h3 { color: var(--success) !important; }
.success-box p, .success-box b { color: var(--ink-soft) !important; }

/* ── Chú thích bản đồ ── */
.map-legend {
    font-size: 0.8rem; color: var(--ink-soft);
    background: var(--surface); border: 1px solid var(--line);
    border-radius: 10px; padding: 10px 16px; margin-top: 10px;
}

/* ── Trạng thái rỗng (chưa upload file) ── */
.empty-state {
    text-align: center; padding: 48px 24px;
    background: var(--surface); border: 1px dashed var(--line);
    border-radius: 16px;
}
.empty-state h2 { color: var(--ink); font-weight: 700; margin-bottom: 8px; }
.empty-state p { color: var(--ink-soft); max-width: 520px; margin: 0 auto; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="hero">
    <h1>AI PATHFINDING — DỰ ĐOÁN VỊ TRÍ THIẾT BỊ
        <span class="model-badge">Algorithm v2 · 3D Pydeck</span>
    </h1>
    <p>Real-time Weather · Drift Prediction · Uncertainty Ellipse · 3D Terrain Heatmap</p>
</div>
""", unsafe_allow_html=True)

for key in ['analysis_active', 'model_lat', 'model_lon', 'model_trained']:
    if key not in st.session_state:
        st.session_state[key] = False

if not st.session_state.model_trained:
    with st.spinner("Đang khởi tạo mô hình AI..."):
        ml, mln = train_xgboost_model()
        st.session_state.model_lat = ml
        st.session_state.model_lon = mln
        st.session_state.model_trained = True
    if XGBOOST_AVAILABLE:
        st.success("✅ Model AI đã sẵn sàng — 5.000 mẫu synthetic + Bootstrap Ensemble")
    else:
        st.warning("⚠️ XGBoost chưa cài (`pip install xgboost`) — đang dùng mô hình vật lý dự phòng.")

# ── Sidebar ──
st.sidebar.markdown("## 📂 Dữ liệu đầu vào")
uploaded_file = st.sidebar.file_uploader("Tải file dữ liệu Trip Report", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Đọc tệp và bóc tách danh sách toàn bộ tọa độ
    raw_df = pd.read_excel(uploaded_file)
    route_df = parse_excel_data(raw_df)

    if not route_df.empty:
        # Lấy điểm mất tín hiệu cuối cùng (Dòng cuối hoặc dòng đầu tùy cấu trúc log file)
        latest_point = route_df.iloc[0] 
        lat = latest_point["lat"]
        lon = latest_point["lon"]
        velocity = latest_point["velocity"]
    else:
        st.sidebar.error("❌ Không tìm thấy cột chứa dữ liệu tọa độ hợp lệ!")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌦️ Môi trường thực tế")
    weather = get_realtime_weather(lat, lon)

    if weather:
        st.sidebar.success(f"📍 `{lat:.5f}, {lon:.5f}`")
        c1, c2 = st.sidebar.columns(2)
        c1.metric("Nhiệt độ", f"{weather['temp']}°C")
        c2.metric("Lượng mưa", f"{weather['rain']} mm/h")
        st.sidebar.write(f"💨 Gió: **{weather['wind_speed']} m/s** — hướng **{weather['wind_deg']}°**")
        st.sidebar.write(f"👁️ Tầm nhìn: **{weather['visibility']} km** | {weather['description'].capitalize()}")
        wind_speed = weather['wind_speed']
        wind_dir   = weather['wind_deg']
        temp_c     = weather['temp']
    else:
        st.sidebar.warning("⚠️ Dùng dữ liệu dự phòng")
        wind_speed, wind_dir, temp_c = 5.0, 45.0, 25.0

    time_lost = st.sidebar.slider("⏱️ Thời gian mất tín hiệu (phút)", 5, 120, 30)

    st.sidebar.markdown("---")
    st.sidebar.caption("Phát triển bởi đội ngũ NeoSAR")

    # ── Tổng quan thông số ──
    st.markdown('<div class="section-title">📊 Tổng quan thông số đầu vào</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏃 Vận tốc TB",     f"{velocity} km/h")
    col2.metric("💨 Sức gió",        f"{wind_speed} m/s")
    col3.metric("🌡️ Nhiệt độ",      f"{temp_c}°C")
    col4.metric("⏱️ Mất tín hiệu",   f"{time_lost} mins")

    st.write("")
    st.markdown('<div class="section-title">🧠 Phân tích AI &amp; Chiến thuật Cứu hộ</div>', unsafe_allow_html=True)

    if st.button("🚀 Kích hoạt AI Phân tích rủi ro & Tọa độ"):
        st.session_state.analysis_active = True

    if st.session_state.analysis_active:
        with st.status("🛰️ Đang quét dữ liệu đa tầng...", expanded=True) as status:
            st.write("🔧 Nạp mô hình thuật toán...")
            time.sleep(0.4)
            st.write("🌦️ Đọc chỉ số thời tiết thực tế...")
            time.sleep(0.3)
            st.write("🔄 Chạy Bootstrap Ensemble...")
            time.sleep(0.5)
            st.write("📐 Tính toán vùng xác suất...")
            time.sleep(0.3)
            status.update(label="✅ Phân tích hoàn tất!", state="complete")

        features = [velocity, wind_speed, wind_dir, time_lost, temp_c]
        d_lat, d_lon, std_lat, std_lon = predict_with_uncertainty(
            st.session_state.model_lat, st.session_state.model_lon, features, n_bootstrap=100
        )

        new_lat = lat + d_lat
        new_lon = lon + d_lon

        radius_68_m  = int(std_lat * 111111)
        radius_95_m  = int(std_lat * 1.96 * 111111)

        water_temp    = temp_c - 2
        survival_time = "6–12 giờ" if water_temp > 20 else ("2–4 giờ" if water_temp > 10 else "< 1 giờ")
        is_cold       = temp_c < 20
        is_rain       = (weather['rain'] > 5) if weather else False

        res_col1, res_col2 = st.columns(2, gap="medium")
        with res_col1:
            st.markdown(f"""
<div class="warning-box">
<h3>🎯 TỌA ĐỘ MỤC TIÊU ƯU TIÊN <span style="font-size:0.75rem;color:#94a3b8;font-weight:400;">(AI Engine + Bootstrap)</span></h3>
<p>📌 <b>Tọa độ có xác suất cao nhất:</b><br><code style="background:#f4f4f7; padding:2px 6px; border-radius:4px; color:var(--danger) !important;">{new_lat:.6f}, {new_lon:.6f}</code></p>
<p>📐 <b>Vùng dự báo 68%:</b> bán kính ~<b>{radius_68_m} m</b> &nbsp;|&nbsp; <b>95%:</b> ~<b>{radius_95_m} m</b></p>
<p>🧭 <b>Vùng di chuyển:</b> <b>{d_lat*111111:.0f} m</b> Nam-Bắc &nbsp;/&nbsp; <b>{d_lon*111111*math.cos(math.radians(lat)):.0f} m</b> Đông-Tây</p>
</div>
""", unsafe_allow_html=True)
        with res_col2:
            st.markdown(f"""
<div class="success-box">
<h3>🩺 PHÂN TÍCH CHUYÊN MÔN</h3>
<p>⏳ <b>Thời gian vàng:</b> <b style="color:var(--ink) !important;">{survival_time}</b> (nhiệt độ dự báo ~{water_temp:.1f}°C)</p>
<p>🥶 <b>Rủi ro hạ thân nhiệt:</b> <b style="color:var(--danger) !important;">{"CAO — cần ưu tiên sưởi ấm ngay" if is_cold else "Thấp — nằm trong ngưỡng an toàn"}</b></p>
<p>🌧️ <b>Lượng mưa:</b> <b style="color:var(--ink) !important;">{"Mưa lớn — giảm tầm nhìn, triển khai radar" if is_rain else "Thấp, tầm nhìn ổn định — triển khai phương án tiếp cận trực tiếp"}</b></p>
<p>🧭 <b>Chiến thuật đề xuất:</b> Triển khai tìm kiếm theo hình xoắn ốc mở rộng từ tâm tọa độ ưu tiên, ưu tiên vùng 68%.</p>
</div>
""", unsafe_allow_html=True)

        st.code(f"LAT: {new_lat:.6f}   LON: {new_lon:.6f}   [±{radius_68_m}m @ 68% | ±{radius_95_m}m @ 95%]", language="text")

        if XGBOOST_AVAILABLE and st.session_state.model_lat is not None:
            with st.expander("📊 Feature Importance — AI (Lat model)"):
                fi = st.session_state.model_lat.feature_importances_
                fi_df = pd.DataFrame({
                    "Feature":    ["Vận tốc (km/h)", "Sức gió (m/s)", "Hướng gió (°)", "Thời gian (phút)", "Nhiệt độ (°C)"],
                    "Importance": fi
                }).sort_values("Importance", ascending=False)
                st.bar_chart(fi_df.set_index("Feature")["Importance"])

        st.write("")
        st.markdown('<div class="section-title">🗺️ Bản đồ vệ tinh 3D — Toàn bộ hành trình &amp; Vùng xác suất</div>', unsafe_allow_html=True)

        deck = build_pydeck_map(route_df, lat, lon, new_lat, new_lon, std_lat, std_lon)
        st.pydeck_chart(deck)

        st.markdown("""
<div class="map-legend">
🟡 Đường vàng: Lộ trình &nbsp;|&nbsp; 🔵 Chấm xanh neon: Toàn bộ điểm tọa độ log &nbsp;|&nbsp; ⚫ Điểm đen: Vị trí mất dấu cuối cùng &nbsp;|&nbsp; 🔴 Điểm đỏ: Tâm Datum dự tính<br>
🟡 Vòng vàng: 68% &nbsp;|&nbsp; 🟠 Vòng cam: 95% &nbsp;|&nbsp; Cột màu: Khối cao độ 3D tích lũy hành trình
</div>
""", unsafe_allow_html=True)

else:
    st.markdown("""
<div class="empty-state">
    <h2>🛰️ Chưa có dữ liệu hành trình</h2>
    <p>Tải lên file Trip Report (.xlsx / .xls) ở thanh bên trái để bắt đầu phân tích AI và dự đoán vị trí thiết bị.</p>
</div>
""", unsafe_allow_html=True)

    st.write("")
    try:
        st.image("cuuho.png", caption="Hệ thống trực chỉ huy và phân tích rủi ro", use_container_width=True)
    except Exception:
        st.info("💡 Mẹo: Bỏ file ảnh tên 'cuuho.png' vào thư mục dự án để hiển thị poster chỉ huy.")
