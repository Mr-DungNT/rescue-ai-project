import math
import time

import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

# ============================================================
# CẤU HÌNH CHUNG
# ============================================================
st.set_page_config(
    page_title="AI Pathfinding — Dự đoán vị trí thiết bị",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_KEY = "23913db94b60da48fe4dd64dbab2344f"

ASSET_PROFILES = {
    "Người đi bộ, mất tích trên cạn": {"speed_range": (2.0, 6.0), "leeway": 0.020},
    "Người rơi xuống nước, không có phao": {"speed_range": (0.0, 1.0), "leeway": 0.030},
    "Người mặc áo phao hoặc bám vật nổi": {"speed_range": (0.0, 1.5), "leeway": 0.045},
    "Xuồng nhỏ, không động cơ": {"speed_range": (1.0, 4.0), "leeway": 0.060},
    "Phương tiện có động cơ, mất liên lạc": {"speed_range": (5.0, 40.0), "leeway": 0.015},
}

# ============================================================
# GIAO DIỆN — THEME SÁNG, TƯƠNG PHẢN RÕ
# ============================================================
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
    --ink:        #111827;
    --ink-soft:   #4b5563;
    --line:       #e2e5eb;
    --surface:    #ffffff;
    --bg:         #f6f7fb;
    --accent:     #1d4ed8;
    --accent-soft:#eef2ff;
    --danger:     #b91c1c;
    --danger-soft:#fdecea;
    --success:    #166534;
    --success-soft:#eaf6ee;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: var(--bg);
    color: var(--ink);
}

.block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1180px; }

section[data-testid="stSidebar"] {
    background-color: var(--surface);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] * { color: var(--ink) !important; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { font-weight: 700; }
section[data-testid="stSidebar"] hr { border-color: var(--line); }

[data-testid="stFileUploaderDropzone"] {
    background-color: var(--bg) !important;
    border: 1px dashed #b7bec9 !important;
    border-radius: 10px !important;
}
div[data-baseweb="select"] > div {
    background-color: var(--surface) !important;
    border: 1px solid var(--line) !important;
    color: var(--ink) !important;
}

h1, h2, h3 { font-family: 'Inter', sans-serif !important; color: var(--ink); letter-spacing: -0.2px; }

.hero {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 34px 36px;
    margin-bottom: 24px;
}
.hero .eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--ink-soft);
    margin-bottom: 10px;
}
.hero h1 {
    font-size: 1.9rem;
    font-weight: 800;
    margin: 0 0 10px 0;
}
.hero p {
    color: var(--ink-soft);
    font-size: 0.98rem;
    margin: 0;
    max-width: 640px;
    line-height: 1.6;
}

.card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 22px 24px;
    margin-bottom: 18px;
}
.card h3 { margin-top: 0; font-size: 1.02rem; font-weight: 700; }

.section-title {
    font-size: 1.05rem; font-weight: 700; color: var(--ink);
    margin: 30px 0 14px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--line);
}

.stMetric {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
    border-left: 3px solid var(--accent) !important;
}
.stMetric label { color: var(--ink-soft) !important; font-size: 0.76rem !important; font-weight: 500; }
.stMetric [data-testid="stMetricValue"] { color: var(--ink) !important; font-weight: 800; font-size: 1.4rem !important; }

div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    color: var(--ink) !important;
    background-color: var(--surface) !important;
}

div.stButton > button {
    background: var(--ink);
    color: #ffffff; border: none; border-radius: 8px;
    font-weight: 600; font-size: 0.92rem; padding: 11px 26px;
    width: 100%;
    transition: background 0.15s ease;
}
div.stButton > button:hover { background: #000000; }

div[data-testid="stDownloadButton"] > button {
    background: var(--surface);
    color: var(--ink);
    border: 1px solid var(--line);
    border-radius: 8px;
    font-weight: 600;
    width: 100%;
}

.result-box {
    background: var(--surface);
    border: 1px solid var(--line);
    border-left: 4px solid var(--accent);
    border-radius: 10px;
    padding: 22px 24px;
    margin-bottom: 16px;
}
.result-box.danger { border-left-color: var(--danger); }
.result-box.success { border-left-color: var(--success); }
.result-box h3 { font-size: 0.98rem; font-weight: 700; margin-top: 0; }
.result-box p { color: var(--ink-soft); margin: 6px 0; line-height: 1.6; font-size: 0.92rem; }
.result-box code {
    background: var(--bg); padding: 3px 8px; border-radius: 6px;
    color: var(--ink); font-family: 'JetBrains Mono', monospace; font-size: 0.95rem;
}

.legend {
    font-size: 0.8rem; color: var(--ink-soft);
    background: var(--surface); border: 1px solid var(--line);
    border-radius: 8px; padding: 12px 16px; margin-top: 10px; line-height: 1.7;
}

.empty-state {
    text-align: center; padding: 56px 24px;
    background: var(--surface); border: 1px dashed var(--line);
    border-radius: 12px;
}
.empty-state h2 { color: var(--ink); font-weight: 700; margin-bottom: 8px; font-size: 1.3rem; }
.empty-state p { color: var(--ink-soft); max-width: 480px; margin: 0 auto; line-height: 1.6; font-size: 0.92rem; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


# ============================================================
# DỮ LIỆU THỜI TIẾT
# ============================================================
def get_current_weather(lat, lon):
    try:
        url = (f"https://api.openweathermap.org/data/2.5/weather"
               f"?lat={lat}&lon={lon}&appid={API_KEY}&units=metric&lang=vi")
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            d = resp.json()
            return {
                "wind_speed":  d["wind"].get("speed", 0),
                "wind_deg":    d["wind"].get("deg", 0),
                "temp":        d["main"].get("temp", 25),
                "humidity":    d["main"].get("humidity", 70),
                "rain":        d.get("rain", {}).get("1h", 0),
                "description": d["weather"][0].get("description", "Không rõ"),
                "visibility":  d.get("visibility", 10000) / 1000,
            }
    except Exception:
        pass
    return None


# ============================================================
# ĐỌC FILE TRIP REPORT
# ============================================================
def parse_excel_data(df):
    lat_col, lon_col, vel_col = None, None, None
    for col in df.columns:
        c = str(col).lower()
        if any(k in c for k in ["vĩ độ", "latitude", "lat"]):
            lat_col = col
        elif any(k in c for k in ["kinh độ", "longitude", "lon"]):
            lon_col = col
        elif any(k in c for k in ["vận tốc", "velocity", "speed", "mileage"]):
            vel_col = col

    if lat_col is None or lon_col is None:
        lat_col = df.columns[2] if len(df.columns) > 2 else df.columns[0]
        lon_col = df.columns[3] if len(df.columns) > 3 else df.columns[1]
    if vel_col is None:
        vel_col = df.columns[-1]

    rows = []
    for idx, row in df.iterrows():
        try:
            lat_val = float(str(row[lat_col]).replace(",", "").strip())
            lon_val = float(str(row[lon_col]).replace(",", "").strip())
            vel_match = pd.Series(str(row[vel_col])).str.extract(r"([-+]?\d+\.?\d*)")[0][0]
            vel_val = float(vel_match) if vel_match is not None else 0.0
            rows.append({"lat": lat_val, "lon": lon_val, "velocity": vel_val, "step": idx})
        except (ValueError, TypeError):
            continue
    return pd.DataFrame(rows)


def initial_bearing(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return math.degrees(math.atan2(x, y)) % 360


# ============================================================
# MÔ PHỎNG MONTE CARLO — HÀNH VI TRÔI DẠT
# ============================================================
def run_monte_carlo(origin_lat, origin_lon, last_heading_deg, wind_speed, wind_deg,
                     time_lost_min, asset_profile, n_particles=4000, seed=42):
    rng = np.random.default_rng(seed)
    speed_lo, speed_hi = asset_profile["speed_range"]
    leeway = asset_profile["leeway"]

    if last_heading_deg is None:
        headings = rng.uniform(0, 360, n_particles)
    else:
        kappa = max(0.4, 5.0 - time_lost_min / 25)
        headings = np.degrees(rng.vonmises(math.radians(last_heading_deg), kappa, n_particles)) % 360

    own_speed = rng.uniform(speed_lo, speed_hi, n_particles)
    wind_drift_kmh = wind_speed * 3.6 * leeway
    t_hours = (time_lost_min / 60) * rng.uniform(0.85, 1.15, n_particles)

    own_dx = own_speed * np.sin(np.radians(headings)) * t_hours
    own_dy = own_speed * np.cos(np.radians(headings)) * t_hours
    wind_dx = wind_drift_kmh * np.sin(np.radians(wind_deg)) * t_hours
    wind_dy = wind_drift_kmh * np.cos(np.radians(wind_deg)) * t_hours

    dx_km = own_dx + wind_dx
    dy_km = own_dy + wind_dy

    d_lat = dy_km * 1000 / 111111
    d_lon = dx_km * 1000 / (111111 * math.cos(math.radians(origin_lat)))

    lats = origin_lat + d_lat
    lons = origin_lon + d_lon
    return lats, lons


def compute_datum_and_rings(lats, lons, origin_lat):
    datum_lat = float(np.mean(lats))
    datum_lon = float(np.mean(lons))
    dy = (lats - datum_lat) * 111111
    dx = (lons - datum_lon) * 111111 * math.cos(math.radians(origin_lat))
    dist = np.sqrt(dx**2 + dy**2)
    r50 = float(np.percentile(dist, 50))
    r90 = float(np.percentile(dist, 90))
    r95 = float(np.percentile(dist, 95))
    return datum_lat, datum_lon, r50, r90, r95


def recommend_search_effort(radius_m, visibility_km):
    area_km2 = math.pi * (radius_m / 1000) ** 2
    sweep_width_km = max(0.05, min(visibility_km * 0.3, 1.5))
    track_length_km = area_km2 / sweep_width_km
    search_hours = track_length_km / 20  # tốc độ quét hiệu dụng giả định 20 km/h

    if area_km2 < 0.5:
        pattern = "Tìm kiếm hình vuông mở rộng (Expanding Square)"
    elif area_km2 < 5:
        pattern = "Tìm kiếm hình quạt (Sector Search)"
    else:
        pattern = "Tìm kiếm đường song song (Parallel Track)"

    return area_km2, sweep_width_km, track_length_km, search_hours, pattern


def make_ring(center_lat, center_lon, radius_m, n=72):
    pts = []
    r_lat = radius_m / 111111
    r_lon = radius_m / (111111 * math.cos(math.radians(center_lat)))
    for k in range(n + 1):
        a = 2 * math.pi * k / n
        pts.append([center_lon + r_lon * math.sin(a), center_lat + r_lat * math.cos(a)])
    return pts


# ============================================================
# BẢN ĐỒ
# ============================================================
def build_map(route_df, particle_df, origin_lat, origin_lon, datum_lat, datum_lon, r50, r90, r95):
    layers = []

    if len(route_df) > 1:
        path_data = [{"path": [[r.lon, r.lat] for _, r in route_df.iterrows()]}]
        layers.append(pdk.Layer(
            "PathLayer", data=path_data, get_path="path",
            get_color=[29, 78, 216, 200], width_min_pixels=2,
        ))

    layers.append(pdk.Layer(
        "ScatterplotLayer", data=route_df, get_position=["lon", "lat"],
        get_fill_color=[71, 85, 105, 160], get_radius=25, pickable=True,
    ))

    layers.append(pdk.Layer(
        "HeatmapLayer", data=particle_df, get_position=["lon", "lat"],
        opacity=0.45, radius_pixels=40,
    ))

    rings_data = [
        {"path": make_ring(datum_lat, datum_lon, r50), "color": [29, 78, 216, 200]},
        {"path": make_ring(datum_lat, datum_lon, r90), "color": [180, 130, 20, 190]},
        {"path": make_ring(datum_lat, datum_lon, r95), "color": [185, 28, 28, 160]},
    ]
    layers.append(pdk.Layer(
        "PathLayer", data=rings_data, get_path="path", get_color="color", width_min_pixels=2,
    ))

    marker_data = [
        {"lat": origin_lat, "lon": origin_lon, "color": [17, 24, 39, 255], "radius": 90, "label": "Điểm mất tín hiệu"},
        {"lat": datum_lat, "lon": datum_lon, "color": [185, 28, 28, 255], "radius": 110, "label": "Tâm datum dự báo"},
    ]
    layers.append(pdk.Layer(
        "ScatterplotLayer", data=marker_data, get_position=["lon", "lat"],
        get_fill_color="color", get_radius="radius", pickable=True,
    ))

    view_state = pdk.ViewState(
        latitude=(origin_lat + datum_lat) / 2,
        longitude=(origin_lon + datum_lon) / 2,
        zoom=13, pitch=0, bearing=0,
    )

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"text": "{label}"},
    )


# ============================================================
# TRẠNG THÁI PHIÊN
# ============================================================
for key, default in [("page", "landing"), ("analysis_active", False)]:
    if key not in st.session_state:
        st.session_state[key] = default


def go_to_dashboard():
    st.session_state.page = "dashboard"


def go_to_landing():
    st.session_state.page = "landing"
    st.session_state.analysis_active = False


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("### Dữ liệu đầu vào")
uploaded_file = st.sidebar.file_uploader("Tải file Trip Report (.xlsx / .xls)", type=["xlsx", "xls"])

route_df = pd.DataFrame()
if uploaded_file is not None:
    raw_df = pd.read_excel(uploaded_file)
    route_df = parse_excel_data(raw_df)
    if route_df.empty:
        st.sidebar.error("Không tìm thấy cột tọa độ hợp lệ trong file.")
    else:
        order = st.sidebar.radio(
            "Điểm mất tín hiệu cuối cùng nằm ở",
            ["Dòng cuối file (thứ tự thời gian tăng dần)", "Dòng đầu file"],
            index=0,
        )
        last_point = route_df.iloc[-1] if "cuối" in order else route_df.iloc[0]

        st.sidebar.markdown("---")
        asset_label = st.sidebar.selectbox("Đối tượng cần tìm kiếm", list(ASSET_PROFILES.keys()))

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Điều kiện thực tế")
        weather = get_current_weather(last_point["lat"], last_point["lon"])
        if weather:
            st.sidebar.write(f"Toạ độ: {last_point['lat']:.5f}, {last_point['lon']:.5f}")
            c1, c2 = st.sidebar.columns(2)
            c1.metric("Nhiệt độ", f"{weather['temp']}°C")
            c2.metric("Mưa", f"{weather['rain']} mm/h")
            st.sidebar.write(f"Gió: {weather['wind_speed']} m/s, hướng {weather['wind_deg']}°")
            st.sidebar.write(f"Tầm nhìn: {weather['visibility']:.1f} km — {weather['description'].capitalize()}")
        else:
            st.sidebar.warning("Không lấy được dữ liệu thời tiết thực — dùng giá trị dự phòng.")
            weather = {"wind_speed": 5.0, "wind_deg": 45.0, "temp": 25.0, "rain": 0, "visibility": 8.0, "description": "không rõ"}

        time_lost = st.sidebar.slider("Thời gian mất tín hiệu (phút)", 5, 240, 30)

        st.sidebar.markdown("---")
        if st.session_state.page == "landing":
            st.sidebar.button("Bắt đầu phân tích", on_click=go_to_dashboard)
        else:
            st.sidebar.button("Quay lại trang đầu", on_click=go_to_landing)


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <div class="eyebrow">Hệ thống hỗ trợ cứu hộ</div>
    <h1>AI Pathfinding — Dự đoán vị trí thiết bị mất tín hiệu</h1>
    <p>Mô phỏng Monte Carlo dựa trên gió, dòng trôi và hành vi di chuyển của đối tượng, kết hợp dữ liệu thời tiết thời gian thực để xác định vùng tìm kiếm ưu tiên và đề xuất phương án triển khai.</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# TRANG LANDING
# ============================================================
if st.session_state.page == "landing" or uploaded_file is None:
    st.markdown("""
    <div class="empty-state">
        <h2>Chưa có dữ liệu hành trình</h2>
        <p>Tải lên file Trip Report (.xlsx hoặc .xls) ở thanh bên trái. Hệ thống sẽ đọc toàn bộ điểm tọa độ, lấy điều kiện thời tiết thực tế tại vị trí mất tín hiệu, và mô phỏng vùng trôi dạt có xác suất cao nhất.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="card"><h3>Dữ liệu hành trình</h3><p style="color:var(--ink-soft);font-size:0.88rem;">
        Đọc toàn bộ log tọa độ, không chỉ điểm cuối, để suy ra hướng di chuyển gần nhất của đối tượng.</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card"><h3>Mô phỏng trôi dạt</h3><p style="color:var(--ink-soft);font-size:0.88rem;">
        4.000 kịch bản Monte Carlo kết hợp gió thực tế, hệ số leeway theo loại đối tượng và độ bất định thời gian.</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card"><h3>Đề xuất chiến thuật</h3><p style="color:var(--ink-soft);font-size:0.88rem;">
        Vùng tin cậy 50/90/95%, kiểu hình tìm kiếm phù hợp và ước tính số giờ quét cần thiết.</p></div>""", unsafe_allow_html=True)

# ============================================================
# TRANG DASHBOARD
# ============================================================
else:
    lat, lon, velocity = last_point["lat"], last_point["lon"], last_point["velocity"]
    wind_speed, wind_dir, temp_c = weather["wind_speed"], weather["wind_deg"], weather["temp"]

    heading = None
    if len(route_df) >= 2:
        p1, p2 = route_df.iloc[-2], route_df.iloc[-1]
        heading = initial_bearing(p1["lat"], p1["lon"], p2["lat"], p2["lon"])

    st.markdown('<div class="section-title">Tổng quan thông số đầu vào</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Vận tốc gần nhất", f"{velocity:.1f} km/h")
    col2.metric("Sức gió", f"{wind_speed} m/s")
    col3.metric("Nhiệt độ", f"{temp_c}°C")
    col4.metric("Thời gian mất tín hiệu", f"{time_lost} phút")

    st.markdown('<div class="section-title">Phân tích và chiến thuật cứu hộ</div>', unsafe_allow_html=True)

    if st.button("Kích hoạt phân tích"):
        st.session_state.analysis_active = True

    if st.session_state.analysis_active:
        with st.status("Đang chạy mô phỏng...", expanded=False) as status:
            st.write("Nạp thông số đầu vào và dữ liệu thời tiết.")
            time.sleep(0.3)
            st.write("Chạy 4.000 kịch bản Monte Carlo.")
            time.sleep(0.4)
            st.write("Tính vùng tin cậy và đề xuất chiến thuật.")
            time.sleep(0.2)
            status.update(label="Hoàn tất", state="complete")

        asset_profile = ASSET_PROFILES[asset_label]
        lats, lons = run_monte_carlo(lat, lon, heading, wind_speed, wind_dir, time_lost, asset_profile)
        datum_lat, datum_lon, r50, r90, r95 = compute_datum_and_rings(lats, lons, lat)
        area_km2, sweep_km, track_km, search_hours, pattern = recommend_search_effort(r90, weather["visibility"])

        water_temp = temp_c - 2
        if water_temp > 20:
            survival_time = "6 đến 12 giờ"
        elif water_temp > 10:
            survival_time = "2 đến 4 giờ"
        else:
            survival_time = "dưới 1 giờ"
        is_cold = temp_c < 20
        is_rain = weather["rain"] > 5

        rc1, rc2 = st.columns(2, gap="medium")
        with rc1:
            st.markdown(f"""
            <div class="result-box danger">
            <h3>Tọa độ trung tâm vùng tìm kiếm</h3>
            <p>Datum dự báo: <code>{datum_lat:.6f}, {datum_lon:.6f}</code></p>
            <p>Vùng tin cậy: 50% trong bán kính {r50:.0f} m — 90% trong {r90:.0f} m — 95% trong {r95:.0f} m</p>
            <p>Hướng di chuyển ưu tiên: {"chưa xác định, dùng phân bố đều" if heading is None else f"khoảng {heading:.0f}° so với hướng Bắc, dựa trên hành trình đã ghi nhận"}</p>
            </div>
            """, unsafe_allow_html=True)
        with rc2:
            st.markdown(f"""
            <div class="result-box success">
            <h3>Đánh giá tình huống</h3>
            <p>Thời gian vàng ước tính: {survival_time} (nhiệt độ môi trường ~{water_temp:.1f}°C)</p>
            <p>Rủi ro hạ thân nhiệt: {"Cao, cần ưu tiên sưởi ấm ngay khi tiếp cận" if is_cold else "Thấp, trong ngưỡng an toàn"}</p>
            <p>Điều kiện quan sát: {"Mưa lớn, tầm nhìn giảm, cân nhắc bổ sung thiết bị hỗ trợ" if is_rain else "Ổn định, phù hợp tiếp cận trực tiếp"}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-box">
        <h3>Đề xuất triển khai tìm kiếm</h3>
        <p>Diện tích vùng tìm kiếm (bán kính 90%): {area_km2:.2f} km²</p>
        <p>Kiểu hình tìm kiếm phù hợp: {pattern}</p>
        <p>Khoảng cách quét đề xuất giữa các tuyến: {sweep_km:.2f} km, tổng chiều dài tuyến quét: {track_km:.1f} km</p>
        <p>Thời gian quét ước tính (một tổ tìm kiếm, tốc độ hiệu dụng 20 km/h): {search_hours:.1f} giờ</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Độ nhạy: bán kính vùng tìm kiếm theo thời gian mất tín hiệu</div>', unsafe_allow_html=True)
        sens_rows = []
        for factor in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            t = max(5, time_lost * factor)
            l2, o2 = run_monte_carlo(lat, lon, heading, wind_speed, wind_dir, t, asset_profile, n_particles=800, seed=7)
            _, _, _, r90_t, _ = compute_datum_and_rings(l2, o2, lat)
            sens_rows.append({"Thời gian (phút)": round(t), "Bán kính 90% (m)": round(r90_t)})
        sens_df = pd.DataFrame(sens_rows).drop_duplicates(subset="Thời gian (phút)").set_index("Thời gian (phút)")
        st.line_chart(sens_df)

        st.markdown('<div class="section-title">Bản đồ vùng tìm kiếm</div>', unsafe_allow_html=True)
        n_show = min(2000, len(lats))
        particle_df = pd.DataFrame({"lat": lats[:n_show], "lon": lons[:n_show]})
        deck = build_map(route_df, particle_df, lat, lon, datum_lat, datum_lon, r50, r90, r95)
        st.pydeck_chart(deck)

        st.markdown("""
        <div class="legend">
        Đường xanh: hành trình đã ghi nhận — Vùng nhiệt: mật độ khả năng vị trí theo mô phỏng —
        Điểm đen: vị trí mất tín hiệu cuối cùng — Điểm đỏ: tâm datum dự báo —
        Vòng xanh 50% — Vòng vàng 90% — Vòng đỏ 95%
        </div>
        """, unsafe_allow_html=True)

        report = f"""BÁO CÁO PHÂN TÍCH VỊ TRÍ TÌM KIẾM — AI PATHFINDING

Vị trí mất tín hiệu cuối cùng: {lat:.6f}, {lon:.6f}
Đối tượng tìm kiếm: {asset_label}
Thời gian mất tín hiệu: {time_lost} phút
Điều kiện thời tiết: gió {wind_speed} m/s hướng {wind_dir}°, nhiệt độ {temp_c}°C, {weather['description']}

KẾT QUẢ MÔ PHỎNG (Monte Carlo, 4000 kịch bản)
Tâm datum dự báo: {datum_lat:.6f}, {datum_lon:.6f}
Bán kính vùng tin cậy 50%: {r50:.0f} m
Bán kính vùng tin cậy 90%: {r90:.0f} m
Bán kính vùng tin cậy 95%: {r95:.0f} m

ĐỀ XUẤT TRIỂN KHAI
Diện tích vùng tìm kiếm (90%): {area_km2:.2f} km2
Kiểu hình tìm kiếm: {pattern}
Khoảng cách quét đề xuất: {sweep_km:.2f} km
Thời gian quét ước tính: {search_hours:.1f} giờ (1 tổ, tốc độ hiệu dụng 20 km/h)

ĐÁNH GIÁ TÌNH HUỐNG
Thời gian vàng ước tính: {survival_time}
Rủi ro hạ thân nhiệt: {"Cao" if is_cold else "Thấp"}
Điều kiện quan sát: {"Mưa lớn, tầm nhìn giảm" if is_rain else "Ổn định"}
"""
        st.download_button("Tải báo cáo tóm tắt (.txt)", report, file_name="bao_cao_tim_kiem.txt")
