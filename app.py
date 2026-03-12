import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import re
import time
import math
import requests

# --- CẤU HÌNH API THỜI TIẾT ---
API_KEY = "23913db94b60da48fe4dd64dbab2344f"

def get_realtime_weather(lat, lon):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric&lang=vi"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "wind_speed": data['wind'].get('speed', 0),
                "wind_deg": data['wind'].get('deg', 0),
                "temp": data['main'].get('temp', 0),
                "rain": data.get('rain', {}).get('1h', 0),
                "description": data['weather'][0].get('description', 'N/A'),
                "visibility": data.get('visibility', 10000) / 1000
            }
    except Exception:
        return None
    return None

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="AI Rescue System - Fix", layout="wide")

# Khởi tạo trạng thái lưu trữ (Session State) để không bị mất phần phân tích khi load lại
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

st.title("🚢 HỆ THỐNG AI CỨU HỘ TÍCH HỢP DỮ LIỆU THIÊN TAI")

# --- SIDEBAR ---
st.sidebar.header("📂 Dữ liệu đầu vào")
uploaded_file = st.sidebar.file_uploader("Tải lên file Trip Report (Excel)", type=["xlsx", "xls"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    
    def extract_coords(text):
        try:
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", str(text))
            return float(numbers[0]), float(numbers[1])
        except: return None, None

    latest = df.iloc[0]
    lat, lon = extract_coords(latest.iloc[5]) 
    velocity_str = str(latest.iloc[9])
    velocity_match = re.findall(r"[-+]?\d*\.\d+|\d+", velocity_str)
    velocity = float(velocity_match[0]) if velocity_match else 0.0

    # Lấy thời tiết
    weather = get_realtime_weather(lat, lon)
    if weather:
        st.sidebar.success(f"📍 Tọa độ hiện tại: {lat}, {lon}")
        wind_speed = weather['wind_speed']
        wind_dir = weather['wind_deg']
    else:
        wind_speed, wind_dir = 5.0, 45

    time_lost = st.sidebar.slider("Thời gian mất tín hiệu (phút)", 5, 120, 30)

    # Hiển thị Metric
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Vận tốc tàu cuối", f"{velocity} km/h")
    with col2: st.metric("Sức gió thực tế", f"{wind_speed} m/s")
    with col3: st.metric("Nhiệt độ môi trường", f"{weather['temp'] if weather else '--'}°C")
    with col4: st.metric("Lượng mưa hiện tại", f"{weather['rain'] if weather else '--'} mm/h")

    # Tính toán tọa độ mới
    drift_speed_kmh = velocity + (wind_speed * 3.6 * 0.03)
    total_drift_meters = (drift_speed_kmh / 60) * time_lost * 1000
    bearing = math.radians(wind_dir)
    offset_dist = total_drift_meters / 2 
    new_lat = lat + (offset_dist * math.cos(bearing) / 111111)
    new_lon = lon + (offset_dist * math.sin(bearing) / (111111 * math.cos(math.radians(lat))))

    # --- XỬ LÝ NÚT BẤM VÀ HIỂN THỊ PHÂN TÍCH ---
    st.divider()
    st.subheader("🤖 Phân tích AI & Chiến thuật Cứu hộ")

    # Nếu nhấn nút, đổi trạng thái thành True
    if st.button("Kích hoạt AI Phân tích rủi ro & Tọa độ"):
        st.session_state.analysis_done = True
        with st.status("Đang quét dữ liệu đa tầng...", expanded=True):
            time.sleep(1)
            st.write("🛰️ Đang xác định tọa độ mục tiêu...")
            st.write("📊 Đang phân tích rủi ro hạ thân nhiệt...")

    # Hiển thị nội dung phân tích nếu trạng thái là True
    if st.session_state.analysis_done:
        water_temp = (weather['temp'] - 2) if weather else 20
        survival_time = "6-12 giờ" if water_temp > 20 else "2-4 giờ"
        
        st.warning(f"""
        ### 🎯 TỌA ĐỘ MỤC TIÊU ƯU TIÊN:
        * **Tâm vùng tìm kiếm:** `{new_lat:.6f}, {new_lon:.6f}`
        * **Độ lệch dự kiến:** {offset_dist:.0f}m về hướng {wind_dir}°.

        ### 📋 PHÂN TÍCH CHUYÊN MÔN:
        1. **Thời gian vàng cứu hộ:** **{survival_time}** dựa trên nhiệt độ {weather['temp']}°C.
        2. **Cảnh báo:** Nguy cơ nạn nhân bị hạ thân nhiệt nhanh.
        3. **Chiến thuật:** Tìm kiếm theo hình xoắn ốc mở rộng (Expanding Square Search).
        """)
        st.code(f"{new_lat:.6f}, {new_lon:.6f}", language="text")

    # --- BẢN ĐỒ ---
    st.divider()
    st.subheader("📍 Bản đồ vệ tinh & Dự đoán vùng trôi dạt")
    m = folium.Map(location=[new_lat, new_lon], zoom_start=15)
    folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', 
                     attr='Esri', name='Vệ tinh').add_to(m)
    folium.Marker([lat, lon], popup="Vị trí cuối", icon=folium.Icon(color='black')).add_to(m)
    folium.Circle(location=[new_lat, new_lon], radius=total_drift_meters/2 + 200, 
                  color="red", fill=True, fill_opacity=0.3).add_to(m)
    st_folium(m, width="100%", height=500)

else:
    st.info("👋 Chào mừng! Hãy tải file Excel để bắt đầu.")
    st.image("https://media.vietnamplus.vn/images/db3eecc2e589c60996480488f99e20f49ca9bb5a263a4de8d02595b616691c38aa3bf5d5b92561c2a6a2ce192fbe6b5e74e94f2aa426d84316be5dd1ba1bf47f/mua_ngap_han_quoc.jpg", use_container_width=True)
