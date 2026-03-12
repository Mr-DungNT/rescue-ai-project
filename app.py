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
    """Lấy dữ liệu thời tiết chi tiết từ OpenWeatherMap"""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric&lang=vi"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "wind_speed": data['wind'].get('speed', 0),
                "wind_deg": data['wind'].get('deg', 0),
                "temp": data['main'].get('temp', 0),
                "humidity": data['main'].get('humidity', 0),
                "rain": data.get('rain', {}).get('1h', 0),
                "description": data['weather'][0].get('description', 'N/A'),
                "visibility": data.get('visibility', 10000) / 1000
            }
    except Exception:
        return None
    return None

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="AI Rescue System - Advanced", layout="wide")

# KHỞI TẠO BỘ NHỚ TẠM (SESSION STATE) - Giải quyết lỗi tự động load lại trang
if 'analysis_active' not in st.session_state:
    st.session_state.analysis_active = False

st.title("HỆ THỐNG AI DỰ ĐOÁN VÙNG TÌM KIẾM TÍCH HỢP DỮ LIỆU THỜI TIẾT")
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: TẢI FILE VÀ TÙY CHỈNH ---
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

    # --- LẤY THỜI TIẾT THỰC TẾ ---
    st.sidebar.markdown("---")
    st.sidebar.header("Dữ liệu môi trường thực tế")
    weather = get_realtime_weather(lat, lon)
    
    if weather:
        st.sidebar.success(f"📍 Tọa độ: {lat}, {lon}")
        st.sidebar.metric("🌡️ Nhiệt độ", f"{weather['temp']}°C")
        st.sidebar.metric("🌧️ Lượng mưa", f"{weather['rain']} mm/h")
        st.sidebar.write(f"👁️ Tầm nhìn: {weather['visibility']} km")
        st.sidebar.write(f"☁️ Trạng thái: {weather['description'].capitalize()}")
        
        wind_speed = weather['wind_speed']
        wind_dir = weather['wind_deg']
    else:
        st.sidebar.error("⚠️ Đang dùng dữ liệu dự phòng")
        wind_speed = 5.0
        wind_dir = 45

    time_lost = st.sidebar.slider("Thời gian mất tín hiệu (phút)", 5, 120, 30)

    # --- HIỂN THỊ CHỈ SỐ NHANH ---
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Vận tốc tàu cuối", f"{velocity} km/h")
    with col2: st.metric("Sức gió thực tế", f"{wind_speed} m/s")
    with col3: st.metric("Nhiệt độ môi trường", f"{weather['temp'] if weather else '--'}°C")
    with col4: st.metric("Lượng mưa hiện tại", f"{weather['rain'] if weather else '--'} mm/h")

    # --- LOGIC TÍNH TOÁN DRIFT ---
    drift_speed_kmh = velocity + (wind_speed * 3.6 * 0.03)
    total_drift_meters = (drift_speed_kmh / 60) * time_lost * 1000
    bearing = math.radians(wind_dir)
    offset_dist = total_drift_meters / 2 
    new_lat = lat + (offset_dist * math.cos(bearing) / 111111)
    new_lon = lon + (offset_dist * math.sin(bearing) / (111111 * math.cos(math.radians(lat))))

    # --- PHẦN PHÂN TÍCH AI CHUYÊN SÂU ---
    st.divider()
    st.subheader("🤖 Phân tích AI & Phương án cứu hộ")
    
    # Nút bấm kích hoạt trạng thái
    if st.button("Kích hoạt AI phân tích rủi ro & tọa độ"):
        st.session_state.analysis_active = True
        with st.status("Đang quét dữ liệu đa tầng...", expanded=True) as status:
            time.sleep(0.5)
            st.write("🛰️ Đang trích xuất tọa độ mục tiêu ưu tiên...")
            time.sleep(0.5)
            st.write("🌡️ Đang tính toán rủi ro hạ thân nhiệt...")
            status.update(label="Hoàn tất phân tích!", state="complete")

    # Hiển thị kết quả nếu trạng thái Active là True
    if st.session_state.analysis_active:
        is_heavy_rain = weather['rain'] > 5 if weather else False
        is_cold = weather['temp'] < 20 if weather else False
        water_temp = (weather['temp'] - 2) if weather else 20
        survival_time = "6-12 giờ" if water_temp > 20 else "2-4 giờ"
        
        danger_msg = "🚨 CẢNH BÁO NGUY HIỂM:" if is_heavy_rain else "✅ ĐIỀU KIỆN ỔN ĐỊNH:"
        
        st.warning(f"""
        ### TỌA ĐỘ MỤC TIÊU ƯU TIÊN:
        * **Tâm vùng tìm kiếm (Datum):** `{new_lat:.6f}, {new_lon:.6f}`
        * **Độ lệch dự kiến:** Dịch chuyển {offset_dist:.0f}m về hướng {wind_dir}°.

        ### PHÂN TÍCH CHUYÊN MÔN:
        1. **Thời gian vàng cứu hộ:** Ước tính **{survival_time}** (Dựa trên nhiệt độ {weather['temp']}°C).
        2. **Tình trạng thiên tai:** {weather['description'].capitalize() if weather else 'N/A'}.
        3. **Cảnh báo:** { "Nguy cơ hạ thân nhiệt nhanh." if is_cold else "Nhiệt độ nước ổn định." }
        
        ### PHƯƠNG PHÁP TÌM KIẾM ĐỀ XUẤT:
        * **Phương pháp:** Tìm kiếm theo hình xoắn ốc mở rộng (Expanding Square Search).
        * **Hành động:** Triển khai thiết bị tầm nhiệt tại tọa độ mục tiêu.
        """)
        st.code(f"{new_lat:.6f}, {new_lon:.6f}", language="text")

    # --- BẢN ĐỒ ---
    st.divider()
    st.subheader("📍 Bản đồ vệ tinh & Dự đoán vùng trôi dạt")
    m = folium.Map(location=[new_lat, new_lon], zoom_start=15)
    folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', 
                     attr='Esri', name='Vệ tinh').add_to(m)

    folium.Marker([lat, lon], popup="Điểm mất dấu", icon=folium.Icon(color='black', icon='off')).add_to(m)
    folium.Circle(location=[new_lat, new_lon], radius=total_drift_meters/2 + 200, 
                  color="red", fill=True, fill_opacity=0.3, popup="Vùng mục tiêu ưu tiên").add_to(m)
    
    # Mũi tên hướng gió
    folium.PolyLine([[lat, lon], [lat + 0.005 * math.cos(bearing), lon + 0.005 * math.sin(bearing)]], 
                    color="yellow", weight=4).add_to(m)

    st_folium(m, width="100%", height=600)

else:
    st.info("👋 Chào mừng! Hãy tải file Excel để bắt đầu cập nhật dữ liệu thiên tai thời gian thực.")
    st.image("https://media.vietnamplus.vn/images/db3eecc2e589c60996480488f99e20f49ca9bb5a263a4de8d02595b616691c38aa3bf5d5b92561c2a6a2ce192fbe6b5e74e94f2aa426d84316be5dd1ba1bf47f/mua_ngap_han_quoc.jpg", caption="Hệ thống trực chiến 24/7", use_container_width=True)
