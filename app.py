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
                "rain": data.get('rain', {}).get('1h', 0), # Lượng mưa 1h qua
                "description": data['weather'][0].get('description', 'N/A'),
                "visibility": data.get('visibility', 10000) / 1000 # Tầm nhìn (km)
            }
    except Exception:
        return None
    return None

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="AI Rescue System - Advanced", layout="wide")

st.title("🚢 HỆ THỐNG AI CỨU HỘ TÍCH HỢP DỮ LIỆU THIÊN TAI")
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b; }
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
    st.sidebar.header("🌍 Chỉ số môi trường thực tế")
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
        st.sidebar.error("⚠️ Đang dùng dữ liệu dự phòng (Kết nối API gián đoạn)")
        wind_speed = 5.0
        wind_dir = 45

    time_lost = st.sidebar.slider("Thời gian mất tín hiệu (phút)", 5, 120, 30)

    # --- HIỂN THỊ CHỈ SỐ NHANH ---
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Vận tốc tàu cuối", f"{velocity} km/h")
    with col2: st.metric("Sức gió", f"{wind_speed} m/s")
    with col3: st.metric("Nhiệt độ môi trường", f"{weather['temp'] if weather else '--'}°C")
    with col4: st.metric("Lượng mưa", f"{weather['rain'] if weather else '--'} mm/h")

    # --- LOGIC TÍNH TOÁN DRIFT ---
    drift_speed_kmh = velocity + (wind_speed * 3.6 * 0.03)
    total_drift_meters = (drift_speed_kmh / 60) * time_lost * 1000
    bearing = math.radians(wind_dir)
    offset_dist = total_drift_meters / 2 
    new_lat = lat + (offset_dist * math.cos(bearing) / 111111)
    new_lon = lon + (offset_dist * math.sin(bearing) / (111111 * math.cos(math.radians(lat))))

    # --- PHẦN PHÂN TÍCH AI CHUYÊN SÂU ---
    st.divider()
    st.subheader("🤖 Phân tích AI & Cảnh báo thiên tai")
    
    if st.button("Kích hoạt AI Phân tích rủi ro"):
        with st.status("Đang quét dữ liệu đa tầng...", expanded=True) as status:
            time.sleep(0.5)
            st.write("🔍 Đang phân tích lượng mưa và tầm nhìn...")
            time.sleep(0.5)
            st.write("🌡️ Đang tính toán rủi ro hạ thân nhiệt...")
            status.update(label="Hoàn tất phân tích!", state="complete")

        # Logic đánh giá tình trạng thiên tai/nguy hiểm
        is_heavy_rain = weather['rain'] > 5 if weather else False
        is_cold = weather['temp'] < 20 if weather else False
        is_low_visibility = weather['visibility'] < 2 if weather else False
        
        danger_msg = "🚨 CẢNH BÁO NGUY HIỂM:" if (is_heavy_rain or is_low_visibility) else "✅ ĐIỀU KIỆN ỔN ĐỊNH:"
        
        advice = []
        if is_heavy_rain: advice.append("- Mưa lớn làm giảm hiệu quả radar và tầm nhìn bằng mắt.")
        if is_low_visibility: advice.append("- Tầm nhìn cực thấp. Cần sử dụng đèn tín hiệu công suất cao.")
        if is_cold: advice.append("- Cảnh báo: Nguy cơ nạn nhân bị hạ thân nhiệt nhanh (Hypothermia).")
        if not advice: advice.append("- Điều kiện môi trường thuận lợi cho việc tìm kiếm.")

        st.warning(f"""
        ### {danger_msg}
        * **Trạng thái:** {weather['description'].capitalize() if weather else 'Không xác định'}
        * **Phân tích:** Lượng mưa {weather['rain'] if weather else 0}mm/h ảnh hưởng đến khả năng cơ động của cano cứu hộ.
        * **Vùng tìm kiếm:** Đã dịch chuyển {offset_dist:.0f}m theo hướng {wind_dir}°.
        * **Khuyến nghị từ AI:**
        {chr(10).join(advice)}
        """)

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
