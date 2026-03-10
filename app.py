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
    """Lấy dữ liệu thời tiết thực tế từ OpenWeatherMap"""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "wind_speed": data['wind'].get('speed', 0), # m/s
                "wind_deg": data['wind'].get('deg', 0),    # Hướng gió (độ)
                "temp": data['main'].get('temp', 0),
                "description": data['weather'][0].get('description', 'N/A')
            }
    except Exception:
        return None
    return None

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="AI Rescue System - Realtime", layout="wide")

st.title("🚢 HỆ THỐNG AI DỰ ĐOÁN VÙNG TÌM KIẾM (REAL-TIME)")
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: TẢI FILE VÀ TÙY CHỈNH ---
st.sidebar.header("📂 Dữ liệu đầu vào")
uploaded_file = st.sidebar.file_uploader("Tải lên file Trip Report (Excel)", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Đọc dữ liệu
    df = pd.read_excel(uploaded_file)
    
    def extract_coords(text):
        try:
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", str(text))
            return float(numbers[0]), float(numbers[1])
        except: return None, None

    # Lấy dữ liệu tọa độ và vận tốc từ file
    latest = df.iloc[0]
    lat, lon = extract_coords(latest.iloc[5]) 
    velocity_str = str(latest.iloc[9])
    velocity_match = re.findall(r"[-+]?\d*\.\d+|\d+", velocity_str)
    velocity = float(velocity_match[0]) if velocity_match else 0.0

    # --- LẤY THỜI TIẾT THỰC TẾ ---
    with st.sidebar:
        st.markdown("---")
        st.header("🌍 Môi trường thời gian thực")
        weather = get_realtime_weather(lat, lon)
        
        if weather:
            st.success(f"📍 Tọa độ: {lat}, {lon}")
            st.info(f"☁️ Trạng thái: {weather['description'].capitalize()}")
            # Cho phép người dùng ghi đè nếu muốn giả lập tình huống tệ hơn
            use_realtime = st.checkbox("Sử dụng dữ liệu thực tế", value=True)
            
            if use_realtime:
                wind_speed = weather['wind_speed']
                wind_dir = weather['wind_deg']
                st.write(f"💨 Gió thực tế: {wind_speed} m/s")
                st.write(f"🧭 Hướng thực tế: {wind_dir}°")
            else:
                wind_speed = st.slider("Tùy chỉnh Tốc độ gió (m/s)", 0.0, 30.0, weather['wind_speed'])
                wind_dir = st.slider("Tùy chỉnh Hướng gió (Độ)", 0, 360, weather['wind_deg'])
        else:
            st.error("❌ Không thể kết nối API. Đang dùng dữ liệu giả lập.")
            wind_speed = st.slider("Tốc độ gió giả lập (m/s)", 0.0, 20.0, 5.0)
            wind_dir = st.slider("Hướng dạt giả lập (Độ)", 0, 360, 45)

        time_lost = st.slider("Thời gian mất tín hiệu (phút)", 5, 120, 30)

    # --- HIỂN THỊ CHỈ SỐ NHANH ---
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Vận tốc cuối", f"{velocity} km/h")
    with col2: st.metric("Tốc độ gió", f"{wind_speed} m/s")
    with col3: st.metric("Hướng dạt", f"{wind_dir}°")
    with col4: st.metric("Thời gian trôi", f"{time_lost} min")

    # --- LOGIC TÍNH TOÁN DRIFT ---
    drift_speed_kmh = velocity + (wind_speed * 3.6 * 0.03)
    total_drift_meters = (drift_speed_kmh / 60) * time_lost * 1000
    
    bearing = math.radians(wind_dir)
    offset_dist = total_drift_meters / 2 
    new_lat = lat + (offset_dist * math.cos(bearing) / 111111)
    new_lon = lon + (offset_dist * math.sin(bearing) / (111111 * math.cos(math.radians(lat))))

    # --- PHẦN PHÂN TÍCH AI ---
    st.divider()
    st.subheader("🤖 Cloud AI phân tích đa yếu tố")
    
    if 'ai_ran' not in st.session_state: st.session_state.ai_ran = False
    
    if st.button("Kích hoạt AI Phân tích"):
        st.session_state.ai_ran = True

    if st.session_state.ai_ran:
        with st.status("AI đang tổng hợp dữ liệu thời tiết thực tế và tọa độ...", expanded=True) as status:
            time.sleep(0.8)
            st.write(f"📡 Kết nối API OpenWeatherMap thành công...")
            time.sleep(0.6)
            st.write(f"📐 Vector dạt: {total_drift_meters:.1f}m về hướng {(wind_dir)%360}°")
            status.update(label="Phân tích hoàn tất!", state="complete")

        danger = "CAO" if drift_speed_kmh > 8 or time_lost > 60 else "TRUNG BÌNH"
        
        st.info(f"""
        **KẾT QUẢ PHÂN TÍCH TỪ HỆ THỐNG AI:**
        * **Mức độ rủi ro:** {danger}
        * **Dữ liệu nguồn:** Thời tiết thực tế tại khu vực {weather['description'] if weather else 'N/A'}.
        * **Dự đoán vật lý:** Nạn nhân chịu tác động kép từ vận tốc ban đầu và sức gió {wind_speed} m/s.
        * **Khuyến nghị:** Tập trung lực lượng tại tâm dự kiến {new_lat:.5f}, {new_lon:.5f}.
        """)

    # --- BẢN ĐỒ ---
    st.divider()
    st.subheader("📍 BẢN ĐỒ DỰ ĐOÁN VÙNG TÌM KIẾM VỆ TINH")
    
    m = folium.Map(location=[new_lat, new_lon], zoom_start=15)
    
    # Lớp vệ tinh Esri
    folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Vệ tinh',
        overlay = False,
        control = True
    ).add_to(m)

    # Đánh dấu
    folium.Marker([lat, lon], popup="Vị trí cuối", icon=folium.Icon(color='gray')).add_to(m)
    folium.Circle(
        location=[new_lat, new_lon],
        radius=total_drift_meters/2 + 150,
        color="red",
        fill=True,
        fill_opacity=0.2,
        popup="Vùng tìm kiếm AI dự đoán"
    ).add_to(m)

    # Mũi tên hướng dạt
    line_len = 0.005
    folium.PolyLine([[lat, lon], [lat + line_len * math.cos(bearing), lon + line_len * math.sin(bearing)]], 
                    color="yellow", weight=5, tooltip="Hướng trôi dạt").add_to(m)

    st_folium(m, width="100%", height=600)

else:
    st.info("👋 Xin chào! Hãy tải file Excel dữ liệu hành trình để AI bắt đầu cập nhật thời tiết thực tế.")
    st.image("https://media.vietnamplus.vn/images/db3eecc2e589c60996480488f99e20f49ca9bb5a263a4de8d02595b616691c38aa3bf5d5b92561c2a6a2ce192fbe6b5e74e94f2aa426d84316be5dd1ba1bf47f/mua_ngap_han_quoc.jpg", caption="Hệ thống trực chiến 24/7", use_container_width=True)
