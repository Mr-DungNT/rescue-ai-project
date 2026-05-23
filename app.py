import streamlit as st
import pandas as pd
import re
import time
import math
import requests
import pydeck as pdk
import numpy as np

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

st.title("HỆ THỐNG CỨU HỘ AI TÍCH HỢP DỮ LIỆU ĐỊA HÌNH & THỜI TIẾT")
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
        st.sidebar.error("⚠️ Đang dùng dữ liệu dự phòng")
        wind_speed = 5.0
        wind_dir = 45

    time_lost = st.sidebar.slider("Thời gian mất tín hiệu (phút)", 5, 120, 30)

    # --- HIỂN THỊ CHỈ SỐ NHANH ---
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Vận tốc trung bình", f"{velocity} km/h")
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
    st.subheader("🤖 Phân tích AI & Chiến thuật Cứu hộ chuyên nghiệp")
    
    # Nút bấm kích hoạt trạng thái
    if st.button("Kích hoạt AI Phân tích rủi ro & Tọa độ"):
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
        ### 🎯 TỌA ĐỘ MỤC TIÊU ƯU TIÊN:
        * **Tâm vùng tìm kiếm (Datum):** `{new_lat:.6f}, {new_lon:.6f}`
        * **Độ lệch dự kiến:** Dịch chuyển {offset_dist:.0f}m về hướng {wind_dir}°.

        ### 📋 PHÂN TÍCH CHUYÊN MÔN:
        1. **Thời gian vàng cứu hộ:** Ước tính **{survival_time}** (Dựa trên nhiệt độ {weather['temp']}°C).
        2. **Tình trạng thiên tai:** {weather['description'].capitalize() if weather else 'N/A'}.
        3. **Cảnh báo:** { "Nguy cơ hạ thân nhiệt nhanh." if is_cold else "Nhiệt độ nước ổn định." }
        
        ### 🚤 CHIẾN THUẬT TÌM KIẾM ĐỀ XUẤT:
        * **Phương pháp:** Tìm kiếm theo hình xoắn ốc mở rộng (Expanding Square Search).
        * **Hành động:** Triển khai lực lượng quan sát tại tọa độ mục tiêu.
        """)
        st.code(f"{new_lat:.6f}, {new_lon:.6f}", language="text")

    # --- BẢN ĐỒ KHỐI 3D HEATMAP CHUYÊN SÂU (MỚI) ---
    st.divider()
    st.subheader("🔮 Bản đồ mô phỏng 3D Xác suất trôi dạt & Heatmap")
    st.caption("💡 Mẹo thuyết trình: Nhấn giữ nút chuột phải hoặc tổ hợp phím Ctrl + Chuột trái để xoay nghiêng/lật góc nhìn 3D.")

    # Giả lập ma trận điểm xác suất phân phối quanh tâm AI dự đoán để vẽ khối 3D Heatmap
    np.random.seed(42)
    num_points = 600
    std_dev = (total_drift_meters / 2) / 111111
    
    lats_sim = np.random.normal(new_lat, std_dev, num_points)
    lons_sim = np.random.normal(new_lon, std_dev / math.cos(math.radians(lat)), num_points)
    
    map_data = pd.DataFrame({
        "latitude": lats_sim,
        "longitude": lons_sim
    })

    # Cấu hình lớp bản đồ khối lục giác HexagonLayer 3D
    layer_3d = pdk.Layer(
        "HexagonLayer",
        data=map_data,
        get_position=["longitude", "latitude"],
        radius=35,            # Bán kính cột lục giác (mét)
        elevation_scale=12,   # Hệ số kéo cao cột khối dữ liệu
        elevation_range=[0, 1000],
        extruded=True,        # Đổ khối 3D
        pickable=True,
        coverage=1,
        # Dải màu Heatmap chuyển từ Xanh (Lạnh) -> Đỏ rực (Nóng/Xác suất cao)
        color_range=[
            [65, 182, 196, 180],
            [127, 205, 187, 200],
            [199, 233, 180, 220],
            [252, 141, 89, 230],
            [227, 26, 28, 240],
            [177, 0, 38, 255]
        ]
    )

    # Định vị Camera góc nhìn nghiêng 3D mặc định khi load trang
    view_state = pdk.ViewState(
        latitude=new_lat,
        longitude=new_lon,
        zoom=14.5,
        pitch=55,             # Góc nghiêng camera (55 độ)
        bearing=-15           # Góc xoay la bàn bản đồ
    )

    # Đẩy biểu đồ Pydeck lên giao diện với lớp bản đồ vệ tinh Mapbox
    st.pydeck_chart(pdk.Deck(
        layers=[layer_3d],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/satellite-v9",
        tooltip={"text": "Mật độ xác suất: {count} điểm tin cậy"}
    ))

else:
    st.info("👋 Chào mừng! Hãy tải file Excel để bắt đầu cập nhật dữ liệu thiên tai thời gian thực.")
    st.image("https://capovelo.com/wp-content/uploads/2021/04/newmaps.jpeg", caption="Hệ thống trực chiến 24/7", use_container_width=True)
