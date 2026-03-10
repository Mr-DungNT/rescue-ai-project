import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import re
import time
import math

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="AI Rescue System - Pro", layout="wide")

st.title("🚢 Hệ thống Giám sát & Dự đoán Cứu hộ AI (Tích hợp Môi trường)")
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
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

    # Lấy dữ liệu dòng mới nhất (thường là dòng 0 trong Trip Report)
    latest = df.iloc[0]
    lat, lon = extract_coords(latest.iloc[5]) 
    
    # Trích xuất vận tốc
    velocity_str = str(latest.iloc[9])
    velocity_match = re.findall(r"[-+]?\d*\.\d+|\d+", velocity_str)
    velocity = float(velocity_match[0]) if velocity_match else 0.0

    # --- BỘ GIẢ LẬP MÔI TRƯỜNG (KHÔNG CẦN API - SIÊU ỔN ĐỊNH) ---
    st.sidebar.markdown("---")
    st.sidebar.header("🌊 Thông số Môi trường")
    sea_state = st.sidebar.selectbox("Tình trạng biển", ["Yên tĩnh (Lực cản thấp)", "Sóng nhẹ (Dòng chảy trung bình)", "Biển động (Dòng chảy mạnh)"])
    
    # Giả lập tốc độ gió dựa trên tình trạng biển
    if sea_state == "Yên tĩnh (Lực cản thấp)": wind_speed = 2.5
    elif sea_state == "Sóng nhẹ (Dòng chảy trung bình)": wind_speed = 6.0
    else: wind_speed = 13.5
    
    wind_dir = st.sidebar.slider("Hướng Gió/Dòng chảy (Độ)", 0, 360, 45)
    time_lost = st.sidebar.slider("Thời gian mất tín hiệu (phút)", 5, 120, 30)

    # --- HIỂN THỊ CHỈ SỐ NHANH ---
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Vận tốc cuối", f"{velocity} km/h")
    with col2: st.metric("Tốc độ gió", f"{wind_speed} m/s")
    with col3: st.metric("Hướng dạt", f"{wind_dir}°")
    with col4: st.metric("Thời gian trôi", f"{time_lost} min")

    # --- LOGIC TÍNH TOÁN DRIFT (VÉCTƠ TRÔI DẠT) ---
    # Drift speed = Vận tốc tàu + (3% vận tốc gió)
    drift_speed_kmh = velocity + (wind_speed * 3.6 * 0.03)
    total_drift_meters = (drift_speed_kmh / 60) * time_lost * 1000
    
    # Tính tọa độ tâm vùng tìm kiếm mới (lệch theo hướng gió)
    bearing = math.radians(wind_dir)
    # 1 độ vĩ độ ~ 111,111m
    offset_dist = total_drift_meters / 2 
    new_lat = lat + (offset_dist * math.cos(bearing) / 111111)
    new_lon = lon + (offset_dist * math.sin(bearing) / (111111 * math.cos(math.radians(lat))))

    # --- PHẦN PHÂN TÍCH AI CHUYÊN GIA ---
    st.divider()
    st.subheader("🤖 Chuyên gia cứu hộ AI phân tích đa yếu tố")
    
    if 'ai_ran' not in st.session_state: st.session_state.ai_ran = False
    
    if st.button("Kích hoạt AI Phân tích"):
        st.session_state.ai_ran = True

    if st.session_state.ai_ran:
        with st.status("AI đang tổng hợp dữ liệu môi trường và vị trí...", expanded=True) as status:
            time.sleep(0.8)
            st.write(f"📡 Đang kết nối trạm khí tượng ảo... (Gió: {wind_speed} m/s)")
            time.sleep(0.6)
            st.write(f"📐 Tính toán Vector trôi dạt: Hướng {wind_dir}°, Quãng đường {total_drift_meters:.1f}m")
            status.update(label="Phân tích hoàn tất!", state="complete")

        danger = "CAO" if drift_speed_kmh > 8 or time_lost > 60 else "TRUNG BÌNH"
        
        st.info(f"""
        **KẾT QUẢ PHÂN TÍCH TỪ HỆ THỐNG AI:**
        * **Mức độ rủi ro:** {danger}
        * **Dự đoán vật lý:** Dưới tác động của hướng gió {wind_dir}°, nạn nhân trôi dạt về phía {(wind_dir)%360}° với tốc độ tổng hợp {drift_speed_kmh:.2f} km/h.
        * **Khuyến nghị vùng tìm kiếm:** Tâm vùng tìm kiếm đã được dịch chuyển {offset_dist:.0f}m về phía hạ lưu.
        * **Hành động ưu tiên:** Triển khai lực lượng quan sát tại tọa độ dự kiến: {new_lat:.5f}, {new_lon:.5f}.
        """)

    # --- BẢN ĐỒ TRỰC QUAN ---
    st.divider()
    st.subheader("📍 Bản đồ Dự đoán vùng trôi dạt (Drift Map)")
    
    # Tạo bản đồ, tập trung vào vùng tìm kiếm mới
    m = folium.Map(location=[new_lat, new_lon], zoom_start=15, control_scale=True)
    
    # Thêm lớp bản đồ vệ tinh để nhìn chuyên nghiệp hơn
    folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Vệ tinh',
        overlay = False,
        control = True
    ).add_to(m)

    # Vị trí cuối cùng trước khi mất tín hiệu
    folium.Marker(
        [lat, lon], 
        popup="Vị trí cuối", 
        icon=folium.Icon(color='gray', icon='info-sign')
    ).add_to(m)
    
    # Vùng tìm kiếm dự kiến (Hình tròn lệch tâm)
    folium.Circle(
        location=[new_lat, new_lon],
        radius=total_drift_meters/2 + 150, # Bán kính bao phủ sai số
        color="red",
        weight=3,
        fill=True,
        fill_color="red",
        fill_opacity=0.2,
        popup=f"Vùng trôi dạt dự kiến sau {time_lost} phút"
    ).add_to(m)

    # Vẽ mũi tên hướng gió (giả lập đơn giản bằng đường thẳng)
    line_len = 0.005
    end_lat = lat + line_len * math.cos(bearing)
    end_lon = lon + line_len * math.sin(bearing)
    folium.PolyLine([[lat, lon], [end_lat, end_lon]], color="yellow", weight=5, opacity=0.8, tooltip="Hướng gió/dòng chảy").add_to(m)

    st_folium(m, width="100%", height=600)

else:
    st.info("👋 Chào mừng cậu! Hãy tải file Excel dữ liệu hành trình lên ở thanh bên trái để AI bắt đầu phân tích vùng cứu hộ.")
    # st.image("https://img.freepik.com/free-vector/modern-world-map-background_1035-18967.jpg", use_column_width=True)
st.image("https://www.vietnamplus.vn/mua-lu-ngap-lut-han-han-hoanh-hanh-tai-nhieu-khu-vuc-tren-the-gioi-post888450.vnp", caption="Hệ thống trực chiến 24/7", use_column_width=True)
