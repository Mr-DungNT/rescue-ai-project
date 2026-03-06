import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import re

st.set_page_config(page_title="AI Rescue System", layout="wide")

st.title("🚢 Hệ thống Dự đoán Cứu hộ AI")
st.markdown("---")

# 1. Chức năng Upload file trực tiếp lên Web
uploaded_file = st.file_uploader("Tải lên file Trip Report (CSV)", type="csv")

if uploaded_file is not None:
    # Đọc dữ liệu và bỏ qua các dòng tiêu đề thừa nếu có
    df = pd.read_csv(uploaded_file)
    
    # Hàm xử lý chuỗi tọa độ "21.xxx, 105.xxx" thành số thực
    def extract_coords(text):
        try:
            # Tìm tất cả số thực trong chuỗi
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", str(text))
            return float(numbers[0]), float(numbers[1])
        except:
            return None, None

    # Lấy dữ liệu dòng mới nhất (Dòng đầu tiên trong file của cậu)
    latest_data = df.iloc[0]
    
    # Trích xuất tọa độ và vận tốc
    start_lat, start_lon = extract_coords(latest_data.iloc[5]) # Cột Start Coord
    end_lat, end_lon = extract_coords(latest_data.iloc[6])     # Cột End Coord
    
    # Xử lý vận tốc (Xóa chữ 'km/h' để lấy số)
    velocity_str = str(latest_data.iloc[9])
    velocity = float(re.findall(r"[-+]?\d*\.\d+|\d+", velocity_str)[0])

    # 2. Giao diện hiển thị chỉ số
    col1, col2, col3 = st.columns(3)
    col1.metric("Vị trí cuối cùng", f"{end_lat}, {end_lon}")
    col2.metric("Vận tốc cuối", f"{velocity} km/h")
    
    # 3. Thuật toán AI dự đoán vùng tìm kiếm
    st.sidebar.header("Cấu hình Dự đoán AI")
    time_lost = st.sidebar.slider("Thời gian mất liên lạc (phút)", 5, 120, 30)
    
    # Tính bán kính dự đoán: R = (Vận tốc / 60) * Thời gian
    # Đổi sang mét để vẽ vòng tròn
    predict_radius = (velocity / 60) * time_lost * 1000 

    # 4. Vẽ bản đồ
    st.subheader("Bản đồ cứu hộ trực tuyến")
    m = folium.Map(location=[end_lat, end_lon], zoom_start=15)

    # Đánh dấu vị trí cuối cùng
    folium.Marker(
        [end_lat, end_lon], 
        popup="Vị trí cuối cùng còn tín hiệu",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)

    # Vẽ vòng tròn dự đoán của AI (Vùng tìm kiếm)
    folium.Circle(
        location=[end_lat, end_lon],
        radius=predict_radius,
        color="crimson",
        fill=True,
        fill_color="red",
        fill_opacity=0.3,
        popup=f"Vùng tìm kiếm sau {time_lost} phút"
    ).add_to(m)

    # Hiển thị bản đồ
    st_folium(m, width="100%", height=500)
    
    st.success(f"AI xác định vùng tìm kiếm có bán kính: {predict_radius:.2f} mét")
else:
    st.info("Vui lòng tải file dữ liệu lên để bắt đầu phân tích.")
