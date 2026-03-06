import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import re

st.set_page_config(page_title="AI Rescue System", layout="wide")

st.title("🚢 Hệ thống Giám sát & Dự đoán Cứu hộ AI")

# CHỖ NÀY ĐÃ ĐỔI SANG EXCEL
uploaded_file = st.file_uploader("Tải lên file Trip Report (Excel)", type=["xlsx", "xls"])

if uploaded_file is not None:
    # ĐỌC FILE EXCEL
    df = pd.read_excel(uploaded_file)
    
    def extract_coords(text):
        try:
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", str(text))
            return float(numbers[0]), float(numbers[1])
        except: return None, None

    # Lấy dòng mới nhất (Dòng đầu tiên sau tiêu đề)
    latest = df.iloc[0]
    
    # Dựa trên file của cậu: Cột số 5 là Start, Cột số 9 là Velocity
    lat, lon = extract_coords(latest.iloc[5]) 
    velocity_val = str(latest.iloc[9])
    velocity = float(re.findall(r"[-+]?\d*\.\d+|\d+", velocity_val)[0])

    # Hiển thị thông số trên thanh bên
    st.sidebar.header("Thông số thực tế")
    st.sidebar.metric("Vận tốc cuối", f"{velocity} km/h")
    
    # Thanh trượt AI
    time_lost = st.sidebar.slider("Thời gian mất tín hiệu (phút)", 0, 120, 30)
    # Công thức dự đoán vùng tìm kiếm
    radius = (velocity / 60) * time_lost * 1000 

    # Vẽ bản đồ
    st.subheader(f"Vị trí cuối ghi nhận: {lat}, {lon}")
    m = folium.Map(location=[lat, lon], zoom_start=15)
    
    # Vị trí cuối
    folium.Marker([lat, lon], popup="Nạn nhân", icon=folium.Icon(color='red', icon='user')).add_to(m)
    
    # Vùng dự đoán AI
    folium.Circle(
        [lat, lon], 
        radius=radius, 
        color="red", 
        fill=True, 
        fill_opacity=0.2,
        popup="Vùng tìm kiếm khả thi"
    ).add_to(m)
    
    st_folium(m, width="100%", height=500)
    st.info(f"Dựa trên vận tốc {velocity}km/h, vùng tìm kiếm sau {time_lost} phút có bán kính khoảng {radius:.1f} mét.")

else:
    st.warning("👈 Cậu hãy chọn file Excel từ máy tính để chạy demo nhé!")
