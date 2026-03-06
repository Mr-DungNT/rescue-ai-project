import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import re
import google.generativeai as genai

# --- CẤU HÌNH GEMINI AI ---
genai.configure(api_key="AIzaSyAMAVgxsEGeHeC9WxINnR3NZOVlRQ1llrQ")
model = genai.GenerativeModel('gemini-1.5-flash')

st.set_page_config(page_title="AI Rescue System", layout="wide")

st.title("🚢 Hệ thống Giám sát & Dự đoán Cứu hộ AI (Gemini Integrated)")

# Tải lên file Excel
uploaded_file = st.file_uploader("Tải lên file Trip Report (Excel)", type=["xlsx", "xls"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    
    def extract_coords(text):
        try:
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", str(text))
            return float(numbers[0]), float(numbers[1])
        except: return None, None

    # Lấy dữ liệu dòng mới nhất
    latest = df.iloc[0]
    lat, lon = extract_coords(latest.iloc[5]) 
    velocity_str = str(latest.iloc[9])
    velocity = float(re.findall(r"[-+]?\d*\.\d+|\d+", velocity_str)[0])

    # Hiển thị thông số bên trái
    st.sidebar.header("📊 Thông số thực tế")
    st.sidebar.metric("Vận tốc cuối", f"{velocity} km/h")
    time_lost = st.sidebar.slider("Thời gian mất tín hiệu (phút)", 0, 120, 30)
    radius = (velocity / 60) * time_lost * 1000 

    # --- PHẦN PHÂN TÍCH AI GEMINI ---
    st.subheader("🤖 Chuyên gia cứu hộ AI phân tích")
    if st.button("Nhấn để Gemini đưa ra lời khuyên cứu hộ"):
        with st.spinner('Đang kết nối với trí tuệ nhân tạo...'):
            prompt = f"""
            Bạn là một sĩ quan chỉ huy cứu hộ hàng hải cao cấp. 
            Dữ liệu từ thiết bị của nạn nhân:
            - Tọa độ cuối: {lat}, {lon}
            - Vận tốc di chuyển khi mất tín hiệu: {velocity} km/h
            - Thời gian đã trôi qua: {time_lost} phút
            - Vùng tìm kiếm dự kiến: bán kính {radius:.1f} mét xung quanh tọa độ cuối.

            Dựa trên dữ liệu này, hãy:
            1. Đánh giá mức độ nguy hiểm (Thấp/Trung bình/Cao).
            2. Đưa ra 3 hành động khẩn cấp cho đội cứu hộ.
            3. Dự đoán các rủi ro nạn nhân có thể gặp phải (hạ thân nhiệt, trôi dạt...).
            Yêu cầu: Trả lời bằng tiếng Việt, giọng văn chuyên nghiệp, ngắn gọn.
            """
            try:
                response = model.generate_content(prompt)
                st.success("Phân tích hoàn tất!")
                st.write(response.text)
            except Exception as e:
                st.error(f"Lỗi kết nối AI: {e}")

    # Vẽ bản đồ
    st.divider()
    st.subheader(f"📍 Bản đồ vùng tìm kiếm (Vị trí cuối: {lat}, {lon})")
    m = folium.Map(location=[lat, lon], zoom_start=15)
    folium.Marker([lat, lon], popup="Vị trí cuối", icon=folium.Icon(color='red', icon='info-sign')).add_to(m)
    folium.Circle([lat, lon], radius=radius, color="red", fill=True, fill_opacity=0.2).add_to(m)
    st_folium(m, width="100%", height=500)

else:
    st.info("👈 Vui lòng tải file Excel lên để hệ thống và AI bắt đầu làm việc!")
