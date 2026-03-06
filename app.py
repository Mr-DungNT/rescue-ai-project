import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import re
import google.generativeai as genai

# --- CẤU HÌNH GEMINI AI (BẢN ĐƠN GIẢN HÓA ĐỂ TRÁNH LỖI) ---
genai.configure(api_key="AIzaSyAMAVgxsEGeHeC9WxINnR3NZOVlRQ1llrQ")

# Khởi tạo model theo cách tương thích rộng nhất
model = genai.GenerativeModel('gemini-1.5-flash')

# --- GIAO DIỆN WEB ---
st.set_page_config(page_title="AI Rescue System", layout="wide")
st.title("🚢 Hệ thống Giám sát & Dự đoán Cứu hộ AI")
st.markdown("---")

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
    velocity_match = re.findall(r"[-+]?\d*\.\d+|\d+", velocity_str)
    velocity = float(velocity_match[0]) if velocity_match else 0.0

    # Sidebar thông số
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
            Dữ liệu: Tọa độ {lat}, {lon}, vận tốc {velocity} km/h, bán kính tìm kiếm {radius:.1f}m.
            Hãy: 
            1. Đánh giá mức độ nguy hiểm.
            2. Đưa ra 3 hành động khẩn cấp.
            Trả lời ngắn gọn bằng tiếng Việt.
            """
            try:
                # Dùng lệnh cơ bản nhất để tránh lỗi version
                response = model.generate_content(prompt)
                st.info(response.text)
            except Exception as e:
                # Nếu vẫn lỗi 404, thử đổi sang tên model đầy đủ
                try:
                    alt_model = genai.GenerativeModel('models/gemini-1.5-flash')
                    response = alt_model.generate_content(prompt)
                    st.info(response.text)
                except:
                    st.error(f"Lỗi kết nối AI: {e}. Cậu hãy kiểm tra lại xem đã nhấn 'Reboot App' chưa.")

    # --- BẢN ĐỒ ---
    st.divider()
    st.subheader(f"📍 Bản đồ vùng tìm kiếm dự kiến")
    m = folium.Map(location=[lat, lon], zoom_start=15)
    folium.Marker([lat, lon], popup="Nạn nhân", icon=folium.Icon(color='red', icon='warning')).add_to(m)
    folium.Circle([lat, lon], radius=radius, color="red", fill=True, fill_opacity=0.2).add_to(m)
    st_folium(m, width="100%", height=500)

else:
    st.warning("👈 Vui lòng tải file Excel lên để bắt đầu.")
