import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import re
import google.generativeai as genai

# --- CẤU HÌNH GEMINI AI (BẢN FIX LỖI 404 & VERSION) ---
genai.configure(api_key="AIzaSyD0SwH5WRsfbpJqK-y32vtFKZe_vjzgJb4")

def call_gemini(prompt):
    # Thử danh sách các tên model từ mới đến cũ để tránh lỗi 404
    for model_name in ['gemini-1.5-flash', 'models/gemini-1.5-flash', 'gemini-pro']:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception:
            continue
    return "Không thể kết nối với các phiên bản AI. Cậu hãy kiểm tra lại kết nối mạng hoặc Reboot App."

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

    # Lấy dữ liệu dòng đầu tiên
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

    # --- NÚT BẤM GỌI AI ---
    st.subheader("🤖 Chuyên gia cứu hộ AI phân tích")
    if st.button("Nhấn để Gemini đưa ra lời khuyên cứu hộ"):
        with st.spinner('Gemini đang suy nghĩ...'):
            prompt = f"Bạn là chuyên gia cứu hộ. Nạn nhân ở {lat}, {lon}, vận tốc {velocity}km/h, vùng tìm kiếm {radius}m. Đưa ra 3 lời khuyên ngắn gọn bằng tiếng Việt."
            advice = call_gemini(prompt)
            st.info(advice)

    # --- BẢN ĐỒ ---
    st.divider()
    st.subheader(f"📍 Bản đồ vùng dự kiến")
    m = folium.Map(location=[lat, lon], zoom_start=15)
    folium.Marker([lat, lon], icon=folium.Icon(color='red', icon='warning')).add_to(m)
    folium.Circle([lat, lon], radius=radius, color="red", fill=True, fill_opacity=0.2).add_to(m)
    st_folium(m, width="100%", height=500)

else:
    st.warning("👈 Vui lòng tải file Excel lên để bắt đầu.")
