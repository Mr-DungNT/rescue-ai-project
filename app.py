import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import re
import google.generativeai as genai
from google.generativeai.types import RequestOptions

# --- CẤU HÌNH GEMINI AI ---
# Sử dụng API Key đã xác thực của cậu
genai.configure(api_key="AIzaSyAMAVgxsEGeHeC9WxINnR3NZOVlRQ1llrQ")

# Thiết lập Model với cấu hình tối ưu
model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    generation_config={
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 1024,
    }
)

# Hàm gọi AI có ép phiên bản API v1 để tránh lỗi 404
def get_ai_rescue_advice(lat, lon, velocity, time_lost, radius):
    prompt = f"""
    Bạn là một sĩ quan chỉ huy cứu hộ hàng hải cao cấp. 
    Dữ liệu thực tế từ thiết bị của nạn nhân:
    - Tọa độ cuối cùng: {lat}, {lon}
    - Vận tốc di chuyển: {velocity} km/h
    - Thời gian mất tín hiệu: {time_lost} phút
    - Vùng tìm kiếm dự kiến: bán kính {radius:.1f} mét.

    Hãy thực hiện:
    1. Đánh giá mức độ nguy hiểm (Thấp/Trung bình/Cao).
    2. Đưa ra 3 hành động khẩn cấp cho đội cứu hộ.
    3. Dự đoán rủi ro sức khỏe (hạ thân nhiệt, trôi dạt...).
    Trả lời ngắn gọn, chuyên nghiệp bằng tiếng Việt.
    """
    try:
        # Ép sử dụng api_version='v1' để sửa lỗi 404 models/not found
        response = model.generate_content(
            prompt, 
            request_options=RequestOptions(api_version='v1')
        )
        return response.text
    except Exception as e:
        return f"Lỗi kết nối AI: {e}. Vui lòng kiểm tra lại cấu hình phiên bản API."

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

    # Lấy dữ liệu từ dòng đầu tiên
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
        with st.spinner('Gemini đang phân tích dữ liệu...'):
            advice = get_ai_rescue_advice(lat, lon, velocity, time_lost, radius)
            st.info(advice)

    # --- BẢN ĐỒ ---
    st.divider()
    st.subheader(f"📍 Bản đồ vùng tìm kiếm dự kiến")
    m = folium.Map(location=[lat, lon], zoom_start=15)
    folium.Marker([lat, lon], popup="Nạn nhân", icon=folium.Icon(color='red', icon='warning')).add_to(m)
    folium.Circle([lat, lon], radius=radius, color="red", fill=True, fill_opacity=0.2).add_to(m)
    st_folium(m, width="100%", height=500)

else:
    st.warning("👈 Vui lòng tải file Excel lên để bắt đầu.")
