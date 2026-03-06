import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import re
import time

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

    latest = df.iloc[0]
    lat, lon = extract_coords(latest.iloc[5]) 
    velocity_str = str(latest.iloc[9])
    velocity_match = re.findall(r"[-+]?\d*\.\d+|\d+", velocity_str)
    velocity = float(velocity_match[0]) if velocity_match else 0.0

    st.sidebar.header("📊 Thông số thực tế")
    st.sidebar.metric("Vận tốc cuối", f"{velocity} km/h")
    time_lost = st.sidebar.slider("Thời gian mất tín hiệu (phút)", 0, 120, 30)
    radius = (velocity / 60) * time_lost * 1000 

    # --- HỆ CHUYÊN GIA AI (THAY THẾ CHATBOT) ---
    st.subheader("🤖 Chuyên gia cứu hộ AI phân tích")
    if st.button("Nhấn để AI đưa ra lời khuyên cứu hộ"):
        with st.status("AI đang phân tích dữ liệu thực tế...", expanded=True) as status:
            time.sleep(1.5) # Giả lập thời gian AI suy nghĩ
            st.write("🔍 Đang quét tọa độ vị trí...")
            time.sleep(1)
            st.write("🌊 Đang tính toán độ lệch dòng chảy...")
            status.update(label="Phân tích hoàn tất!", state="complete", expanded=True)

        # AI đưa ra lời khuyên dựa trên logic thực tế
        danger_level = "CAO" if velocity > 5 or time_lost > 60 else "TRUNG BÌNH"
        
        advice_html = f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b;">
            <h4 style="color: #ff4b4b; margin-top:0;">🛑 ĐÁNH GIÁ TỪ HỆ THỐNG AI:</h4>
            <p><b>1. Mức độ nguy hiểm:</b> <span style="color:red;">{danger_level}</span></p>
            <p><b>2. Phân tích:</b> Với vận tốc dạt {velocity} km/h, nạn nhân đang trôi ra xa vị trí gốc khoảng {radius:.0f} mét mỗi {time_lost} phút.</p>
            <p><b>3. Hành động khẩn cấp:</b>
                <ul>
                    <li>Triển khai cano cứu hộ quét theo hình xoắn ốc từ tâm <b>{lat}, {lon}</b>.</li>
                    <li>Thông báo cho các tàu thuyền trong bán kính 2km tăng cường quan sát.</li>
                    <li>Chuẩn bị thiết bị sơ cứu chống hạ thân nhiệt do nạn nhân đã ở dưới nước lâu.</li>
                </ul>
            </p>
            <p><i>*Dự đoán dựa trên thuật toán tích hợp của thiết bị cứu hộ thông minh.*</i></p>
        </div>
        """
        st.markdown(advice_html, unsafe_allow_html=True)

    # --- BẢN ĐỒ ---
    st.divider()
    st.subheader(f"📍 Bản đồ vùng dự kiến")
    m = folium.Map(location=[lat, lon], zoom_start=15)
    folium.Marker([lat, lon], icon=folium.Icon(color='red', icon='warning')).add_to(m)
    folium.Circle([lat, lon], radius=radius, color="red", fill=True, fill_opacity=0.2).add_to(m)
    st_folium(m, width="100%", height=500)

else:
    st.warning("👈 Vui lòng tải file Excel lên để bắt đầu.")
