import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import re
import google.generativeai as genai

# --- CẤU HÌNH GEMINI AI ---
# Sử dụng API Key của bạn
genai.configure(api_key="AIzaSyAMAVgxsEGeHeC9WxINnR3NZOVlRQ1llrQ")

# Khởi tạo model với đường dẫn đầy đủ để tránh lỗi 404
model = genai.GenerativeModel('models/gemini-1.5-flash')

st.set_page_config(page_title="AI Rescue System", layout="wide")

st.title("🚢 Hệ thống Giám sát & Dự đoán Cứu hộ AI")
st.markdown("---")

# Tải lên file Excel trực tiếp (Trip Report)
uploaded_file = st.file_uploader("Tải lên file Trip Report (Excel)", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Đọc dữ liệu từ Excel
    df = pd.read_excel(uploaded_file)
    
    # Hàm tách tọa độ từ chuỗi văn bản
    def extract_coords(text):
        try:
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", str(text))
            return float(numbers[0]), float(numbers[1])
        except:
            return None, None

    # Lấy dữ liệu dòng mới nhất (Dòng đầu tiên trong file của bạn)
    latest = df.iloc[0]
    
    # Trích xuất Lat/Lon từ cột số 5 (Start) và Vận tốc từ cột số 9 (Average velocity)
    lat, lon = extract_coords(latest.iloc[5]) 
    velocity_str = str(latest.iloc[9])
    velocity_match = re.findall(r"[-+]?\d*\.\d+|\d+", velocity_str)
    velocity = float(velocity_match[0]) if velocity_match else 0.0

    # Thiết lập giao diện Sidebar
    st.sidebar.header("📊 Thông số thực tế")
    st.sidebar.metric("Vận tốc cuối cùng", f"{velocity} km/h")
    
    # Thanh trượt giả định thời gian mất tích để AI tính toán vùng tìm kiếm
    time_lost = st.sidebar.slider("Thời gian mất tín hiệu (phút)", 0, 120, 30)
    
    # Công thức tính bán kính vùng tìm kiếm (Vận tốc * thời gian)
    radius = (velocity / 60) * time_lost * 1000 

    # --- PHẦN PHÂN TÍCH AI GEMINI ---
    st.subheader("🤖 Chuyên gia cứu hộ AI phân tích")
    if st.button("Nhấn để Gemini đưa ra lời khuyên cứu hộ"):
        with st.spinner('Đang kết nối với trí tuệ nhân tạo Gemini...'):
            prompt = f"""
            Bạn là một sĩ quan chỉ huy cứu hộ hàng hải cao cấp. 
            Dữ liệu thực tế từ thiết bị của nạn nhân:
            - Tọa độ cuối cùng ghi nhận: {lat}, {lon}
            - Vận tốc di chuyển trước khi mất tín hiệu: {velocity} km/h
            - Thời gian đã trôi qua kể từ khi mất tích: {time_lost} phút
            - Vùng tìm kiếm dự kiến: bán kính khoảng {radius:.1f} mét xung quanh tọa độ cuối.

            Dựa trên dữ liệu này, hãy thực hiện:
            1. Đánh giá mức độ nguy hiểm đối với nạn nhân (Thấp/Trung bình/Cao).
            2. Đưa ra 3 hành động khẩn cấp ưu tiên cho đội cứu hộ.
            3. Dự đoán các rủi ro sức khỏe nạn nhân có thể gặp phải (hạ thân nhiệt, kiệt sức...).
            4. Lời khuyên về hướng tiếp cận dựa trên vị trí tọa độ.

            Yêu cầu: Trả lời bằng tiếng Việt, ngôn ngữ chuyên nghiệp, ngắn gọn và quyết đoán.
            """
            try:
                response = model.generate_content(prompt)
                st.success("AI đã phân tích xong tình huống!")
                st.info(response.text)
            except Exception as e:
                st.error(f"Lỗi kết nối AI: {e}. Vui lòng kiểm tra lại cấu hình model.")

    # --- PHẦN BẢN ĐỒ ---
    st.divider()
    st.subheader(f"📍 Bản đồ vùng tìm kiếm dự kiến")
    st.write(f"Tọa độ tâm điểm: **{lat}, {lon}** | Bán kính quét: **{radius:.1f} mét**")
    
    # Khởi tạo bản đồ Folium
    m = folium.Map(location=[lat, lon], zoom_start=15)
    
    # Đánh dấu vị trí cuối cùng
    folium.Marker(
        [lat, lon], 
        popup="Vị trí cuối ghi nhận", 
        icon=folium.Icon(color='red', icon='warning', prefix='fa')
    ).add_to(m)
    
    # Vẽ vòng tròn dự đoán AI (vùng tìm kiếm)
    folium.Circle(
        [lat, lon], 
        radius=radius, 
        color="red", 
        fill=True, 
        fill_opacity=0.2,
        popup="Vùng khả nghi theo vận tốc dạt"
    ).add_to(m)
    
    # Hiển thị bản đồ lên Web
    st_folium(m, width="100%", height=550)
    
    st.caption("Lưu ý: Vùng đỏ thể hiện phạm vi di chuyển tối đa dựa trên vận tốc ghi nhận cuối cùng.")

else:
    st.warning("👈 Vui lòng tải file Excel 'Trip Report' lên từ thanh bên trái để bắt đầu.")
