import streamlit as st
import google.generativeai as genai
import pandas as pd

# Cấu hình Gemini API
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash')

st.title("Chứng Khoán AI Dashboard")

# Giả lập một DataFrame dữ liệu kỹ thuật sau khi quét từ API chứng khoán hoặc TradingView Webhook
# Trong thực tế, bạn sẽ dùng Pandas để tính toán các giá trị này real-time
stock_data = {
    "HPG": {"price": 28500, "rsi": 42.5, "macd": "Cắt xuống", "support": "27000 - 27500", "resistance": "30000"},
    "SSI": {"price": 35200, "rsi": 68.1, "macd": "Cắt lên", "support": "33000", "resistance": "37500"}
}

# Giao diện chọn mã cổ phiếu trên bảng điện
ticker = st.selectbox("Chọn mã cổ phiếu cần theo dõi:", list(stock_data.keys()))

# Hiển thị thông tin cơ bản trên dashboard
data = stock_data[ticker]
st.write(f"**Giá hiện tại:** {data['price']} | **RSI:** {data['rsi']} | **Hỗ trợ:** {data['support']}")

# Khung chat hỏi đáp với Gemini
st.subheader(f"Hỏi đáp chuyên gia về mã {ticker}")
user_question = st.text_input("Ví dụ: Có nên mua gom ở vùng giá này không? Target bao nhiêu?")

if st.button("Gửi câu hỏi"):
    if user_question:
        with st.spinner("AI đang phân tích đồ thị..."):
            # Xây dựng prompt kèm ngữ cảnh dữ liệu cứng từ hệ thống
            full_prompt = f"""
            Cổ phiếu: {ticker}
            Giá hiện tại: {data['price']}
            RSI: {data['rsi']}
            Trạng thái MACD: {data['macd']}
            Vùng hỗ trợ: {data['support']}
            Vùng kháng cự: {data['resistance']}
            
            Câu hỏi: {user_question}
            
            Hãy dựa vào các thông số kỹ thuật trên để tư vấn chiến lược mua/bán và mức giá cụ thể phù hợp với câu hỏi.
            """
            
            response = model.generate_content(full_prompt)
            st.markdown(response.text)
    else:
        st.warning("Vui lòng nhập câu hỏi.")
