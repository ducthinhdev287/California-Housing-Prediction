import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# --- PHẦN 1: HUẤN LUYỆN NHANH MÔ HÌNH (AUTO-TRAIN) ---
print("⏳ Đang tải dữ liệu và huấn luyện mô hình Random Forest...")

# 1. Tải dữ liệu mẫu từ Sklearn (đảm bảo máy nào cũng chạy được)
raw_data = fetch_california_housing(as_frame=True)
df = raw_data.frame

# 2. Xử lý dữ liệu (Preprocessing Pipeline)
# Xóa nhiễu (Capping)
df = df[df['MedHouseVal'] < 5.0].copy() # 5.0 tương đương 500k trong dataset gốc

# Tạo đặc trưng (Feature Engineering)
# Lưu ý: Dataset sklearn tên cột hơi khác 1 chút so với file csv, ta map lại cho chuẩn
df['rooms_per_household'] = df['AveRooms']
df['bedrooms_per_room'] = df['AveBedrms'] / df['AveRooms']
df['population_per_household'] = df['Population'] / df['AveOccup']

# Vì dataset sklearn không có cột Ocean Proximity (nó dùng Lat/Lon thuần túy)
# Ta sẽ giả lập cột này dựa trên tọa độ để Demo hoạt động trơn tru
# (Logic: Gần biển thì Lat/Lon nhất định)
def simulate_ocean(lat, lon):
    if lat < 34.5 and lon < -118.5: return "NEAR OCEAN"
    if lat > 37.5 and lon < -122.0: return "NEAR BAY"
    if lon > -119.0: return "INLAND"
    return "<1H OCEAN"

df['ocean_proximity'] = df.apply(lambda x: simulate_ocean(x['Latitude'], x['Longitude']), axis=1)

# One-Hot Encoding
df = pd.get_dummies(df, columns=['ocean_proximity'], dtype=int)

# Chuẩn bị X, y
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"] * 100000 # Chuyển về đơn vị USD

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Model
rf_reg = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf_reg.fit(X_scaled, y)

print("✅ Huấn luyện xong! Đang khởi động giao diện...")
print("-" * 30)

# --- PHẦN 2: HÀM DỰ ĐOÁN CHO DEMO ---
def advanced_prediction(median_income, housing_median_age, total_rooms, total_bedrooms,
                        population, households, latitude, longitude, ocean_proximity):
    
    # Tạo input dataframe
    # Map tên input vào đúng tên cột của Sklearn dataset
    input_data = {
        'MedInc': [median_income],
        'HouseAge': [housing_median_age],
        'AveRooms': [total_rooms / (households + 0.001)],       # Tính ngược lại Ave
        'AveBedrms': [total_bedrooms / (households + 0.001)],   # Tính ngược lại Ave
        'Population': [population],
        'AveOccup': [population / (households + 0.001)],        # Tính ngược lại Ave
        'Latitude': [latitude],
        'Longitude': [longitude]
    }
    
    input_df = pd.DataFrame(input_data)
    
    # Feature Engineering trên input
    input_df['rooms_per_household'] = input_df['AveRooms']
    input_df['bedrooms_per_room'] = input_df['AveBedrms'] / input_df['AveRooms']
    input_df['population_per_household'] = input_df['Population'] / input_df['AveOccup']
    
    # One-Hot Encoding thủ công cho input
    # Đảm bảo có đủ cột như lúc train
    expected_cols = [col for col in X.columns if 'ocean_proximity' in col]
    for col in expected_cols:
        input_df[col] = 0
    
    # Bật cột được chọn
    target_col = f"ocean_proximity_{ocean_proximity}"
    if target_col in input_df.columns:
        input_df[target_col] = 1

    # Reorder columns cho đúng thứ tự X
    input_df = input_df[X.columns]
    
    # Scale và Predict
    input_scaled = scaler.transform(input_df)
    predicted_price = rf_reg.predict(input_scaled)[0]
    
    # --- VISUALIZATION ---
    # Đồng hồ đo (Gauge)
    gauge_fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = predicted_price,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Định giá ($)"},
        gauge = {
            'axis': {'range': [10000, 500000], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 150000], 'color': "#90EE90"},
                {'range': [150000, 350000], 'color': "#FFD700"},
                {'range': [350000, 500000], 'color': "#FF6347"}],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': predicted_price}
        }
    ))
    gauge_fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))

    # Bản đồ (Map)
    map_df = pd.DataFrame({'lat': [latitude], 'lon': [longitude], 'text': [ocean_proximity]})
    map_fig = px.scatter_mapbox(map_df, lat="lat", lon="lon", hover_name="text", zoom=5, height=300)
    map_fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
    map_fig.update_traces(marker=dict(size=15, color='red'))

    return f"${predicted_price:,.2f}", gauge_fig, map_fig

# --- PHẦN 3: GIAO DIỆN GRADIO ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏠 AI Real Estate Valuator - California (Random Forest Demo)")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Thông số Vị trí")
            with gr.Row():
                lat_input = gr.Slider(32.5, 42, value=37.8, label="Vĩ độ (Latitude)")
                lon_input = gr.Slider(-124.5, -114, value=-122.2, label="Kinh độ (Longitude)")
            ocean_input = gr.Dropdown(["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"], value="NEAR BAY", label="Vị trí")
            
            gr.Markdown("### 2. Thông số Nhà & Dân cư")
            income_input = gr.Slider(0.5, 15, value=8.3, label="Thu nhập khu vực (x$10k)")
            age_input = gr.Slider(1, 52, value=41, step=1, label="Tuổi nhà")
            rooms_input = gr.Slider(100, 10000, value=880, label="Tổng số phòng")
            bedrooms_input = gr.Slider(10, 5000, value=129, label="Tổng phòng ngủ")
            pop_input = gr.Slider(100, 5000, value=322, label="Dân số")
            house_input = gr.Slider(10, 2000, value=126, label="Số hộ")
            
            btn = gr.Button("🔮 ĐỊNH GIÁ NGAY", variant="primary")

        with gr.Column():
            price_output = gr.Textbox(label="Giá trị ước tính", text_align="center")
            gauge_output = gr.Plot(label="Thước đo")
            map_output = gr.Plot(label="Vị trí")

    btn.click(advanced_prediction, 
              inputs=[income_input, age_input, rooms_input, bedrooms_input, pop_input, house_input, lat_input, lon_input, ocean_input],
              outputs=[price_output, gauge_output, map_output])

    gr.Examples([
        [8.3, 41, 880, 129, 322, 126, 37.88, -122.23, "NEAR BAY"],
        [2.5, 20, 2000, 500, 1500, 400, 36.5, -119.5, "INLAND"]
    ], inputs=[income_input, age_input, rooms_input, bedrooms_input, pop_input, house_input, lat_input, lon_input, ocean_input])

if __name__ == "__main__":
    demo.launch()
