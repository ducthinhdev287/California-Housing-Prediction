import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    """Bước 1: Làm sạch và xử lý dữ liệu thô"""
    # 1. Xóa dữ liệu nhiễu (Capping tại 500k)
    df_clean = df[df['median_house_value'] < 500000].copy()
    
    # 2. Điền dữ liệu thiếu
    median_bedrooms = df_clean['total_bedrooms'].median()
    df_clean['total_bedrooms'] = df_clean['total_bedrooms'].fillna(median_bedrooms)
    
    return df_clean

def feature_engineering(df):
    """Bước 2: Tạo đặc trưng mới"""
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']
    return df

def preprocess_pipeline(df):
    """Chạy toàn bộ quy trình xử lý và chia tập Train/Test"""
    # 1. Clean
    df = clean_data(df)
    
    # 2. Engineer
    df = feature_engineering(df)
    
    # 3. One-Hot Encoding
    df = pd.get_dummies(df, columns=['ocean_proximity'], dtype=int)
    
    # 4. Split Data
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Trả về cả scaler để dùng cho demo sau này
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns
