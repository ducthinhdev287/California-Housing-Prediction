import pandas as pd

def load_data(file_path):
    """
    Đọc dữ liệu từ file CSV.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Đã tải dữ liệu thành công: {df.shape}")
        return df
    except Exception as e:
        print(f"Lỗi tải dữ liệu: {e}")
        return None
