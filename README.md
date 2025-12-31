CALIFORNIA HOUSING PRICE PREDICTION

A Comparative Analysis of Regression Models

Học phần: Học máy cơ bản
Học kỳ/Năm học: HK5 2025-2026
Sinh viên thực hiện: Đỗ Đức Thịnh - 10123307

1. GIỚI THIỆU (INTRODUCTION

1.1. Bối cảnh bài toán

Thị trường bất động sản tại bang California có mức độ phức tạp cao, với giá nhà chịu ảnh hưởng mạnh từ vị trí địa lý, thu nhập dân cư và các yếu tố nhân khẩu học. Do đó, việc xây dựng một mô hình hồi quy có khả năng dự đoán giá trị trung vị nhà ở (median_house_value) tại từng khu vực là một bài toán điển hình và có ý nghĩa thực tiễn trong lĩnh vực Khoa học Dữ liệu.

Dự án này sử dụng dữ liệu từ Cuộc điều tra dân số Hoa Kỳ năm 1990 (U.S. Census 1990) nhằm huấn luyện và so sánh hiệu năng của nhiều mô hình học máy khác nhau.

1.2. Mục tiêu nghiên cứu

Xây dựng mô hình: Triển khai và huấn luyện các mô hình hồi quy từ cơ bản đến nâng cao.

Tiền xử lý và kỹ thuật đặc trưng: Khắc phục các vấn đề phi tuyến, đa cộng tuyến và chênh lệch quy mô dữ liệu.

Đánh giá và so sánh: Phân tích hiệu năng mô hình thông qua các chỉ số thống kê chuẩn trong bài toán regression (MAE, RMSE, R²).

2. MÔ TẢ DỮ LIỆU (DATASET DESCRIPTION)

Dữ liệu sử dụng là California Housing Dataset, bao gồm 20,640 mẫu quan sát, mỗi mẫu đại diện cho một khu vực dân cư.

Nguồn dữ liệu: 1990 U.S. Census

Biến mục tiêu (Target): median_house_value

Từ điển dữ liệu (Data Dictionary)
Feature	Mô tả	Ghi chú
longitude	Kinh độ địa lý	Đơn vị độ
latitude	Vĩ độ địa lý	Đơn vị độ
housing_median_age	Tuổi đời trung vị của nhà ở	Giới hạn tối đa 52
total_rooms	Tổng số phòng	Dữ liệu thô
total_bedrooms	Tổng số phòng ngủ	Có giá trị khuyết
population	Dân số khu vực	Người
households	Số hộ gia đình	Hộ
median_income	Thu nhập trung vị	1.0 = 10,000 USD
ocean_proximity	Vị trí so với biển	Biến phân loại
median_house_value	Giá nhà trung vị	Giới hạn 500,000 USD
3. PHƯƠNG PHÁP NGHIÊN CỨU (METHODOLOGY)

Dự án tuân theo quy trình tiêu chuẩn trong một bài toán Machine Learning ứng dụng.

3.1. Phân tích dữ liệu khám phá (EDA)

Phân tích phân phối biến mục tiêu và phát hiện hiện tượng capping tại mức 500,000 USD.

Phân tích ma trận tương quan cho thấy median_income là đặc trưng có ảnh hưởng mạnh nhất.

Trực quan hóa dữ liệu theo tọa độ địa lý để phát hiện các mối quan hệ phi tuyến.

3.2. Tiền xử lý và kỹ thuật đặc trưng

Làm sạch dữ liệu:

Loại bỏ các điểm nhiễu do giới hạn trần giá nhà.

Xử lý giá trị khuyết bằng phương pháp Median Imputation.

Feature Engineering:

rooms_per_household

bedrooms_per_room

population_per_household

Mã hóa dữ liệu: One-Hot Encoding cho ocean_proximity.

Chuẩn hóa dữ liệu: Sử dụng StandardScaler, đặc biệt cần thiết cho mô hình SVR.

3.3. Lựa chọn mô hình

Ba mô hình đại diện cho các hướng tiếp cận khác nhau được triển khai:

Linear Regression – Mô hình cơ sở, giả định quan hệ tuyến tính.

Support Vector Regression (SVR) – Mô hình phi tuyến dựa trên kernel.

Random Forest Regressor – Mô hình ensemble mạnh, xử lý tốt quan hệ phức tạp.

4. KẾT QUẢ THỰC NGHIỆM (EXPERIMENTAL RESULTS)

Đánh giá trên tập kiểm thử (20% dữ liệu):

Mô hình	MAE	R² Score	Nhận xét
Random Forest	31,027	0.7743	Hiệu năng tốt nhất
Linear Regression	44,825	0.6181	Underfitting
SVR (RBF)	76,485	-0.0252	Không hội tụ
Nhận xét

Random Forest cho kết quả vượt trội nhờ khả năng mô hình hóa các tương tác phi tuyến giữa vị trí và thu nhập.

Linear Regression phù hợp làm baseline nhưng không đủ linh hoạt.

SVR hoạt động kém do độ chênh lớn của biến mục tiêu và yêu cầu chuẩn hóa nghiêm ngặt.

5. HƯỚNG DẪN CÀI ĐẶT (INSTALLATION)
Yêu cầu

Python ≥ 3.8

Cài đặt
git clone https://github.com/[Your_Username]/California-Housing-Prediction.git
cd California-Housing-Prediction
pip install -r requirements.txt

Chạy demo (Gradio Web App)
python demo/app_gradio.py


Truy cập: http://127.0.0.1:7860

6. CẤU TRÚC DỰ ÁN (PROJECT STRUCTURE)
California-Housing-Prediction/
│
├── app/
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── model.py
│
├── data/
│   ├── sample_housing.csv
│   └── housing.csv
│
├── demo/
│   ├── demo_notebook.ipynb
│   └── app_gradio.py
│
├── reports/
│   └── Final_Report.pdf
│
├── slides/
│   └── Presentation.pptx
│
├── requirements.txt
└── README.md

7. THÔNG TIN TÁC GIẢ (AUTHOR)

Họ và tên: Đỗ Đức Thịnh

Mã sinh viên: 10123307

Lớp: 124231

Email: dthinh287.dev@gmail.com
