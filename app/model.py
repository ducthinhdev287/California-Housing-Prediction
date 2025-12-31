from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

def train_random_forest(X_train, y_train):
    print("Đang huấn luyện Random Forest...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_linear_regression(X_train, y_train):
    print("Đang huấn luyện Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_svr(X_train, y_train):
    print("Đang huấn luyện SVR...")
    model = SVR(kernel='rbf')
    model.fit(X_train, y_train)
    return model
