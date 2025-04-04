# Simulate data 
 
import pandas as pd 
import numpy as np 
 
# Simulate 200 rows of data 
np.random.seed(42) 
 
# Possible categories 
product_categories = ['Electronics', 'Clothing', 'Books'] 
shipping_methods = ['Standard', 'Express', 'Same-day'] 
customer_locations = ['Navsari', 'Surat', 'Gurugram'] 
 
data = { 
    'product_category': np.random.choice(product_categories, 200), 
    'shipping_method': np.random.choice(shipping_methods, 200), 
    'customer_location': np.random.choice(customer_locations, 200), 
    # For demonstration, create random numeric features 
    'order_volume': np.random.randint(1, 100, 200), 
    'distance': np.random.randint(1, 500, 200), 
    # Delivery time in days, with some random variation 
    'delivery_time': np.random.randint(1, 10, 200) 
} 
 
df = pd.DataFrame(data) 
df.head() 
 
#Categorical Encoding 
 
df_encoded = pd.get_dummies(df, columns=['product_category', 'shipping_method', 
'customer_location']) 
df_encoded.head() 
 
#Define Features (X) and Target (y) 
 
X = df_encoded.drop('delivery_time', axis=1) 
y = df_encoded['delivery_time'] 
 
#Split the Data into training and testing sets 
 
from sklearn.model_selection import train_test_split 
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error, mean_squared_error 
 
model = RandomForestRegressor(n_estimators=100, random_state=42) 
model.fit(X_train, y_train) 
 
# Evaluate on test set 
y_pred = model.predict(X_test) 
mae = mean_absolute_error(y_test, y_pred) 
mse = mean_squared_error(y_test, y_pred) 
rmse = mse ** 0.5 
 
print("MAE:", mae) 
print("RMSE:", rmse) 
 
import joblib 
 
joblib.dump(model, 'delivery_time_model.pkl') 
print("Model saved to delivery_time_model.pkl") 