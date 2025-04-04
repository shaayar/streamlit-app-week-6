import streamlit as st 
import joblib 
import numpy as np 
 
# Load the trained model 
model = joblib.load('delivery_time_model.pkl') 
 
# Title 
st.title("Timelytics: Order-to-Delivery Time Prediction") 
 
# Instructions 
st.write("Enter the order details below to predict the expected delivery time.") 
 
# Example user inputs 
product_category = st.selectbox("Product Category", ["Electronics", "Clothing", "Books"]) 
shipping_method = st.selectbox("Shipping Method", ["Standard", "Express", "Same-day"]) 
customer_location = st.selectbox("Customer Location", ["CityA", "CityB", "CityC"]) 
order_volume = st.number_input("Order Volume", min_value=1, max_value=100, value=10) 
distance = st.number_input("Distance (km)", min_value=1, max_value=1000, value=100) 
 
# For demonstration, assume the model was trained with get_dummies for these columns: 
# product_category_Electronics, product_category_Clothing, product_category_Books 
# shipping_method_Standard, shipping_method_Express, shipping_method_Same-day 
# customer_location_CityA, customer_location_CityB, customer_location_CityC 
# order_volume, distance 
 
def encode_inputs(product_category, shipping_method, customer_location, order_volume, 
distance): 
    # Initialize all zeroes for the dummy variables 
    input_dict = { 
        'product_category_Electronics': 0, 
        'product_category_Clothing': 0, 
        'product_category_Books': 0, 
        'shipping_method_Standard': 0, 
        'shipping_method_Express': 0, 
        'shipping_method_Same-day': 0, 
        'customer_location_CityA': 0, 
        'customer_location_CityB': 0, 
        'customer_location_CityC': 0, 
        'order_volume': order_volume, 
        'distance': distance 
    } 
 
    # Set the appropriate dummy variable to 1 
    input_dict[f'product_category_{product_category}'] = 1 
    input_dict[f'shipping_method_{shipping_method}'] = 1 
    input_dict[f'customer_location_{customer_location}'] = 1 
 
    # Convert to list or NumPy array in the correct column order 
    # The order must match how your training data was fed to the model 
    input_list = [ 
        input_dict['order_volume'], 
        input_dict['distance'], 
        input_dict['product_category_Clothing'], 
        input_dict['product_category_Electronics'], 
        input_dict['product_category_Books'], 
        input_dict['shipping_method_Express'], 
        input_dict['shipping_method_Same-day'], 
        input_dict['shipping_method_Standard'], 
        input_dict['customer_location_CityA'], 
        input_dict['customer_location_CityB'], 
        input_dict['customer_location_CityC'] 
    ] 
 
    return np.array(input_list).reshape(1, -1) 
 
if st.button("Predict Delivery Time"): 
    # Encode the user inputs 
    input_features = encode_inputs(product_category, shipping_method, customer_location, 
order_volume, distance) 
     
    # Make prediction 
    prediction = model.predict(input_features)[0] 
 
    # Display the result 
    st.success(f"Estimated Delivery Time: {round(prediction, 2)} days")