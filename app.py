#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
mart_df = pd.read_csv(r"C:\Users\Patricia\Internship\merged_data.csv")

columns_to_drop = ['Order ID', 'Order Date', 'Ship Date', 'Customer Name', 'Sales']

X = mart_df.drop(columns=columns_to_drop)
y = mart_df['Sales']

categorical_columns = ['City','Country','Region','Segment','Ship Mode','State','Product Name','Category','Sub-Category']

# Apply one-hot encoding to the categorical columns
preprocessor = ColumnTransformer(
transformers=[('encoder', OneHotEncoder(), categorical_columns)],
remainder='passthrough'
)

X_encoded = preprocessor.fit_transform(X)

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.25, random_state=7)

# Initialize the XGBoost Regressor
xgb = XGBRegressor()

# Train the model
xgb.fit(X_train, y_train)

import streamlit as st

# Create the Streamlit app
st.title('Amazing mart - Sales Prediction')

# User input for feature values
user_input = {}
for column in X.columns:
    if column in categorical_columns:
        unique_values = mart_df[column].unique()
        user_input[column] = st.selectbox(column, unique_values)
    else:
        user_input[column] = st.number_input(column, value=0)

# Transform user input to one-hot encoding
user_input_encoded = preprocessor.transform(pd.DataFrame(user_input, index=[0]))

# Make predictions using the trained model
prediction = xgb.predict(user_input_encoded)

# Display the prediction
st.subheader('Prediction')
st.write(f'The predicted Sales is: {prediction[0]}')


# In[ ]:




