import streamlit as st
import pandas as pd 
import random
from sklearn.preprocessing import StandardScaler 
import pickle 

# Title

col = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']

st.title('California Housing Price Prediction')

st.image('https://www.shutterstock.com/image-photo/young-couple-visiting-potential-new-260nw-2222205237.jpg')



st.header('Model of housing prices to predict median house values in California', divider = True)


#st.subheader(''' User Must Enter the given values for price prediction:  
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']''')


st.sidebar.title('Select House Features 🏠')

st.sidebar.image('https://www.shutterstock.com/image-photo/online-real-estate-house-property-600nw-1915758310.jpg')

#read data 
temp_df = pd.read_csv('california.csv')

random.seed(52)

all_values = []
for i in temp_df[col]:
    min_value,max_value = temp_df[i].agg(['min','max'])

    var = st.sidebar.slider(f'Select {i} value ',int(min_value),int(max_value),
                      random.randint(int(min_value),int(max_value)))
    
    all_values.append(var)

ss = StandardScaler()
ss.fit(temp_df[col])

final_value = ss.transform([all_values])

with open('House_Price_Pred_ridge_model.pkl1','rb') as f:
    chatgpt = pickle.load(f)


price = chatgpt.predict(final_value)[0]

import time 

st.write(pd.DataFrame(dict(zip(col,all_values)),index = [1]))

value = 0
progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Price!')
place = st.empty()
place.image('https://elementor.com/blog/wp-content/uploads/2021/12/TRIO.gif', width = 100)

if price>0:
    
    for i in range(100):
      time.sleep(0.05)    
      progress_bar.progress(i*1)
    
    body = f'Predicted Median House Values: ${round(price,2)} Thousand Dollars'
    placeholder.empty()
    #st.success(body)

    st.success(body)

else:
    body = 'Invalid House Features'
    st.warning(body)
