import streamlit as st
import pandas as pd
from pickle import load

st.set_page_config(layout="centered", page_title='Car Price Prediction', page_icon=':rocket:')

about_text = """
This is a ML web app designed for predicting car prices.
Choose your desired features and receive your prediction.
Both the training and testing data were web-scraped.
Navigate to the second page for an EDA on these data.
The model in use as well the data are frequently updated to ensure the best possible accuracy.
"""
st.sidebar.markdown(f'<em>{about_text}<em>', unsafe_allow_html=True)

st.sidebar.markdown(' ')
r2 = 0.90561
mae = 2412.50
st.session_state['model_test_scores'] = f'**Model in Use:**<br>*Untuned XGBRegressor*<br>**Test Scores:**<br>*R-Squared: {r2:.3%}<br>Mean Absolute Error: {mae:,.2f}*'
st.sidebar.markdown(st.session_state['model_test_scores'], unsafe_allow_html=True)

@st.cache_data
def load_model(path):
    model_pkl = open(path, 'rb')
    model = load(model_pkl)
    model_pkl.close()
    return model

@st.cache_data
def get_categories(path):
    data = pd.read_csv(path, sep=';', header=0)
    data.drop(['FullName', 'ProductionDate'], axis=1, inplace=True)

    brands = data['Name'].unique()
    gas_types = data['GasType'].unique()
    gearbox_types = data['GearBox'].unique()
    return brands, gas_types, gearbox_types

@st.cache_data
def transform_input(input):
    input_new = input.copy()
    for key in input.keys():
        if key in ['Name', 'GasType', 'GearBox']:
            input_new.update({f'{key}_{input[key]}' : True})
            input_new.pop(key)
    input_columns = ['Klm', 'CubicCapacity', 'Horsepower', 'Age', 'Name_Alfa-Romeo','Name_Audi', 'Name_Bmw', 'Name_Chevrolet', 'Name_Citroen', 'Name_DS',
       'Name_Dacia', 'Name_Daewoo', 'Name_Daihatsu', 'Name_Fiat', 'Name_Ford','Name_Honda', 'Name_Hyundai', 'Name_Isuzu', 'Name_Jaguar', 'Name_Jeep',
       'Name_Kia', 'Name_Lancia', 'Name_Land-Rover', 'Name_Lexus','Name_Mazda', 'Name_Mercedes-Benz', 'Name_Mini-Cooper','Name_Mitsubishi', 'Name_Nissan', 'Name_Opel', 'Name_Peugeot',
       'Name_Porsche', 'Name_Renault', 'Name_Saab', 'Name_Seat', 'Name_Skoda', 'Name_Smart', 'Name_Subaru', 'Name_Suzuki', 'Name_Toyota', 'Name_Volkswagen', 'Name_Volvo', 'GasType_Αέριο(lpg) - βενζίνη',
       'GasType_Βενζίνη', 'GasType_Πετρέλαιο','GasType_Υβριδικό plug-in βενζίνη', 'GasType_Υβριδικό plug-in πετρέλαιο', 'GasType_Υβριδικό βενζίνη',
       'GasType_Υβριδικό πετρέλαιο', 'GasType_Φυσικό αέριο(cng) - βενζίνη', 'GearBox_Manual', 'GearBox_Αυτόματο', 'GearBox_Ημιαυτόματο']
    input_df = pd.DataFrame(columns=input_columns)
    input_data_df = pd.DataFrame(input_new, index=[0])
    return pd.concat([input_df,input_data_df]).fillna(False)

# load the selected model
model = load_model(r'./models/xgb_reg_20240820_2123.pkl')
st.session_state['model'] = model

# extract categorical features from data
brands, gas_types, gearbox_types = get_categories(r'./data/clean/data_clean_20240811.csv')

with st.form(key='input'):
    name = st.selectbox('Car Brand', options=brands)
    gearbox = st.selectbox('Gear Box Type', options=gearbox_types)
    age = st.number_input('Age', min_value=0, value=8)
    cc = st.number_input('CubicCapacity', min_value=0, value=1300)
    gastype = st.selectbox('GasType', options=gas_types)
    klm = st.number_input('Kilometers', min_value=0, value=80000)
    hp = st.number_input('Horsepower', min_value=0, value=100)

    input_ = {'Name':name, 'GearBox':gearbox, 'Age':age, 'CubicCapacity':cc, 'GasType':gastype, 'Klm':klm, 'Horsepower':hp}

    submitted = st.form_submit_button('Predict')

    if submitted:
        st.session_state['pred'] = model.predict(transform_input(input_))[0]
    
    if 'pred' in st.session_state:
       pred = st.session_state['pred'].copy()
       st.subheader(f'Prediction : {pred:,.0f}€', anchor=False)
