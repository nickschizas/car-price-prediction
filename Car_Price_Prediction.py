import streamlit as st
import pandas as pd
import pickle as pkl

st.set_page_config(layout="centered", page_title='Car Price Prediction', page_icon=':rocket:')

about_text = """
This is a ML web app designed for predicting car prices.
You can choose your desired features to receive your prediction.
Both the training and testing data were web-scraped.
Navigate to the second page for an EDA on these data.
The model in use as well the data are frequently updated to ensure accuracy.
"""

contact_text = """
Feel free to fork this app or contact:<br>
n.schizas@outlook.com
"""
st.sidebar.markdown(f'<em>{about_text}<em>', unsafe_allow_html=True)
st.sidebar.caption(f'<em>{contact_text}<em>', unsafe_allow_html=True)

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

DATA_PATH = './data/clean/data_clean_20240509.csv'

@st.cache_resource
def load_data():
    data = pd.read_csv(DATA_PATH, sep=';')
    brands = data['Name'].unique()
    gear_box = data['GearBox'].unique()
    gas_types = data['GasType'].unique()
    return brands, gear_box, gas_types

@st.cache_resource
def load_model(model_name, path=r'./models/'):
    with open(path + model_name, 'rb') as f:
        model = pkl.load(f)
    return model

brands, gear_box, gas_types = load_data()

model = load_model('RandomForest_tuned.pkl')

with st.columns([1,2.6,1])[1]:
    st.title('Car Price Prediction', anchor=False)

with st.form(key='input'):
    form_col = st.columns(2)
    with form_col[0]:
       name = st.selectbox('Car Brand', options=brands)
       gearbox = st.selectbox('Gear Box Type', options=gear_box)
       age = st.number_input('Age', min_value=0, value=8)
       cc = st.number_input('CubicCapacity', min_value=0, value=1300)

    with form_col[1]:
       gastype = st.selectbox('GasType', options=gas_types)
       klm = st.number_input('Kilometers', min_value=0, value=80000)
       hp = st.number_input('Horsepower', min_value=0, value=100)

    
    predict = st.form_submit_button('Predict')
    if predict:
        input = {'Klm':klm, 'CubicCapacity':cc, 'Horsepower':hp, 'Age':age, 'Name':name, 'GasType':gastype, 'GearBox':gearbox}
        input_data = transform_input(input)
        st.session_state['pred'] = model.predict(input_data)[0]

if 'pred' in st.session_state:
       pred = st.session_state['pred'].copy()
       st.subheader(f'Prediction : {"{:,.0f}".format(pred)} €', anchor=False)
