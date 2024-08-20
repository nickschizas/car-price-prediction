import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title='EDA', page_icon=':chart_with_upwards_trend:')

# Import data
@st.cache_resource
def load_data():
    DATA_PATH = r'./data/clean/data_clean_20240811.csv'
    loaded_data = pd.read_csv(DATA_PATH, sep=';')
    loaded_data.drop(['FullName', 'ProductionDate'], axis=1, inplace=True)
    return loaded_data

data = load_data()
# Shape of the data
shape = data.shape
st.sidebar.markdown(f'**Data shape:**<br>*{data.shape[0]:,.0f} records*<br>*{data.shape[1]:,.0f} attributes*', unsafe_allow_html=True)

# Top-n car brands with respect to ads count
top_n = st.sidebar.number_input('top-n', min_value=3, max_value=10, step=1)
ads_counts = data['Name'].value_counts().head(top_n).reset_index()
ads_counts.columns = ['Brand', 'Ads Counts']
st.sidebar.dataframe(ads_counts, use_container_width=True, hide_index=True)

@st.cache_data
def dist_plot(num_var):
    dist_hist = go.Figure()
    dist_hist.add_trace(go.Histogram(x=data[num_var], histnorm='probability', nbinsx=300 ,marker={'color' : 'green', 'opacity' : 0.6}, hovertemplate='bin: %{x}<br>probability: %{y:.1%}', name=''))
    dist_hist.update_layout(title_text=f'Distribution of {num_var}', title_font_size=20)
    st.plotly_chart(dist_hist, use_container_width=True)

@st.cache_data
def corr_plot(num_var):
    if num_var != 'Price':
        num_scatter = go.Figure()
        num_scatter.add_trace(go.Scatter(x=data[num_var], y=data['Price'], mode='markers', marker={'color' : 'green', 'opacity' : 0.6}, hovertemplate='x: %{x}<br>Price: %{y:,.1f}', name=''))
        num_scatter.update_layout(title_text=f'{num_var} - Price', title_font_size=20)
        st.plotly_chart(num_scatter, use_container_width=True)

@st.cache_data
def pie_plot(cat_var):
    pie_data = data[cat_var].value_counts(normalize=True)
    pie = go.Figure()
    pie.add_trace(go.Pie(labels=pie_data.index, values=pie_data.values, hovertemplate='%{label}<br>%{value:.1%}', name=''))
    pie.update_layout(showlegend=False)
    st.plotly_chart(pie, use_container_width=True)

@st.cache_data
def box_plot(cat_var):
    box = go.Figure()
    for category in data[cat_var].unique():
        box_data = data[data[cat_var]==category]
        box.add_trace(go.Box(y=box_data['Price'], name=category))
    box.update_layout(title_text=f'Price per Category of {cat_var}', title_font_size=20, showlegend=False)
    st.plotly_chart(box, use_container_width=True)

COL_1, COL_2 = st.columns(2)
with COL_1:

    # Select Numeric Variable
    num_vars = data.select_dtypes(include=['float', 'int']).columns
    num_var = st.selectbox('Select Numeric Variable', options=num_vars)

    # Numeric Variables Distribution
    dist_plot(num_var)
    
    # Numeric Variables Correlations with Price
    corr_plot(num_var)
    
with COL_2:
    # Select Categorical Variable
    cat_vars = list(data.select_dtypes(exclude=['float', 'int']).columns)
    cat_vars.remove('Name')
    cat_var = st.selectbox('Select Categorical Variable', options=cat_vars)

    # Categorical Variables Count
    pie_plot(cat_var)

    # Categorical Variable Boxplots
    box_plot(cat_var)
