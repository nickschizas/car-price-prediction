import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(layout="wide", page_title='Model', page_icon=':robot_face:')

st.sidebar.markdown(st.session_state['model_test_scores'], unsafe_allow_html=True)

model = st.session_state['model']

st.markdown('*Work in progress...*')
st.markdown('[SHAP](https://shap.readthedocs.io/en/latest/index.html)')
