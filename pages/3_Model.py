import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", page_title='Model', page_icon=':robot_face:')

st.sidebar.markdown(st.session_state['model_test_scores'], unsafe_allow_html=True)

model = st.session_state['model']

st.markdown('*Work in progress...*')
st.markdown('[SHAP](https://shap.readthedocs.io/en/latest/index.html)')





# st.dataframe(pd.DataFrame({'FeaturName' : model.feature_names_in_, 'FeatureImportance' : model.feature_importances_}), use_container_width=True)