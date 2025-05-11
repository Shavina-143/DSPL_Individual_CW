import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy.stats as stats

# Set page configuration
st.set_page_config(
    page_title="Sri Lanka CPI Dashboard",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #1E3D59;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #1E3D59;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .card {
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
        background-color: #f9f9f9;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        padding: 15px;
        text-align: center;
        margin: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1E3D59;
    }
    .metric-label {
        font-size: 14px;
        color: #777;
    }
    .insight-box {
        background-color: #f0f7ff;
        border-left: 4px solid #1E3D59;
        padding: 10px 15px;
        margin: 15px 0;
        border-radius: 0 5px 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)