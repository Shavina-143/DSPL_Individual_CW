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
