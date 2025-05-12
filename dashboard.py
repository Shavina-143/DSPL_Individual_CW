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

# Dashboard title
st.markdown('<p class="main-header">Sri Lanka Consumer Price Index Analysis</p>', unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("cleaned_cpi_sri_lanka.csv")
        
        # Convert date columns to datetime
        df['Start_Date'] = pd.to_datetime(df['Start_Date'])
        df['End_Date'] = pd.to_datetime(df['End_Date'])
        
        # Create a date column for easier plotting (midpoint between start and end)
        df['Date'] = df['Start_Date'] + (df['End_Date'] - df['Start_Date']) / 2
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is not None:
    # Create tabs for different dashboard views
    tab1, tab2, tab3 = st.tabs(["Overview", "Detailed Analysis", "Economic Indicators"])
    
    # Sidebar for filters (applied to all tabs)
    st.sidebar.markdown("## Filters")
    
    # Year range filter
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())
    selected_years = st.sidebar.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    
    # Item filter
    item_list = sorted(df['Indicator'].unique())
    selected_items = st.sidebar.multiselect(
        "Select Items",
        options=item_list,
        default=item_list[:3] if len(item_list) > 3 else item_list
    )
    
    # Apply filters
    filtered_df = df[(df['Year'] >= selected_years[0]) & 
                     (df['Year'] <= selected_years[1]) & 
                     (df['Indicator'].isin(selected_items))]
    
        # Add comparative analyses
    if "Item_Code" in filtered_df.columns:
        # Create item category grouping if applicable
        item_categories = filtered_df.groupby('Item_Code')['Indicator'].first().to_dict()
        filtered_df['Category'] = filtered_df['Item_Code'].map(item_categories)
    
    # Calculate overall CPI metrics
    if not filtered_df.empty:
        latest_date = filtered_df['Date'].max()
        previous_year_date = latest_date - pd.DateOffset(years=1)
        
        latest_data = filtered_df[filtered_df['Date'] == latest_date]
        previous_year_data = filtered_df[filtered_df['Date'] == previous_year_date]
        
        # Calculate overall average CPI
        current_avg_cpi = latest_data['CPI_Value'].mean()
        prev_avg_cpi = previous_year_data['CPI_Value'].mean() if not previous_year_data.empty else None
        
        # Calculate YoY change
        if prev_avg_cpi:
            yoy_change = ((current_avg_cpi - prev_avg_cpi) / prev_avg_cpi) * 100
        else:
            yoy_change = None