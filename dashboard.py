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
    # TAB 1: OVERVIEW
    with tab1:
        # Key Metrics Row
        st.markdown('<p class="sub-header">Key Inflation Metrics</p>', unsafe_allow_html=True)
        
        if not filtered_df.empty:
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            # Latest CPI Value
            with metric_col1:
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{:.2f}</div>
                        <div class="metric-label">Latest Average CPI</div>
                    </div>
                """.format(current_avg_cpi), unsafe_allow_html=True)
            
            # YoY Change
            with metric_col2:
                if yoy_change is not None:
                    color = "green" if yoy_change < 0 else "red"
                    arrow = "↓" if yoy_change < 0 else "↑"
                    st.markdown("""
                        <div class="metric-card">
                            <div class="metric-value" style="color: {};">{}{:.2f}%</div>
                            <div class="metric-label">Year-over-Year Change</div>
                        </div>
                    """.format(color, arrow, abs(yoy_change)), unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="metric-card">
                            <div class="metric-value">N/A</div>
                            <div class="metric-label">Year-over-Year Change</div>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Category with highest CPI
            with metric_col3:
                if not latest_data.empty:
                    highest_item = latest_data.loc[latest_data['CPI_Value'].idxmax()]
                    st.markdown("""
                        <div class="metric-card">
                            <div class="metric-value">{}</div>
                            <div class="metric-label">Highest CPI Category</div>
                        </div>
                    """.format(highest_item['Indicator']), unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="metric-card">
                            <div class="metric-value">N/A</div>
                            <div class="metric-label">Highest CPI Category</div>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Category with highest inflation
            with metric_col4:
                if not latest_data.empty and not previous_year_data.empty:
                    # Calculate inflation by category
                    inflation_data = []
                    
                    for item in latest_data['Indicator'].unique():
                        latest_item_value = latest_data[latest_data['Indicator'] == item]['CPI_Value'].values[0]
                        
                        prev_item_data = previous_year_data[previous_year_data['Indicator'] == item]
                        if not prev_item_data.empty:
                            prev_item_value = prev_item_data['CPI_Value'].values[0]
                            item_inflation = ((latest_item_value - prev_item_value) / prev_item_value) * 100
                            inflation_data.append((item, item_inflation))
                    
                    if inflation_data:
                        highest_inflation_item = max(inflation_data, key=lambda x: x[1])
                        st.markdown("""
                            <div class="metric-card">
                                <div class="metric-value">{} ({:.2f}%)</div>
                                <div class="metric-label">Highest Inflation Category</div>
                            </div>
                        """.format(highest_inflation_item[0], highest_inflation_item[1]), unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class="metric-card">
                                <div class="metric-value">N/A</div>
                                <div class="metric-label">Highest Inflation Category</div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="metric-card">
                            <div class="metric-value">N/A</div>
                            <div class="metric-label">Highest Inflation Category</div>
                        </div>
                    """, unsafe_allow_html=True)
        
        # Main Trend Chart - REPLACED WITH STACKED AREA CHART
        st.markdown('<p class="sub-header">CPI Trends Over Time</p>', unsafe_allow_html=True)

        if not filtered_df.empty:
            # Create a pivot table with time series data for area chart
            # Group by date and indicator
            pivot_data = filtered_df.pivot_table(
                index='Date',
                columns='Indicator',
                values='CPI_Value',
                aggfunc='mean'
            ).reset_index()
            
            # Create the stacked area chart
            fig1 = go.Figure()
            
            # Get a list of all indicators and assign colors
            all_indicators = filtered_df['Indicator'].unique()
            all_colors = px.colors.qualitative.Bold[:len(all_indicators)]
            
            # Add traces for each indicator
            for i, indicator in enumerate(all_indicators):
                fig1.add_trace(
                    go.Scatter(
                        x=pivot_data['Date'],
                        y=pivot_data[indicator],
                        name=indicator,
                        mode='lines',
                        line=dict(width=0.5, color=all_colors[i % len(all_colors)]),
                        stackgroup='one',  # This creates the stacked area effect
                        hovertemplate='%{x}<br>%{y:.2f}',
                    )
                )
            
            # Update layout
            fig1.update_layout(
                height=500,
                title_text="Stacked Area Chart of CPI Trends Over Time",
                xaxis_title="Date",
                yaxis_title="CPI Value",
                legend_title="Categories",
                hovermode="x unified",
                template="plotly_white",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Display the figure
            st.plotly_chart(fig1, use_container_width=True)
            # Add a line chart option for comparison
            st.markdown('<p class="sub-header">CPI Line Chart (Alternative View)</p>', unsafe_allow_html=True)
            
            # Create a line chart for comparison
            fig1_line = go.Figure()
            
            # Add traces for each indicator
            for i, indicator in enumerate(all_indicators):
                fig1_line.add_trace(
                    go.Scatter(
                        x=pivot_data['Date'],
                        y=pivot_data[indicator],
                        name=indicator,
                        mode='lines',
                        line=dict(width=2, color=all_colors[i % len(all_colors)]),
                        hovertemplate='%{x}<br>%{y:.2f}',
                    )
                )
            
            # Update layout
            fig1_line.update_layout(
                height=400,
                title_text="Line Chart of CPI Trends Over Time",
                xaxis_title="Date",
                yaxis_title="CPI Value",
                legend_title="Categories",
                hovermode="x unified",
                template="plotly_white",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Display the alternative figure in an expander
            with st.expander("View Line Chart"):
                st.plotly_chart(fig1_line, use_container_width=True)

                # Display the alternative figure in an expander
            with st.expander("View Line Chart"):
                st.plotly_chart(fig1_line, use_container_width=True)
            
            # Add insights about the trends
            if not filtered_df.empty:
                # Calculate overall trend
                latest_year_data = filtered_df[filtered_df['Year'] == max_year]
                earliest_year_data = filtered_df[filtered_df['Year'] == min_year]
                
                if not latest_year_data.empty and not earliest_year_data.empty:
                    avg_latest = latest_year_data['CPI_Value'].mean()
                    avg_earliest = earliest_year_data['CPI_Value'].mean()
                    
                    overall_change = ((avg_latest - avg_earliest) / avg_earliest) * 100
                    
                    trend_description = "increasing" if overall_change > 0 else "decreasing"
                    
                    st.markdown(f"""
                        <div class="insight-box">
                            <strong>Key Insight:</strong> The CPI values show an overall {trend_description} trend of {abs(overall_change):.2f}% 
                            from {min_year} to {max_year}. This indicates that the cost of living in Sri Lanka has 
                            {"increased" if overall_change > 0 else "decreased"} over this period.
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No data available with the selected filters.")

        # Category Comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<p class="sub-header">Category Comparison</p>', unsafe_allow_html=True)
            
            if not filtered_df.empty:
                # Get average CPI by category for the latest year
                latest_year = filtered_df['Year'].max()
                latest_year_data = filtered_df[filtered_df['Year'] == latest_year]
                
                category_avg = latest_year_data.groupby('Indicator')['CPI_Value'].mean().sort_values(ascending=False).reset_index()
                
                fig2 = px.bar(
                    category_avg,
                    x='Indicator',
                    y='CPI_Value',
                    title=f'Average CPI by Category ({latest_year})',
                    labels={'CPI_Value': 'Average CPI', 'Indicator': 'Category'},
                    template='plotly_white',
                    color='CPI_Value',
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                fig2.update_layout(
                    height=400,
                    coloraxis_showscale=False,
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning("No data available with the selected filters.")
        with col2:
            st.markdown('<p class="sub-header">Monthly Distribution</p>', unsafe_allow_html=True)
            
            if not filtered_df.empty:
                # Monthly distribution of CPI values
                monthly_data = filtered_df.copy()
                
                # Create box plot for monthly distribution
                fig3 = px.box(
                    monthly_data,
                    x='Month',
                    y='CPI_Value',
                    title='Monthly Distribution of CPI Values',
                    labels={'CPI_Value': 'CPI Value', 'Month': 'Month'},
                    template='plotly_white',
                    color='Month'
                )
                
                # Define month order
                months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                               'July', 'August', 'September', 'October', 'November', 'December']
                
                fig3.update_layout(
                    height=400,
                    xaxis={'categoryorder':'array', 'categoryarray': months_order},
                    showlegend=False
                )
                st.plotly_chart(fig3, use_container_width=True)
                
                # Identify months with highest and lowest values
                monthly_avg = monthly_data.groupby('Month')['CPI_Value'].mean()
                highest_month = monthly_avg.idxmax()
                lowest_month = monthly_avg.idxmin()
                
                st.markdown(f"""
                    <div class="insight-box">
                        <strong>Seasonal Patterns:</strong> {highest_month} typically shows the highest CPI values, 
                        while {lowest_month} shows the lowest. This suggests potential seasonal factors affecting prices in Sri Lanka.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No data available with the selected filters.")    
                    