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
        # TAB 2: DETAILED ANALYSIS
    with tab2:
        # Inflation Rate Analysis
        st.markdown('<p class="sub-header">Inflation Rate Analysis</p>', unsafe_allow_html=True)
        
        if not filtered_df.empty:
            # Calculate monthly inflation rates for each indicator
            # Group by date and indicator to get average CPI values
            monthly_avg = filtered_df.groupby(['Year', 'Month', 'Indicator'])['CPI_Value'].mean().reset_index()
            
            # Sort by year and month
            months_order = {month: i for i, month in enumerate(['January', 'February', 'March', 'April', 'May', 'June', 
                                                          'July', 'August', 'September', 'October', 'November', 'December'])}
            
            monthly_avg['Month_Num'] = monthly_avg['Month'].map(months_order)
            monthly_avg = monthly_avg.sort_values(['Year', 'Month_Num'])
            
            # Calculate inflation rate (% change from previous period)
            inflation_data = []
            
            for indicator in monthly_avg['Indicator'].unique():
                indicator_data = monthly_avg[monthly_avg['Indicator'] == indicator].copy()
                indicator_data['Inflation_Rate'] = indicator_data['CPI_Value'].pct_change() * 100
                inflation_data.append(indicator_data)
            
            inflation_df = pd.concat(inflation_data)
            inflation_df = inflation_df.dropna(subset=['Inflation_Rate'])
            
            # Create a date column for plotting
            inflation_df['Period'] = inflation_df['Year'].astype(str) + '-' + (inflation_df['Month_Num'] + 1).astype(str).str.zfill(2)
            
            # Plot inflation rates
            fig4 = px.line(
                inflation_df,
                x='Period',
                y='Inflation_Rate',
                color='Indicator',
                title='Monthly Inflation Rates by Category',
                labels={'Inflation_Rate': 'Monthly Inflation Rate (%)', 'Period': 'Time Period'},
                template='plotly_white',
                markers=True
            )
            
            fig4.update_layout(
                height=400,
                legend_title_text='Categories',
                hovermode='x unified',
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig4, use_container_width=True)
            
            # Add insights about inflation volatility
            inflation_volatility = inflation_df.groupby('Indicator')['Inflation_Rate'].std().sort_values(ascending=False)
            most_volatile = inflation_volatility.index[0]
            least_volatile = inflation_volatility.index[-1]
            
            st.markdown(f"""
                <div class="insight-box">
                    <strong>Inflation Volatility:</strong> {most_volatile} shows the highest inflation volatility 
                    with a standard deviation of {inflation_volatility.iloc[0]:.2f}%, while {least_volatile} is the most stable 
                    with a standard deviation of {inflation_volatility.iloc[-1]:.2f}%. High volatility indicates unpredictable price changes 
                    that can impact consumer budgeting and economic planning.
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No data available with the selected filters.")

                # Correlation Analysis
        st.markdown('<p class="sub-header">Category Correlation Analysis</p>', unsafe_allow_html=True)
        
        if not filtered_df.empty and len(selected_items) > 1:
            # Create pivot table for correlation analysis
            pivot_df = filtered_df.pivot_table(
                index=['Year', 'Month'], 
                columns='Indicator', 
                values='CPI_Value',
                aggfunc='mean'
            ).reset_index()
            
            # Calculate correlation matrix
            corr_matrix = pivot_df.drop(['Year', 'Month'], axis=1).corr()
            
            # Plot correlation heatmap
            fig5 = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale='RdBu_r',
                zmin=-1,
                zmax=1,
                title='Correlation Between Categories'
            )
            
            fig5.update_layout(
                height=500,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig5, use_container_width=True)
            
            # Find highest correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            
            # Sort by absolute correlation value
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            if corr_pairs:
                highest_corr = corr_pairs[0]
                lowest_corr = min(corr_pairs, key=lambda x: abs(x[2]))
                
                st.markdown(f"""
                    <div class="insight-box">
                        <strong>Category Relationships:</strong> The strongest relationship is between {highest_corr[0]} and {highest_corr[1]} 
                        with a correlation of {highest_corr[2]:.2f}, suggesting these categories tend to move together. 
                        In contrast, {lowest_corr[0]} and {lowest_corr[1]} show the weakest relationship 
                        with a correlation of {lowest_corr[2]:.2f}.
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Select at least two categories for correlation analysis.")
        # Seasonal Decomposition
        st.markdown('<p class="sub-header">Seasonal Decomposition Analysis</p>', unsafe_allow_html=True)
        
        if not filtered_df.empty and len(selected_items) > 0:
            # Select indicator for decomposition
            decomp_indicator = st.selectbox(
                "Select category for seasonal decomposition",
                options=selected_items
            )
            
            # Filter data for selected indicator
            decomp_data = filtered_df[filtered_df['Indicator'] == decomp_indicator].copy()
            
            # Create a time series
            decomp_data = decomp_data.sort_values('Date')
            decomp_data.set_index('Date', inplace=True)
            
            # Resample to monthly frequency if needed
            monthly_series = decomp_data['CPI_Value'].resample('M').mean()
            
            if len(monthly_series) >= 12:  # Need at least 12 months for seasonal decomposition
                try:
                    # Perform seasonal decomposition
                    result = seasonal_decompose(monthly_series, model='additive', period=12)
                    
                    # Create figure with subplots
                    fig6 = go.Figure()
                    
                    # Original series
                    fig6.add_trace(go.Scatter(
                        x=result.observed.index,
                        y=result.observed.values,
                        mode='lines',
                        name='Observed',
                        line=dict(color='blue')
                    ))
                    
                    # Trend component
                    fig6.add_trace(go.Scatter(
                        x=result.trend.index,
                        y=result.trend.values,
                        mode='lines',
                        name='Trend',
                        line=dict(color='red')
                    ))
                    
                    # Seasonal component
                    fig6.add_trace(go.Scatter(
                        x=result.seasonal.index,
                        y=result.seasonal.values,
                        mode='lines',
                        name='Seasonal',
                        line=dict(color='green')
                    ))
                    
                    # Residual component
                    fig6.add_trace(go.Scatter(
                        x=result.resid.index,
                        y=result.resid.values,
                        mode='lines',
                        name='Residual',
                        line=dict(color='purple')
                    ))
                    
                    fig6.update_layout(
                        height=500,
                        title=f'Seasonal Decomposition for {decomp_indicator}',
                        xaxis_title='Date',
                        yaxis_title='Value',
                        legend_title='Component',
                        hovermode='x unified',
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig6, use_container_width=True)
                    
                    # Extract seasonal patterns
                    seasonal_patterns = result.seasonal.groupby(result.seasonal.index.month).mean().reset_index()
                    seasonal_patterns.columns = ['Month', 'Seasonal Effect']
                    
                    # Map month numbers to names
                    month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                                7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
                    seasonal_patterns['Month_Name'] = seasonal_patterns['Month'].map(month_names)
                    
                    # Plot seasonal patterns
                    fig7 = px.bar(
                        seasonal_patterns,
                        x='Month_Name',
                        y='Seasonal Effect',
                        title=f'Seasonal Effects by Month for {decomp_indicator}',
                        labels={'Seasonal Effect': 'Effect on CPI', 'Month_Name': 'Month'},
                        template='plotly_white',
                        color='Seasonal Effect',
                        color_continuous_scale=px.colors.diverging.RdBu_r
                    )
                    
                    # Sort months in correct order
                    fig7.update_layout(
                        height=400,
                        xaxis={'categoryorder':'array', 'categoryarray': list(month_names.values())},
                        coloraxis_showscale=False
                    )
                    
                    st.plotly_chart(fig7, use_container_width=True)
                    
                    # Add insights about seasonality
                    max_effect_month = seasonal_patterns.loc[seasonal_patterns['Seasonal Effect'].idxmax(), 'Month_Name']
                    min_effect_month = seasonal_patterns.loc[seasonal_patterns['Seasonal Effect'].idxmin(), 'Month_Name']
                    
                    st.markdown(f"""
                        <div class="insight-box">
                            <strong>Seasonal Patterns:</strong> For {decomp_indicator}, prices tend to be higher in {max_effect_month} 
                            and lower in {min_effect_month}. This seasonal pattern might be related to supply and demand factors, 
                            agricultural cycles, or cultural events that affect consumption patterns in Sri Lanka.
                        </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error in seasonal decomposition: {e}")
            else:
                st.warning("Not enough data points for seasonal decomposition. Need at least 12 months of data.")
        else:
            st.warning("No data available with the selected filters.")
    # TAB 3: ECONOMIC INDICATORS
    with tab3:
        # CPI Distribution Analysis
st.markdown('<p class="sub-header">CPI Distribution Analysis</p>', unsafe_allow_html=True)

if not filtered_df.empty:
    # Create histograms for each time period
    period_col1, period_col2 = st.columns(2)
    
    with period_col1:
        # Get most recent year's data
        latest_year = filtered_df['Year'].max()
        latest_year_data = filtered_df[filtered_df['Year'] == latest_year]
        
        # Create histogram
        fig8 = px.histogram(
            latest_year_data,
            x='CPI_Value',
            color='Indicator',
            title=f'CPI Distribution for {latest_year}',
            labels={'CPI_Value': 'CPI Value', 'count': 'Frequency'},
            template='plotly_white',
            opacity=0.7,
            barmode='overlay'
        )
        
        fig8.update_layout(
            height=400,
            legend_title_text='Categories',
            bargap=0.1
        )
        
        st.plotly_chart(fig8, use_container_width=True)
    
    with period_col2:
        # Get earliest year's data for comparison
        earliest_year = filtered_df['Year'].min()
        earliest_year_data = filtered_df[filtered_df['Year'] == earliest_year]
        
        # Create histogram
        fig9 = px.histogram(
            earliest_year_data,
            x='CPI_Value',
            color='Indicator',
            title=f'CPI Distribution for {earliest_year}',
            labels={'CPI_Value': 'CPI Value', 'count': 'Frequency'},
            template='plotly_white',
            opacity=0.7,
            barmode='overlay'
        )
        
        fig9.update_layout(
            height=400,
            legend_title_text='Categories',
            bargap=0.1
        )
        
        st.plotly_chart(fig9, use_container_width=True)
    
    # Skewness and kurtosis analysis
    if not latest_year_data.empty and not earliest_year_data.empty:
        latest_skew = stats.skew(latest_year_data['CPI_Value'].dropna())
        earliest_skew = stats.skew(earliest_year_data['CPI_Value'].dropna())
        
        latest_kurt = stats.kurtosis(latest_year_data['CPI_Value'].dropna())
        earliest_kurt = stats.kurtosis(earliest_year_data['CPI_Value'].dropna())
        
        # Interpret skewness
        if latest_skew > 0.5:
            skew_interpretation = "positively skewed (many lower values with few high outliers)"
        elif latest_skew < -0.5:
            skew_interpretation = "negatively skewed (many higher values with few low outliers)"
        else:
            skew_interpretation = "approximately symmetric"
        
        st.markdown(f"""
            <div class="insight-box">
                <strong>Distribution Analysis:</strong> The current CPI distribution is {skew_interpretation}. 
                From {earliest_year} to {latest_year}, the distribution's skewness changed from {earliest_skew:.2f} to {latest_skew:.2f}, 
                indicating a shift in how prices are distributed across categories. This suggests 
                {"more price outliers" if abs(latest_skew) > abs(earliest_skew) else "more uniform pricing"} in recent periods.
            </div>
        """, unsafe_allow_html=True)
else:
    st.warning("No data available with the selected filters.")

    # Volatility Analysis
st.markdown('<p class="sub-header">Price Volatility Analysis</p>', unsafe_allow_html=True)

if not filtered_df.empty:
    # Calculate rolling standard deviation (volatility) for each indicator
    volatility_data = []
    
    for indicator in filtered_df['Indicator'].unique():
        indicator_data = filtered_df[filtered_df['Indicator'] == indicator].copy()
        indicator_data = indicator_data.sort_values('Date')
        
        # Calculate rolling 3-period standard deviation
        if len(indicator_data) >= 3:
            indicator_data['Volatility'] = indicator_data['CPI_Value'].rolling(window=3).std()
            volatility_data.append(indicator_data)
    
    volatility_df = pd.concat(volatility_data)
    volatility_df = volatility_df.dropna(subset=['Volatility'])
    
    if not volatility_df.empty:
        # Plot volatility over time
        fig10 = px.line(
            volatility_df,
            x='Date',
            y='Volatility',
            color='Indicator',
            title='Price Volatility Over Time (Rolling 3-Period Standard Deviation)',
            labels={'Volatility': 'Volatility (Std Dev)', 'Date': 'Date'},
            template='plotly_white'
        )
        
        fig10.update_layout(
            height=400,
            legend_title_text='Categories',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig10, use_container_width=True)
        
        # Calculate average volatility by indicator
        avg_volatility = volatility_df.groupby('Indicator')['Volatility'].mean().sort_values(ascending=False).reset_index()
        
        # Plot average volatility by category
        fig11 = px.bar(
            avg_volatility,
            x='Indicator',
            y='Volatility',
            title='Average Price Volatility by Category',
            labels={'Volatility': 'Average Volatility', 'Indicator': 'Category'},
            template='plotly_white',
            color='Volatility',
            color_continuous_scale=px.colors.sequential.Reds
        )
        
        fig11.update_layout(
            height=400,
            coloraxis_showscale=False,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig11, use_container_width=True)
        
        # Add insights about volatility
        most_volatile = avg_volatility['Indicator'].iloc[0]
        least_volatile = avg_volatility['Indicator'].iloc[-1]
        
        st.markdown(f"""
            <div class="insight-box">
                <strong>Volatility Analysis:</strong> {most_volatile} shows the highest price volatility, 
                suggesting less predictable price movements. This may indicate supply chain issues, seasonal availability, 
                or market disruptions. In contrast, {least_volatile} shows the most stable prices, which may reflect 
                more consistent supply, government price controls, or stable demand patterns.
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Not enough data points to calculate volatility.")
else:
    st.warning("No data available with the selected filters.")

    # Extreme Value Analysis
st.markdown('<p class="sub-header">Extreme Value Analysis</p>', unsafe_allow_html=True)

if not filtered_df.empty:
    # Find extreme values (outliers) in the data
    # Calculate z-scores
    z_score_df = filtered_df.copy()
    z_score_df['z_score'] = (z_score_df['CPI_Value'] - z_score_df['CPI_Value'].mean()) / z_score_df['CPI_Value'].std()
    
    # Define outliers (z-score > 2 or < -2)
    outliers = z_score_df[abs(z_score_df['z_score']) > 2].copy()
    
    if not outliers.empty:
        # Add columns for highlighting purposes
        outliers['Direction'] = np.where(outliers['z_score'] > 0, 'High', 'Low')
        
        # Sort by absolute z-score (most extreme first)
        outliers = outliers.sort_values(by='z_score', key=abs, ascending=False)
        
        # Show outliers in a scatter plot
        fig12 = px.scatter(
            outliers,
            x='Date',
            y='CPI_Value',
            color='Direction',
            size=abs(outliers['z_score']),
            hover_data=['Indicator', 'z_score'],
            title='Extreme CPI Values (Outliers)',
            labels={'CPI_Value': 'CPI Value', 'Date': 'Date'},
            template='plotly_white',
            color_discrete_map={'High': 'red', 'Low': 'blue'}
        )
        
        fig12.update_layout(
            height=400,
            legend_title_text='Direction',
            hovermode='closest'
        )
        
        st.plotly_chart(fig12, use_container_width=True)
        
        # Display outlier details in a table
        outlier_table = outliers[['Date', 'Indicator', 'CPI_Value', 'z_score', 'Direction']].head(10)
        outlier_table['z_score'] = outlier_table['z_score'].round(2)
        outlier_table['Date'] = outlier_table['Date'].dt.strftime('%Y-%m-%d')
        
        st.markdown('<p class="sub-header">Top 10 Most Extreme CPI Values</p>', unsafe_allow_html=True)
        st.dataframe(outlier_table, use_container_width=True, hide_index=True)
        
        # Add insights about extreme values
        outlier_categories = outliers['Indicator'].value_counts()
        most_outliers_category = outlier_categories.idxmax()
        
        st.markdown(f"""
            <div class="insight-box">
                <strong>Extreme Value Analysis:</strong> {most_outliers_category} shows the highest number of outliers ({outlier_categories.max()}) 
                in the dataset. Extreme values can indicate special events, market disruptions, policy changes, or data collection issues. 
                Decision-makers should investigate these periods to understand the underlying causes and potential economic impacts.
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No extreme values detected in the current dataset.")
else:
    st.warning("No data available with the selected filters.")

    # Policy Impact Analysis
st.markdown('<p class="sub-header">Potential Policy Impact Analysis</p>', unsafe_allow_html=True)

if not filtered_df.empty:
    # Calculate year-over-year changes for each indicator
    yoy_changes = []
    
    for indicator in filtered_df['Indicator'].unique():
        indicator_data = filtered_df[filtered_df['Indicator'] == indicator].copy()
        
        # Group by year
        yearly_avg = indicator_data.groupby('Year')['CPI_Value'].mean().reset_index()
        
        # Calculate YoY changes
        yearly_avg['YoY_Change'] = yearly_avg['CPI_Value'].pct_change() * 100
        yearly_avg['Indicator'] = indicator
        
        yoy_changes.append(yearly_avg)
    
    yoy_df = pd.concat(yoy_changes)
    yoy_df = yoy_df.dropna(subset=['YoY_Change'])
    
    if not yoy_df.empty:
        # Create heatmap of YoY changes
        pivot_yoy = yoy_df.pivot(index='Year', columns='Indicator', values='YoY_Change')
        
        fig13 = px.imshow(
            pivot_yoy,
            labels=dict(x="Category", y="Year", color="YoY Change (%)"),
            x=pivot_yoy.columns,
            y=pivot_yoy.index,
            aspect="auto",
            title='Year-over-Year CPI Changes by Category',
            color_continuous_scale='RdBu_r',
            zmin=-10,  # Set reasonable limits for better color distribution
            zmax=10
        )
        fig13.update_layout(height=400, xaxis_tickangle=-45)
        
        st.plotly_chart(fig13, use_container_width=True)
        
        # Calculate average YoY change by category
        avg_yoy = yoy_df.groupby('Indicator')['YoY_Change'].mean().sort_values(ascending=False).reset_index()
        
        # Plot average annual inflation by category
        fig14 = px.bar(
            avg_yoy,
            x='Indicator',
            y='YoY_Change',
            title='Average Annual Inflation by Category',
            labels={'YoY_Change': 'Avg. Annual Change (%)', 'Indicator': 'Category'},
            template='plotly_white',
            color='YoY_Change',
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0
        )
        
        fig14.update_layout(
            height=400,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig14, use_container_width=True)
        
        # Policy recommendations based on data
        highest_inflation = avg_yoy['Indicator'].iloc[0]
        lowest_inflation = avg_yoy['Indicator'].iloc[-1]
        
        st.markdown(f"""
            <div class="insight-box">
                <strong>Policy Implications:</strong><br>
                <ul>
                    <li><strong>High Inflation Areas:</strong> {highest_inflation} shows the highest average annual inflation 
                    ({avg_yoy['YoY_Change'].iloc[0]:.2f}%). Policymakers might consider supply-side interventions, targeted subsidies, 
                    or import policies to address price pressures in this category.</li>
                    <li><strong>Low Inflation/Deflation Areas:</strong> {lowest_inflation} shows the lowest inflation rate 
                    ({avg_yoy['YoY_Change'].iloc[-1]:.2f}%). This could indicate oversupply, decreasing demand, or effective price controls. 
                    If this rate is negative, it might warrant economic stimulus in this sector.</li>
                    <li><strong>Targeted Interventions:</strong> The heatmap highlights years and categories with exceptional inflation, 
                    suggesting where and when policy interventions might have been needed or were effective.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Not enough data points to calculate year-over-year changes.")
else:
    st.warning("No data available with the selected filters.")

    # Data Summary Table (shown on all tabs)
with st.expander("View Data Summary"):
    if not filtered_df.empty:
        # Get statistics for the filtered dataset
        summary_df = filtered_df.groupby('Indicator')['CPI_Value'].agg(['mean', 'min', 'max', 'std', 'count']).reset_index()
        summary_df.columns = ['Item', 'Average CPI', 'Minimum CPI', 'Maximum CPI', 'Standard Deviation', 'Data Points']
        
        # Format to 2 decimal places
        for col in ['Average CPI', 'Minimum CPI', 'Maximum CPI', 'Standard Deviation']:
            summary_df[col] = summary_df[col].round(2)
        
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No data available with the selected filters.")

# Show raw data (expandable)
with st.expander("View Raw Data"):
    st.dataframe(filtered_df, use_container_width=True)

# Data download option
if not filtered_df.empty:
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="sri_lanka_cpi_filtered_data.csv",
        mime="text/csv",
    )