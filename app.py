import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_cpi_sri_lanka.csv')
    return df

df = load_data()

# ---------------- Main Dashboard ----------------

st.title("🌐 Sri Lanka Consumer Price Index (CPI) Dashboard")

# Metrics
st.markdown("### 📊 Key Metrics")
col1, col2, col3 = st.columns(3)

latest_cpi = df.sort_values('Start_Date', ascending=False).iloc[0]['CPI_Value']
highest_cpi_year = df.groupby('Year')['CPI_Value'].mean().idxmax()
avg_cpi = df['CPI_Value'].mean()

col1.metric("Latest CPI Value", f"{latest_cpi:.2f}")
col2.metric("Year with Highest Avg CPI", highest_cpi_year)
col3.metric("Overall Avg CPI", f"{avg_cpi:.2f}")

st.markdown("---")

# 1. CPI Trend Over Time
st.subheader("📈 CPI Trend Over Time")
years = sorted(df['Year'].unique())
selected_years_trend = st.multiselect("Select Year(s) for Trend Analysis", options=years, default=years)
filtered_trend_df = df[df['Year'].isin(selected_years_trend)]
fig1 = px.line(filtered_trend_df, x='Start_Date', y='CPI_Value', title='CPI Over Time', markers=True)
fig1.update_layout(xaxis_title='Date', yaxis_title='CPI Value')
st.plotly_chart(fig1, use_container_width=True)

# 2. Monthly Seasonality
st.subheader("📍 Monthly Seasonality Analysis")
months = sorted(df['Month'].unique())
selected_months_seasonality = st.multiselect("Select Month(s) for Seasonality Analysis", options=months, default=months)
filtered_seasonality_df = df[df['Month'].isin(selected_months_seasonality)]
fig2 = px.box(filtered_seasonality_df, x='Month', y='CPI_Value', color='Month', title='CPI Distribution by Month')
fig2.update_layout(xaxis_title='Month', yaxis_title='CPI Value')
st.plotly_chart(fig2, use_container_width=True)

# 3. Year-on-Year CPI Comparison
st.subheader("📅 Average CPI by Year")
yearly_avg = df.groupby('Year')['CPI_Value'].mean().reset_index()
selected_years_yoy = st.multiselect("Select Year(s) for Year-on-Year Comparison", options=years, default=years)
filtered_yearly_avg = yearly_avg[yearly_avg['Year'].isin(selected_years_yoy)]
fig3 = px.bar(filtered_yearly_avg, x='Year', y='CPI_Value', title='Average CPI Per Year', color='CPI_Value', color_continuous_scale='blues')
fig3.update_layout(xaxis_title='Year', yaxis_title='Average CPI Value')
st.plotly_chart(fig3, use_container_width=True)

# 4. Heatmap Month vs Year
st.subheader("🔍 CPI Heatmap (Month vs Year)")
fig4 = go.Figure(data=go.Heatmap(
    z=df.pivot_table(index='Month', columns='Year', values='CPI_Value', aggfunc='mean').values,
    x=df.pivot_table(index='Month', columns='Year', values='CPI_Value', aggfunc='mean').columns,
    y=df.pivot_table(index='Month', columns='Year', values='CPI_Value', aggfunc='mean').index,
    colorscale='Viridis'))
fig4.update_layout(title='CPI Heatmap: Month vs Year', xaxis_title='Year', yaxis_title='Month')
st.plotly_chart(fig4, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Developed by [Your Name] for 5DATA004W | 📱")
