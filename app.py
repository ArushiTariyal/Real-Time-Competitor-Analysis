import streamlit as st  # Streamlit library for creating web apps
import pandas as pd  # Pandas library for data manipulation and analysis
import numpy as np  # NumPy library for numerical computations
import plotly.express as px  # Plotly Express for creating interactive visualizations
import plotly.graph_objects as go  # Plotly Graph Objects for more complex visualizations
from datetime import datetime, timedelta  # Datetime and timedelta for handling dates and times
from sklearn.ensemble import RandomForestRegressor  # Random Forest Regressor for machine learning
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from statsmodels.tsa.arima.model import ARIMA  # ARIMA model for time series forecasting
import json  # JSON library for handling JSON data
import requests  # Requests library for making HTTP requests

# Page configuration for the Streamlit app
st.set_page_config(
    page_title="Realtime Competitor Strategy AI",  # Title of the web app
    page_icon="📊",  # Icon for the web app
    layout="wide"  # Layout set to wide for better use of screen space
)

# Custom CSS to style the app
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stPlotlyChart {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)  # Inline CSS to style the app

# Load data from CSV files
@st.cache_data(ttl=3600)  # Cache data for 1 hour to improve performance
def load_data():
    competitor_data = pd.read_csv("competitor_data.csv")  # Load competitor data from CSV
    competitor_data['Date'] = pd.to_datetime(competitor_data['Date'])  # Convert 'Date' column to datetime
    reviews_data = pd.read_csv("reviews.csv")  # Load reviews data from CSV
    reviews_data['date'] = pd.to_datetime(reviews_data['date'])  # Convert 'date' column to datetime
    return competitor_data, reviews_data  # Return both datasets

def mock_sentiment_analysis(reviews):
    """Mock sentiment analysis function to replace transformers."""
    # Simulate sentiment analysis by returning a DataFrame with positive sentiment for all reviews
    return pd.DataFrame([
        {'label': 'POSITIVE', 'score': 0.99} for _ in reviews
    ])

def train_price_predictor(data):
    """Train a Random Forest model for price prediction."""
    # Features: Price and Discount
    X = data[['Price', 'Discount']]  # Select features for the model
    # Target: Next day's price (shifted by -1)
    y = data['Price'].shift(-1)  # Shift the price column to predict the next day's price
    y = y.fillna(method='ffill')  # Fill NaN values with the last known value

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42  # 80% training, 20% testing, random state for reproducibility
    )

    # Train a Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)  # Initialize the model
    model.fit(X_train, y_train)  # Train the model
    return model  # Return the trained model

def forecast_discounts(data, days=7):
    """Forecast discounts using ARIMA."""
    # Ensure the data has a datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        data = data.set_index('Date')  # Set 'Date' as the index if not already

    # Fit an ARIMA model to the discount data
    model = ARIMA(data['Discount'], order=(5, 1, 0))  # ARIMA(5,1,0) model
    model_fit = model.fit()  # Fit the model to the data

    # Forecast discounts for the next `days` days
    forecast = model_fit.forecast(steps=days)  # Generate forecast
    # Generate future dates for the forecast
    future_dates = pd.date_range(
        start=data.index[-1] + pd.Timedelta(days=1),  # Start from the day after the last date in the data
        periods=days  # Number of days to forecast
    )

    # Return the forecasted discounts with corresponding dates
    return pd.DataFrame({
        'Date': future_dates,
        'Predicted_Discount': forecast
    })

def calculate_market_position(data, product_name):
    """Calculate market position metrics for a specific product."""
    # Get the latest data for the selected product
    product_data = data[data['product_name'] == product_name].iloc[-1]  # Get the last entry for the product
    # Get the latest data for all products
    all_products = data[data['Date'] == data['Date'].max()]  # Filter data to the latest date

    # Calculate price and discount percentiles
    price_percentile = (all_products['Price'] < product_data['Price']).mean() * 100  # Calculate price percentile
    discount_percentile = (all_products['Discount'] < product_data['Discount']).mean() * 100  # Calculate discount percentile

    return {
        'price_percentile': price_percentile,
        'discount_percentile': discount_percentile,
        'price': product_data['Price'],
        'discount': product_data['Discount']
    }

def generate_recommendations(market_position, sentiment_data, forecast_data):
    """Generate strategic recommendations based on market position, sentiment, and forecasts."""
    recommendations = []  # Initialize an empty list for recommendations

    # Price-based recommendations
    if market_position['price_percentile'] > 75:
        recommendations.append("Consider price reduction to improve market position")
    elif market_position['price_percentile'] < 25:
        recommendations.append("Potential opportunity to increase prices")

    # Discount-based recommendations
    if market_position['discount_percentile'] < 50:
        recommendations.append("Increase promotional activities to match competitor discounts")

    # Sentiment-based recommendations
    sentiment_counts = sentiment_data['label'].value_counts()  # Count the number of positive and negative sentiments
    if 'NEGATIVE' in sentiment_counts and sentiment_counts['NEGATIVE'] > sentiment_counts.get('POSITIVE', 0):
        recommendations.append("Address customer concerns to improve sentiment")

    return recommendations  # Return the list of recommendations

# Main Dashboard
def main():
    st.title("📊 Realtime Competitor Strategy AI Dashboard")  # Set the title of the dashboard

    # Load data
    competitor_data, reviews_data = load_data()  # Load the data using the load_data function

    # Sidebar filters
    st.sidebar.header("Filters")  # Add a header to the sidebar
    selected_product = st.sidebar.selectbox(
        "Select Product",  # Label for the select box
        competitor_data['product_name'].unique()  # Unique product names for the dropdown
    )

    date_range = st.sidebar.date_input(
        "Date Range",  # Label for the date input
        [
            competitor_data['Date'].max() - timedelta(days=30),  # Default start date: 30 days before the last date
            competitor_data['Date'].max()  # Default end date: the last date in the data
        ]
    )

    # Filter data based on selected product and date range
    filtered_data = competitor_data[
        (competitor_data['Date'] >= pd.Timestamp(date_range[0])) &  # Filter data from the start date
        (competitor_data['Date'] <= pd.Timestamp(date_range[1])) &  # Filter data to the end date
        (competitor_data['product_name'] == selected_product)  # Filter data for the selected product
        ]

    # Layout: Two columns for visualizations
    col1, col2 = st.columns(2)  # Create two columns for layout

    # Price Trends
    with col1:
        st.subheader("Price Trends")  # Subheader for the price trends section
        fig_price = px.line(
            filtered_data,
            x='Date',  # X-axis: Date
            y='Price',  # Y-axis: Price
            title='Historical Price Trends'  # Title of the chart
        )
        st.plotly_chart(fig_price, use_container_width=True)  # Display the chart

    # Discount Analysis
    with col2:
        st.subheader("Discount Analysis")  # Subheader for the discount analysis section
        fig_discount = px.bar(
            filtered_data,
            x='Date',  # X-axis: Date
            y='Discount',  # Y-axis: Discount
            title='Discount Distribution'  # Title of the chart
        )
        st.plotly_chart(fig_discount, use_container_width=True)  # Display the chart

    # Market Position Analysis
    st.subheader("Market Position Analysis")  # Subheader for the market position analysis section
    market_position = calculate_market_position(competitor_data, selected_product)  # Calculate market position

    col3, col4, col5 = st.columns(3)  # Create three columns for layout

    with col3:
        st.metric(
            "Price Percentile",  # Label for the metric
            f"{market_position['price_percentile']:.1f}%",  # Value of the metric
            delta=None  # No delta value
        )

    with col4:
        st.metric(
            "Discount Percentile",  # Label for the metric
            f"{market_position['discount_percentile']:.1f}%",  # Value of the metric
            delta=None  # No delta value
        )

    with col5:
        st.metric(
            "Current Price",  # Label for the metric
            f"₹{market_position['price']:,.2f}",  # Value of the metric
            delta=f"-{market_position['discount']}%"  # Delta value showing the discount
        )

    # Customer Sentiment Analysis
    st.subheader("Customer Sentiment Analysis")  # Subheader for the sentiment analysis section
    product_reviews = reviews_data[reviews_data['product_name'] == selected_product]  # Filter reviews for the selected product

    if not product_reviews.empty:
        sentiments = mock_sentiment_analysis(product_reviews['reviews'].tolist())  # Perform sentiment analysis
        fig_sentiment = px.pie(
            sentiments,
            names='label',  # Column for pie chart labels
            title='Sentiment Distribution'  # Title of the chart
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)  # Display the chart

    # Price & Discount Forecasting
    st.subheader("Price & Discount Forecasting")  # Subheader for the forecasting section
    forecast_data = forecast_discounts(filtered_data)  # Forecast discounts

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=filtered_data['Date'],  # X-axis: Historical dates
        y=filtered_data['Discount'],  # Y-axis: Historical discounts
        name='Historical Discounts'  # Name for the trace
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast_data['Date'],  # X-axis: Forecasted dates
        y=forecast_data['Predicted_Discount'],  # Y-axis: Forecasted discounts
        name='Forecasted Discounts',  # Name for the trace
        line=dict(dash='dash')  # Dashed line for forecasted data
    ))
    st.plotly_chart(fig_forecast, use_container_width=True)  # Display the chart

    # Strategic Recommendations
    st.subheader("Strategic Recommendations")  # Subheader for the recommendations section
    recommendations = generate_recommendations(
        market_position,
        sentiments if not product_reviews.empty else pd.DataFrame(),  # Pass sentiment data if available
        forecast_data
    )

    for i, rec in enumerate(recommendations, 1):  # Loop through recommendations
        st.write(f"{i}. {rec}")  # Display each recommendation

if __name__ == "__main__":
    main()  # Run the main function when the script is executed