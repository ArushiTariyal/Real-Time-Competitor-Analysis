# Import necessary libraries
import streamlit as st  # For creating the web app interface
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
import plotly.express as px  # For creating interactive visualizations
import plotly.graph_objects as go  # For advanced custom visualizations
from datetime import datetime, timedelta  # For handling date and time
from sklearn.ensemble import RandomForestRegressor  # For training a Random Forest model
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from statsmodels.tsa.arima.model import ARIMA  # For time series forecasting using ARIMA
import json  # For handling JSON data
import requests  # For making HTTP requests

# Configure the Streamlit page settings
st.set_page_config(
    page_title="Realtime Competitor Strategy AI",  # Title of the web app
    page_icon="📊",  # Icon for the web app
    layout="wide"  # Use a wide layout for better use of screen space
)

# Add custom CSS to style the web app
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;  # Add padding to the main content area
    }
    .stPlotlyChart {
        background-color: #ffffff;  # Set background color for Plotly charts
        border-radius: 5px;  # Add rounded corners to the charts
        padding: 1rem;  # Add padding around the charts
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);  # Add a subtle shadow to the charts
    }
    </style>
""", unsafe_allow_html=True)  # Allow HTML in the markdown for custom styling


# Function to load data with caching to improve performance
@st.cache_data(ttl=3600)  # Cache the data for 1 hour (3600 seconds)
def load_data():
    # Load competitor data from a CSV file and parse the 'date' column as datetime
    competitor_data = pd.read_csv("competitor_data.csv")
    competitor_data['date'] = pd.to_datetime(competitor_data['date'])

    # Load reviews data from a CSV file and parse the 'date' column as datetime
    reviews_data = pd.read_csv("reviews.csv")
    reviews_data['date'] = pd.to_datetime(reviews_data['date'])

    # Return the loaded data
    return competitor_data, reviews_data


# Mock sentiment analysis function (placeholder for actual sentiment analysis)
def mock_sentiment_analysis(reviews):
    """Mock sentiment analysis function to replace transformers."""
    # Create a DataFrame with mock sentiment analysis results
    # Assumes all reviews are positive for testing purposes
    return pd.DataFrame([
        {'label': 'POSITIVE', 'score': 0.99} for _ in reviews
    ])


# Function to train a Random Forest model for price prediction
def train_price_predictor(data):
    """Train Random Forest model for price prediction."""
    # Define features (X) and target (y) for the model
    X = data[['Price', 'Discount']]  # Features: Price and Discount
    y = data['Price'].shift(-1)  # Target: Next day's price (shifted by -1)
    y = y.fillna(method='ffill')  # Fill any NaN values in the target with the last valid value

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42  # 80% training, 20% testing, fixed random state for reproducibility
    )

    # Initialize and train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)  # 100 trees in the forest
    model.fit(X_train, y_train)  # Train the model on the training data

    # Return the trained model
    return model


# Function to forecast discounts using ARIMA (commented out for now)
# def forecast_discounts(data, days=7):
#     """Forecast discounts using ARIMA."""
#     # Ensure the data has a datetime index
#     if not isinstance(data.index, pd.DatetimeIndex):
#         data = data.set_index('date')

#     # Initialize and fit the ARIMA model
#     model = ARIMA(data['discount'], order=(5,1,0))  # ARIMA model with order (5,1,0)
#     model_fit = model.fit()  # Fit the model to the data

#     # Forecast discounts for the specified number of days
#     forecast = model_fit.forecast(steps=days)

#     # Generate future dates for the forecast period
#     future_dates = pd.date_range(
#         start=data.index[-1] + pd.Timedelta(days=1),  # Start from the day after the last date in the data
#         periods=days  # Number of days to forecast
#     )

#     # Return the forecasted discounts with corresponding dates
#     return pd.DataFrame({
#         'Date': future_dates,
#         'Predicted_Discount': forecast
#     })

def forecast_discounts(data, days=7):
    """Forecast discounts using ARIMA."""

    # Check if the data index is not a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        # If not, set the 'date' column as the index
        data = data.set_index('date')

    # Ensure the 'discount' column is numeric, coercing non-numeric values to NaN
    data['discount'] = pd.to_numeric(data['discount'], errors='coerce')
    # Drop rows where the 'discount' column has NaN values
    data = data.dropna(subset=['discount'])

    # Check if there are at least 5 data points for ARIMA forecasting
    if len(data) < 5:
        # Raise an error if there are not enough data points
        raise ValueError("Not enough data points for ARIMA forecasting.")

    # Check if the discount values are constant (standard deviation is 0)
    if data['discount'].std() == 0:
        # Raise an error if discount values are constant
        raise ValueError("Constant discount values cannot be used for ARIMA.")

    # Try to infer the frequency of the date index and set it explicitly
    try:
        data = data.asfreq(pd.infer_freq(data.index))
    except:
        # If inference fails, force the frequency to daily ('D')
        data = data.asfreq('D')

        # Fit an ARIMA model with order (1, 0, 0) to the 'discount' column
    model = ARIMA(data['discount'], order=(1, 0, 0))
    # Train the ARIMA model
    model_fit = model.fit()

    # Forecast the discount values for the specified number of days
    forecast = model_fit.forecast(steps=days)
    # Generate future dates for the forecasted period
    future_dates = pd.date_range(
        start=data.index[-1] + pd.Timedelta(days=1),  # Start from the day after the last date
        periods=days  # Number of days to forecast
    )

    # Return a DataFrame with future dates and predicted discounts
    return pd.DataFrame({'Date': future_dates, 'Predicted_Discount': forecast})


def calculate_market_position(data, product_name):
    """Calculate market position metrics."""
    # Filter data for the specified product and get the latest entry
    product_data = data[data['product_name'] == product_name].iloc[-1]
    # Filter data for the most recent date
    all_products = data[data['date'] == data['date'].max()]

    # Calculate the percentile of the product's price compared to all products
    price_percentile = (all_products['price'] < product_data['price']).mean() * 100
    # Calculate the percentile of the product's discount compared to all products
    discount_percentile = (all_products['discount'] < product_data['discount']).mean() * 100

    # Return a dictionary with market position metrics
    return {
        'price_percentile': price_percentile,
        'discount_percentile': discount_percentile,
        'price': product_data['price'],
        'discount': product_data['discount']
    }


def generate_recommendations(market_position, sentiment_data, forecast_data):
    """Generate strategic recommendations."""
    recommendations = []

    # Price-based recommendations
    if market_position['price_percentile'] > 75:
        # If the product's price is in the top 25%, suggest reducing the price
        recommendations.append("Consider price reduction to improve market position")
    elif market_position['price_percentile'] < 25:
        # If the product's price is in the bottom 25%, suggest increasing the price
        recommendations.append("Potential opportunity to increase prices")

    # Discount-based recommendations
    if market_position['discount_percentile'] < 50:
        # If the product's discount is below the median, suggest increasing promotions
        recommendations.append("Increase promotional activities to match competitor discounts")

    # Sentiment-based recommendations
    sentiment_counts = sentiment_data['label'].value_counts()
    if 'NEGATIVE' in sentiment_counts and sentiment_counts['NEGATIVE'] > sentiment_counts.get('POSITIVE', 0):
        # If negative sentiment outweighs positive sentiment, suggest addressing concerns
        recommendations.append("Address customer concerns to improve sentiment")

    # Return the list of recommendations
    return recommendations


# Main Dashboard
def main():
    # Set the title of the Streamlit dashboard
    st.title("📊 Realtime Competitor Strategy AI Dashboard")

    # Load competitor and reviews data
    competitor_data, reviews_data = load_data()

    # Sidebar filters
    st.sidebar.header("Filters")  # Sidebar section for filtering options

    # Dropdown for selecting a product from the dataset
    selected_product = st.sidebar.selectbox(
        "Select Product",
        competitor_data['product_name'].unique()  # Get unique product names for selection
    )

    # Date range filter for selecting data within a specific time frame
    date_range = st.sidebar.date_input(
        "Date Range",
        [
            competitor_data['date'].max() - timedelta(days=30),  # Default start date (30 days ago)
            competitor_data['date'].max()  # Default end date (most recent date)
        ]
    )

    # Filter the competitor data based on selected product and date range
    filtered_data = competitor_data[
        (competitor_data['date'] >= pd.Timestamp(date_range[0])) &  # Apply start date filter
        (competitor_data['date'] <= pd.Timestamp(date_range[1])) &  # Apply end date filter
        (competitor_data['product_name'] == selected_product)  # Apply product selection filter
        ]

    # Define layout with two columns for visualization
    col1, col2 = st.columns(2)

    # Price Trends Visualization
    with col1:
        st.subheader("Price Trends")  # Section title for price trends
        fig_price = px.line(
            filtered_data,
            x='date',  # Set x-axis as date
            y='price',  # Set y-axis as price
            title='Historical Price Trends'  # Title of the chart
        )
        st.plotly_chart(fig_price, use_container_width=True)  # Display the price trend chart

    # Discount Analysis Visualization
    with col2:
        st.subheader("Discount Analysis")  # Section title for discount trends
        fig_discount = px.bar(
            filtered_data,
            x='date',  # Set x-axis as date
            y='discount',  # Set y-axis as discount percentage
            title='Discount Distribution'  # Title of the chart
        )
        st.plotly_chart(fig_discount, use_container_width=True)  # Display the discount trend chart

    # Market Position Analysis
    st.subheader("Market Position Analysis")  # Section title for market positioning
    market_position = calculate_market_position(competitor_data, selected_product)  # Compute market position

    # Define layout with three columns for displaying key metrics
    col3, col4, col5 = st.columns(3)

    # Display price percentile metric
    with col3:
        st.metric(
            "Price Percentile",
            f"{market_position['price_percentile']:.1f}%",  # Format percentile value
            delta=None  # No change indicator
        )

    # Display discount percentile metric
    with col4:
        st.metric(
            "Discount Percentile",
            f"{market_position['discount_percentile']:.1f}%",  # Format percentile value
            delta=None  # No change indicator
        )

    # Display current price metric
    with col5:
        st.metric(
            "Current Price",
            f"₹{market_position['price']:,.2f}",  # Format price with currency
            delta=f"-{market_position['discount']}%"  # Show discount percentage as delta
        )

    # Sentiment Analysis Section
    st.subheader("Customer Sentiment Analysis")  # Section title for sentiment analysis
    product_reviews = reviews_data[
        reviews_data['product_name'] == selected_product]  # Filter reviews for selected product

    # Check if there are any reviews available
    if not product_reviews.empty:
        sentiments = mock_sentiment_analysis(product_reviews['review'].tolist())  # Perform sentiment analysis
        fig_sentiment = px.pie(
            sentiments,
            names='label',  # Define sentiment labels for pie chart
            title='Sentiment Distribution'  # Title of sentiment analysis chart
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)  # Display sentiment distribution

    # Forecasting Section
    st.subheader("Price & Discount Forecasting")  # Section title for forecasting
    forecast_data = forecast_discounts(filtered_data)  # Generate forecasted data

    # Create a figure for forecast visualization
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=filtered_data['date'],  # X-axis: historical dates
        y=filtered_data['discount'],  # Y-axis: historical discount values
        name='Historical Discounts'  # Label for historical discount data
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast_data['Date'],  # X-axis: future forecasted dates
        y=forecast_data['Predicted_Discount'],  # Y-axis: predicted discount values
        name='Forecasted Discounts',  # Label for forecasted discount data
        line=dict(dash='dash')  # Dashed line style for forecasted values
    ))
    st.plotly_chart(fig_forecast, use_container_width=True)  # Display the forecast visualization

    # Strategic Recommendations Section
    st.subheader("Strategic Recommendations")  # Section title for strategy suggestions
    recommendations = generate_recommendations(
        market_position,  # Pass market position data
        sentiments if not product_reviews.empty else pd.DataFrame(),  # Pass sentiment data if available
        forecast_data  # Pass forecasted discount data
    )

    # Loop through recommendations and display them as a numbered list
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")  # Display each recommendation with numbering


# Entry point of the script
if __name__ == "__main__":
    main()  # Call the main function to run the dashboard