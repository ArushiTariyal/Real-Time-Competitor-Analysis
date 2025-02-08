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
    
    if not isinstance(data.index, pd.DatetimeIndex):
        data = data.set_index('date')

    # Ensure discount column is numeric
    data['discount'] = pd.to_numeric(data['discount'], errors='coerce')
    data = data.dropna(subset=['discount'])  # Remove NaN values
    
    # Ensure enough data points
    if len(data) < 5:
        raise ValueError("Not enough data points for ARIMA forecasting.")

    # Ensure non-constant discount values
    if data['discount'].std() == 0:
        raise ValueError("Constant discount values cannot be used for ARIMA.")

    # Set explicit date frequency
    try:
        data = data.asfreq(pd.infer_freq(data.index))
    except:
        data = data.asfreq('D')  # Force daily frequency

    # Fit ARIMA
    model = ARIMA(data['discount'], order=(1,0,0))
    model_fit = model.fit()

    # Forecast
    forecast = model_fit.forecast(steps=days)
    future_dates = pd.date_range(
        start=data.index[-1] + pd.Timedelta(days=1),
        periods=days
    )

    return pd.DataFrame({'Date': future_dates, 'Predicted_Discount': forecast})


def calculate_market_position(data, product_name):
    """Calculate market position metrics."""
    product_data = data[data['product_name'] == product_name].iloc[-1]
    all_products = data[data['date'] == data['date'].max()]
    
    price_percentile = (all_products['price'] < product_data['price']).mean() * 100
    discount_percentile = (all_products['discount'] < product_data['discount']).mean() * 100
    
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
        recommendations.append("Consider price reduction to improve market position")
    elif market_position['price_percentile'] < 25:
        recommendations.append("Potential opportunity to increase prices")
        
    # Discount-based recommendations
    if market_position['discount_percentile'] < 50:
        recommendations.append("Increase promotional activities to match competitor discounts")
    
    # Sentiment-based recommendations
    sentiment_counts = sentiment_data['label'].value_counts()
    if 'NEGATIVE' in sentiment_counts and sentiment_counts['NEGATIVE'] > sentiment_counts.get('POSITIVE', 0):
        recommendations.append("Address customer concerns to improve sentiment")
        
    return recommendations

# Main Dashboard
def main():
    st.title("📊 Realtime Competitor Strategy AI Dashboard")
    
    # Load data
    competitor_data, reviews_data = load_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    selected_product = st.sidebar.selectbox(
        "Select Product",
        competitor_data['product_name'].unique()
    )
    
    date_range = st.sidebar.date_input(
        "Date Range",
        [
            competitor_data['date'].max() - timedelta(days=30),
            competitor_data['date'].max()
        ]
    )
    
    # Filter data
    filtered_data = competitor_data[
        (competitor_data['date'] >= pd.Timestamp(date_range[0])) &
        (competitor_data['date'] <= pd.Timestamp(date_range[1])) &
        (competitor_data['product_name'] == selected_product)
    ]
    
    # Layout
    col1, col2 = st.columns(2)
    
    # Price Trends
    with col1:
        st.subheader("Price Trends")
        fig_price = px.line(
            filtered_data,
            x='date',
            y='price',
            title='Historical Price Trends'
        )
        st.plotly_chart(fig_price, use_container_width=True)
    
    # Discount Analysis
    with col2:
        st.subheader("Discount Analysis")
        fig_discount = px.bar(
            filtered_data,
            x='date',
            y='discount',
            title='Discount Distribution'
        )
        st.plotly_chart(fig_discount, use_container_width=True)
    
    # Market Position
    st.subheader("Market Position Analysis")
    market_position = calculate_market_position(competitor_data, selected_product)
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric(
            "Price Percentile",
            f"{market_position['price_percentile']:.1f}%",
            delta=None
        )
    
    with col4:
        st.metric(
            "Discount Percentile",
            f"{market_position['discount_percentile']:.1f}%",
            delta=None
        )
    
    with col5:
        st.metric(
            "Current Price",
            f"₹{market_position['price']:,.2f}",
            delta=f"-{market_position['discount']}%"
        )
    
    # Sentiment Analysis
    st.subheader("Customer Sentiment Analysis")
    product_reviews = reviews_data[reviews_data['product_name'] == selected_product]
    
    if not product_reviews.empty:
        sentiments = mock_sentiment_analysis(product_reviews['review'].tolist())
        fig_sentiment = px.pie(
            sentiments,
            names='label',
            title='Sentiment Distribution'
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # Forecasting
    st.subheader("Price & Discount Forecasting")
    forecast_data = forecast_discounts(filtered_data)
    
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=filtered_data['date'],
        y=filtered_data['discount'],
        name='Historical Discounts'
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast_data['Date'],
        y=forecast_data['Predicted_Discount'],
        name='Forecasted Discounts',
        line=dict(dash='dash')
    ))
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Strategic Recommendations
    st.subheader("Strategic Recommendations")
    recommendations = generate_recommendations(
        market_position,
        sentiments if not product_reviews.empty else pd.DataFrame(),
        forecast_data
    )
    
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")

if __name__ == "__main__":
    main()