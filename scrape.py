# Import necessary libraries
import json  # For handling JSON data
import time  # For adding delays and timing operations
from datetime import datetime  # For working with date and time
import pandas as pd  # For data manipulation and analysis
import requests  # For making HTTP requests
import plotly.express as px  # For data visualization
import streamlit as st  # For building web applications
from selenium import webdriver  # For web scraping using Selenium
from selenium.webdriver.chrome.options import Options  # For setting Chrome options
from selenium.webdriver.chrome.service import Service  # For managing the ChromeDriver service
from selenium.webdriver.common.by import By  # For locating elements in the DOM
from selenium.webdriver.support import expected_conditions as EC  # For setting expected conditions
from selenium.webdriver.support.wait import WebDriverWait  # For waiting until an element is found
from webdriver_manager.chrome import ChromeDriverManager  # For managing the ChromeDriver version
import chromedriver_autoinstaller  # For automatically installing the correct ChromeDriver version
from transformers import pipeline  # For using pre-trained models from Hugging Face
from sklearn.ensemble import RandomForestRegressor  # For machine learning regression tasks
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from statsmodels.tsa.arima.model import ARIMA  # For time series forecasting using ARIMA
from selenium.webdriver.chrome.service import Service  # Duplicate import, can be removed

# Dictionary to store product links
links = {
    "Apple iPhone 13 (128GB) - Green": "https://www.amazon.in/Apple-iPhone-13-128GB-Green/dp/B09V4B6K53...",
}

# Function to configure and return a Chrome WebDriver instance
def get_driver():
    chrome_options = Options()  # Initialize Chrome options
    chrome_options.add_argument("--headless")  # Run in headless mode (no GUI)
    chrome_options.add_argument("--no-sandbox")  # Disable sandboxing (useful for running in containers)
    chrome_options.add_argument("--disable-dev-shm-usage")  # Avoid issues with shared memory

    chromedriver_autoinstaller.install()  # Automatically install the correct version of ChromeDriver

    driver = webdriver.Chrome(options=chrome_options)  # Initialize Chrome WebDriver with options
    return driver  # Return the driver instance

# Function to scrape product details from a given link
def scrape_product_data(link):
    driver = get_driver()  # Get a new Chrome WebDriver instance
    driver.set_window_size(1920, 1080)  # Set window size for consistency
    driver.get(link)  # Open the product link

    # Initialize product data dictionary
    product_data = {
        "product_name": "",  # Placeholder for product name
        "selling price": 0,  # Placeholder for selling price
        "original price": 0,  # Placeholder for original price
        "discount": 0,  # Placeholder for discount
        "rating": 0,  # Placeholder for rating
        "reviews": [],  # Placeholder for reviews
        "product_url": link,  # Store product URL
        "date": datetime.now().strftime("%Y-%m-%d"),  # Store current date
    }

    retry = 0  # Retry counter for handling failures
    while retry < 3:  # Try scraping up to 3 times
        try:
            driver.save_screenshot("screenshot.png")  # Take a screenshot for debugging
            wait = WebDriverWait(driver, 10)  # Set up explicit wait for 10 seconds
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "a-offscreen")))  # Wait for price element
            break  # Exit retry loop if successful
        except Exception as e:
            print(f"Retrying... Error: {e}")  # Print error message
            retry += 1  # Increment retry counter
            driver.get(link)  # Reload the page
            time.sleep(5)  # Wait before retrying

    # Try extracting selling price
    try:
        price_elem = driver.find_element(
            By.XPATH, '//*[@id="corePriceDisplay_desktop_feature_div"]/div[1]/span[3]/span[2]/span[2]'
        )
        product_data["selling price"] = int("".join(price_elem.text.strip().split(",")))  # Convert price to integer
    except Exception as e:
        print(f"Error extracting selling price: {e}")  # Print error message

    # Try extracting original price
    try:
        original_price = driver.find_element(
            By.XPATH, '//*[@id="corePriceDisplay_desktop_feature_div"]/div[2]/span/span[1]/span[2]/span/span[2]'
        ).text
        product_data["original price"] = extract_price(original_price)  # Convert price text to integer
    except Exception as e:
        print(f"Error extracting original price: {e}")  # Print error message

    # Try extracting discount information
    try:
        discount = driver.find_element(
            By.XPATH, '//*[@id="corePriceDisplay_desktop_feature_div"]/div[1]/span[2]'
        )
        full_rating_text = discount.get_attribute("innerHTML").strip()  # Get discount text
        if " out of 5 stars" in full_rating_text.lower():  # Check if it's a rating instead
            product_data["rating"] = full_rating_text.lower().split(" out of")[0].strip()  # Extract rating
        else:
            product_data["discount"] = full_rating_text  # Store discount
    except Exception as e:
        print(f"Error extracting discount: {e}")  # Print error message

    # Try extracting rating
    try:
        rating_elem = driver.find_element(By.CLASS_NAME, "a-icon-star")  # Locate rating element
        product_data["rating"] = rating_elem.get_attribute("innerText").strip()  # Extract rating text
    except Exception as e:
        print(f"Error extracting rating: {e}")  # Print error message

    # Try extracting customer reviews
    try:
        reviews_link_elements = driver.find_elements(
            By.XPATH, "//a[contains(text(), 'See customer reviews')]"
        )
        if reviews_link_elements:
            reviews_link = reviews_link_elements[-1].get_attribute("href")  # Get reviews page link
            driver.get(reviews_link)  # Navigate to reviews page
            time.sleep(3)  # Wait for page to load

            reviews_section = driver.find_element(By.ID, "cm-cr-dp-review-list")  # Locate review section
            review_elements = reviews_section.find_elements(By.TAG_NAME, "li")  # Find all reviews

            for review in review_elements:
                product_data["reviews"].append(review.text.strip())  # Add review text to list
        else:
            print("No customer reviews found.")  # Print message if no reviews found
    except Exception as e:
        print(f"Error extracting reviews: {e}")  # Print error message

    driver.quit()  # Close the browser
    return product_data  # Return collected product data

import re  # Import regex module for text processing

# Function to extract numerical price from text
def extract_price(price_text):
    """Extracts and converts price from a string with currency symbols or commas."""
    price_text = re.sub(r"[^\d]", "", price_text)  # Remove non-numeric characters
    return int(price_text) if price_text else 0  # Convert to integer, return 0 if empty

# Function to extract rating from review text
def extract_rating_from_review(review_text):
    match = re.search(r"(\d+\.\d+) out of 5 stars", review_text)  # Search for rating pattern
    if match:
        return float(match.group(1))  # Convert matched rating to float
    return None  # Return None if no match

# Iterate over product links and scrape data
for product_name, link in links.items():
    product_data = scrape_product_data(link)  # Scrape data for each product

    # Try loading existing reviews CSV file
    try:
        reviews_df = pd.read_csv("reviews.csv")  # Read reviews CSV file
    except FileNotFoundError:
        reviews_df = pd.DataFrame(columns=["product_name", "review", "rating", "date"])  # Create empty DataFrame if file is missing

    new_reviews = []  # Initialize a list to store new review entries
    for review_text in product_data["reviews"]:  # Iterate over scraped reviews
        rating = extract_rating_from_review(review_text)  # Extract rating from review text
        new_reviews.append({  # Append extracted review data to the list
            "product_name": product_name,  # Store product name
            "review": review_text,  # Store review text
            "rating": rating,  # Store extracted rating
            "date": datetime.now().strftime("%Y-%m-%d")  # Store current date
        })

    new_reviews_df = pd.DataFrame(new_reviews)  # Convert new reviews list to DataFrame
    reviews_df = pd.concat([reviews_df, new_reviews_df], ignore_index=True)  # Merge new reviews with existing ones
    reviews_df.to_csv("reviews.csv", index=False)  # Save updated reviews to CSV file

    # Try loading competitor data CSV file
    try:
        competitor_df = pd.read_csv("competitor_data.csv")  # Read competitor data CSV
        competitor_df = competitor_df[['product_name', 'price', 'discount', 'date']]  # Ensure only required columns exist
    except FileNotFoundError:
        competitor_df = pd.DataFrame(columns=["product_name", "price", "discount", "date"])  # Create empty DataFrame if file is missing

    # Create new data entry for competitor data
    new_data = {
        "product_name": product_name,  # Store product name
        "price": product_data["selling price"],  # Store selling price
        "discount": product_data["discount"],  # Store discount
        "date": datetime.now().strftime("%Y-%m-%d"),  # Store current date
    }

    new_data_df = pd.DataFrame([new_data], columns=["product_name", "price", "discount", "date"])  # Convert data to DataFrame

    competitor_df = pd.concat([competitor_df, new_data_df], ignore_index=True)  # Append new data to existing data
    competitor_df.to_csv("competitor_data.csv", index=False)  # Save updated competitor data to CSV file

# API keys
'''
This is an API key used to authenticate requests to a service or platform. 
In this case, the key is for the Groq API (likely used for AI or machine learning tasks).
API keys are used to validate that the request comes from a trusted source and often grant access to 
specific functionalities or data within an API.
The key provided is likely part of a larger system that facilitates secure communication with Groq's servers 
for performing tasks such as generating recommendations or interacting with their models.
'''
'''This is a Slack webhook URL. A webhook in Slack is a way for external systems to send messages into a Slack channel automatically.
In this case, the webhook URL is specific to a particular Slack workspace and channel, 
allowing the program to send notifications or updates directly into that Slack channel. 
The webhook URL is like an endpoint where messages are posted from an external source, 
and it is set up to trigger when you make an HTTP request to it.'''

API_KEY = "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"  # Groq API Key
SLACK_WEBHOOK = "https://hooks.slack.com/services/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"  # Slack webhook URL

# Streamlit app setup
st.set_page_config(layout="wide")  # Set Streamlit page layout to wide mode

# Create two columns for layout
col1, col2 = st.columns(2)

# Add title to the first column
with col1:
    st.markdown(
        """
        <div style="font-size: 40px; text-align: left; width: 100%;">
            ❄️❄️❄️<strong>E-Commerce Competitor Strategy Dashboard</strong>❄️❄️❄️
        </div>
        """,
        unsafe_allow_html=True,  # Allow HTML formatting
    )

# Add GIF to the second column
with col2:
    st.markdown(
        """
        <div style="text-align: right;">
            <img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExbzh4dXpuc2hpY3JlNnR1MDdiMXozMXlreHFoZjl0a2g5anJqNWxtMCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/hWe6YajFuxX41eV8I0/giphy.gif" alt="Engaging GIF" width="300">
        </div>
        """,
        unsafe_allow_html=True,  # Allow HTML formatting
    )

# Utility function to truncate text
def truncate_text(text, max_length=512):
    """Truncate text to a maximum length."""
    return text[:max_length]  # Return truncated text if it exceeds max_length

# Load competitor data from CSV file
def load_competitor_data():
    """Load competitor data from a CSV file."""
    data = pd.read_csv("competitor_data.csv")  # Read competitor data from CSV
    st.write(data.head())  # Display first few rows for debugging
    return data  # Return loaded data

# Load reviews data from CSV file
def load_reviews_data():
    """Load reviews data from a CSV file."""
    reviews = pd.read_csv("reviews.csv")  # Read reviews data from CSV
    return reviews  # Return loaded reviews

# Analyze customer sentiment using Hugging Face transformers
def analyze_sentiment(reviews):
    """Analyze customer sentiment for reviews."""
    sentiment_pipeline = pipeline("sentiment-analysis")  # Load sentiment analysis pipeline
    return sentiment_pipeline(reviews)  # Perform sentiment analysis on reviews

# Train predictive model using RandomForestRegressor
def train_predictive_model(data):
    """Train a predictive model for competitor pricing strategy."""
    data["Discount"] = data["Discount"].str.replace("%", "").astype(float)  # Convert discount to numeric format
    data["Price"] = data["Price"].astype(float)  # Ensure price is numeric
    data["Predicted_Discount"] = data["Discount"] + (data["Price"] * 0.05).round(2)  # Compute predicted discount

    X = data[["Price", "Discount"]]  # Features: Price and Discount
    y = data["Predicted_Discount"]  # Target: Predicted discount

    # Split data into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)  # Initialize Random Forest Regressor
    model.fit(X_train, y_train)  # Train the model
    return model  # Return trained model

# Forecast discounts using ARIMA
def forecast_discounts_arima(data, future_days=5):
    """
    Forecast future discounts using ARIMA.
    :param data: DataFrame containing historical discount data (with a datetime index).
    :param future_days: Number of days to forecast.
    :return: DataFrame with historical and forecasted discounts.
    """

    # Sort the data by the index (date) to ensure it's in chronological order
    data = data.sort_index()

    # Convert the 'discount' column to numeric values, coercing any errors to NaN
    data["discount"] = pd.to_numeric(data["discount"], errors="coerce")

    # Drop rows with NaN values in the 'discount' column
    data = data.dropna(subset=["discount"])

    # Extract the 'discount' column as a time series
    discount_series = data["discount"]

    # Ensure that the index is a datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            # Try converting the index to a datetime index if it's not already
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            # Raise an error if the conversion fails
            raise ValueError("Index must be datetime or convertible to datetime.") from e

    # Create an ARIMA model with the specified order (p=5, d=1, q=0)
    model = ARIMA(discount_series, order=(5, 1, 0))

    # Fit the ARIMA model to the historical discount data
    model_fit = model.fit()

    # Generate a forecast for the next 'future_days' days
    forecast = model_fit.forecast(steps=future_days)

    # Create a date range for the forecasted days, starting from the last known date + 1 day
    future_dates = pd.date_range(
        start=discount_series.index[-1] + pd.Timedelta(days=1),
        periods=future_days  # Generate forecasted periods based on 'future_days'
    )

    # Create a DataFrame with the forecasted dates and predicted discounts
    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Discount": forecast})

    # Set the 'Date' column as the index of the DataFrame
    forecast_df.set_index("Date", inplace=True)

    # Return the DataFrame containing the forecasted values
    return forecast_df


# Send notifications to Slack
def send_to_slack(data):
    """
    Send data (text message) as a Slack notification.
    :param data: The message content to be sent to Slack.
    """
    # Create a dictionary with the text message to send to Slack
    payload = {"text": data}

    # Send the message to Slack using an HTTP POST request
    response = requests.post(
        SLACK_WEBHOOK,  # The webhook URL for Slack
        data=json.dumps(payload),  # Convert the payload dictionary to JSON format
        headers={"Content-Type": "application/json"}  # Set the content type as JSON
    )

    # Check if the response status code is not 200 (indicating a failure)
    if response.status_code != 200:
        # Display an error message in the Streamlit app if the request fails
        st.write(f"Failed to send notification to Slack: {response.status_code}")


# Generate strategy recommendations using an LLM (Large Language Model)
def generate_strategy_recommendation(product_name, competitor_data, sentiment):
    """
    Generate strategic recommendations using an LLM.
    :param product_name: The name of the product being analyzed.
    :param competitor_data: Data about competitors' prices and discounts.
    :param sentiment: Sentiment analysis results for customer reviews.
    :return: A string containing the strategic recommendations.
    """
    # Get the current date for the prompt
    date = datetime.now()

    # Create a prompt to generate business strategies using the product name, competitor data, and sentiment
    prompt = f"""
    You are a highly skilled business strategist specializing in e-commerce. Based on the following details, suggest actionable strategies:

    *Product Name*: {product_name}
    *Competitor Data* (including current prices, discounts, and predicted discounts):
    {competitor_data}
    *Sentiment Analysis*: {sentiment}
    *Today's Date*: {str(date)}

    # Task:
    - Analyze the competitor data and identify key pricing trends.
    - Leverage sentiment analysis insights to highlight areas where customer satisfaction can be improved.
    - Use the discount predictions to suggest how pricing strategies can be optimized over the next 5 days.
    - Recommend promotional campaigns or marketing strategies that align with customer sentiments and competitive trends.

    Provide your recommendations in a structured format:
    - **Pricing Strategy**
    - **Promotional Campaign Ideas**
    - **Customer Satisfaction Recommendations**
    """

    # Prepare the request data to send to the LLM API
    data = {
        "messages": [{"role": "user", "content": prompt}],  # Send the prompt as a user message
        "model": "llama3-8b-8192",  # Specify the model to use
        "temperature": 0,  # Set the model's creativity level (0 for deterministic responses)
    }

    # Set the headers for the API request, including the authorization token
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

    # Send the request to the LLM API to generate recommendations
    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",  # The API endpoint for LLM completions
        data=json.dumps(data),  # Convert the request data to JSON format
        headers=headers,  # Include the headers with the API key
    )

    # Parse the response JSON and extract the generated content
    res = res.json()
    response = res["choices"][0]["message"]["content"]

    # Return the strategic recommendations generated by the model
    return response


# Streamlit UI Section

# Sidebar header to guide the user to select a product
st.sidebar.header("❄️Select a Product❄️")


# Function to retrieve the list of products from the competitor data CSV file
def get_product_list():
    try:
        # Read competitor data from a CSV file
        competitor_df = pd.read_csv("competitor_data.csv")
        # Return a list of unique product names
        return competitor_df["product_name"].drop_duplicates().tolist()
    except FileNotFoundError:
        # If the CSV file is not found, return an empty list
        return []


# Get the list of products to display in the Streamlit sidebar
products = get_product_list()

# Create a dropdown in the sidebar to allow the user to select a product
selected_product = st.sidebar.selectbox("Choose a product to analyze:", products)

# Load competitor and review data
competitor_data = load_competitor_data()
reviews_data = load_reviews_data()

# Filter the competitor data to get the data for the selected product
product_data = competitor_data[competitor_data["product_name"] == selected_product]

# Filter the review data to get the reviews for the selected product
product_reviews = reviews_data[reviews_data["product_name"] == selected_product]

# Display the competitor analysis heading and the table with the last 5 rows of competitor data
st.header(f"Competitor Analysis for {selected_product}")
st.subheader("Competitor Data")
st.table(product_data.tail(5))

# If product reviews exist, process the reviews and display sentiment analysis
if not product_reviews.empty:
    # Truncate reviews to 512 characters
    product_reviews.loc[:, "review"] = product_reviews["review"].apply(lambda x: truncate_text(x, 512))

    # Convert the reviews to a list and perform sentiment analysis
    reviews = product_reviews["review"].tolist()
    sentiments = analyze_sentiment(reviews)

    # Display sentiment analysis results as a bar chart
    st.subheader("Customer Sentiment Analysis")
    sentiment_df = pd.DataFrame(sentiments)
    fig = px.bar(sentiment_df, x="label", title="Sentiment Analysis Results")
    st.plotly_chart(fig)
else:
    # If there are no reviews, display a message
    st.write("No reviews available for this product.")

# Convert the 'date' column to datetime format and handle missing values
product_data["date"] = pd.to_datetime(product_data["date"], errors="coerce")

# Set the index of the product data to a range of dates starting from the minimum date
product_data.index = pd.date_range(start=product_data.index.min(), periods=len(product_data), freq="D")

# Convert the 'discount' column to numeric values and drop rows with NaN values in this column
product_data["discount"] = pd.to_numeric(product_data["discount"], errors="coerce")
product_data = product_data.dropna(subset=["discount"])

# Apply the ARIMA forecasting model to predict future discounts
product_data_with_predictions = forecast_discounts_arima(product_data)

# Display the predicted discounts in the Streamlit app
st.subheader("Competitor Current and Predicted Discounts")
st.table(product_data_with_predictions[["Predicted_Discount"]].tail(10))

# Generate strategic recommendations for the selected product based on competitor data and sentiment
recommendations = generate_strategy_recommendation(
    selected_product,
    product_data_with_predictions,
    sentiments if not product_reviews.empty else "No reviews available",
)

# Display the strategic recommendations
st.subheader("Strategic Recommendations")
st.write(recommendations)

# Send the strategic recommendations to Slack for further action
send_to_slack(recommendations)