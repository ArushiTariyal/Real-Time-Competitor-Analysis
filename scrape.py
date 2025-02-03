# Import necessary libraries
import json  # For working with JSON data, parsing and generating JSON format
import time  # Provides time-related functions like sleep (used for delaying execution)
from datetime import datetime  # For working with date and time (e.g., getting the current time)
import pandas as pd  # Powerful library for data manipulation and analysis (e.g., working with DataFrames)
import requests  # For making HTTP requests (though not actively used in the provided code)
import plotly.express as px  # Used for creating interactive plots (though not actively used in the provided code)
import streamlit as st  # Used for creating web applications (though not actively used in the provided code)
from selenium import webdriver  # For automating web browser interaction, useful in web scraping
from selenium.webdriver.chrome.options import Options  # Used to configure options for the Chrome browser in Selenium
from selenium.webdriver.chrome.service import Service  # Manages Chrome driver service
from selenium.webdriver.common.by import By  # For finding elements in the web page (via different locator strategies like By.XPATH, By.CLASS_NAME, etc.)
from selenium.webdriver.support import expected_conditions as EC  # For waiting until certain conditions are met (e.g., an element becomes visible)
from selenium.webdriver.support.wait import WebDriverWait  # Used for explicit waiting for elements in Selenium
from webdriver_manager.chrome import ChromeDriverManager  # Automatically downloads and installs the correct version of ChromeDriver for Selenium
import chromedriver_autoinstaller  # Automatically installs the correct ChromeDriver (if necessary)
from chromedriver_autoinstaller import install  # Function from chromedriver_autoinstaller to install the driver
from transformers import pipeline  # For using pre-trained models, especially in NLP tasks (not used in the provided code)
from sklearn.ensemble import RandomForestRegressor  # For using Random Forest regression models (not used in the provided code)
from sklearn.model_selection import train_test_split  # For splitting datasets into training and testing sets (not used in the provided code)
from statsmodels.tsa.arima.model import ARIMA  # For time series forecasting using ARIMA model (not used in the provided code)
from selenium.webdriver.chrome.service import Service  # Duplicate import (already imported above, can be removed)


links={"Apple Iphone 13 (128 GB)":"https://www.amazon.in/Apple-iPhone-13-128GB-Green/dp/B09V4B6K53/ref=sr_1_1_sspa?crid=2XWF6OQBE9MW2&dib=eyJ2IjoiMSJ9.4Amcm6ymShwYf2cUNy6g87ZAmr160niWSMsGfJ6ktkhVvBfKClhwZifyFoyaaxp3p9CgrK4JD0kka6vg2gnarqoOb62duNBPCD13Tp0i69vRDmk4uzfDB-25bgoJNhIMNFEoNjBAjmfxVst_C0QmW8zulZt3XeCwXmXb04f26KHMlZ8v3WYOdj3IywjwNuQ1kRaqWcGGKYG5719prdWaQTuqcco0NBNjnzPCNlPyH_Y.GrzT8mZU2IyaErRyD0CZZeRLmD9_fnsrr95RqbZorhw&dib_tag=se&keywords=iphone&qid=1737998659&sprefix=iphone%2Caps%2C238&sr=8-1-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1",
    }
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

def get_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Automatically install the chromedriver version that matches the chromium version
    chromedriver_autoinstaller.install()

    # Create the webdriver with the options and use the default path
    driver = webdriver.Chrome(options=chrome_options)
    return driver


def scrape_product_data(link):
    # Get the web driver for Selenium (headless browser)
    driver = get_driver()

    # Set the window size for the browser window
    driver.set_window_size(1920, 1080)

    # Open the product page using the provided link
    driver.get(link)

    # Initialize a dictionary to store product data
    product_data = {
        "product_name": "",  # Placeholder for the product name
        "selling price": 0,  # Placeholder for the selling price
        "original price": 0,  # Placeholder for the original price
        "discount": 0,  # Placeholder for the discount value (if any)
        "rating": 0,  # Placeholder for product rating
        "reviews": [],  # Placeholder for product reviews (list)
        "product_url": link,  # Store the product URL
    }

    # Retry mechanism in case of failure while loading page
    retry = 0
    while retry < 3:
        try:
            # Take a screenshot to capture the current state of the page
            driver.save_screenshot("screenshot.png")

            # Wait until the price element is present on the page
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "a-offscreen")))

            break  # Break the loop once the page loads correctly
        except Exception as e:
            # Retry on failure (up to 3 attempts)
            print(f"Retrying... Error: {e}")
            retry += 1
            driver.get(link)  # Reload the page
            time.sleep(5)  # Wait before retrying

    # Extract the selling price of the product
    try:
        price_elem = driver.find_element(
            By.XPATH, '//*[@id="corePriceDisplay_desktop_feature_div"]/div[1]/span[3]/span[2]/span[2]'
        )
        # Clean and convert the price text to integer
        product_data["selling price"] = int("".join(price_elem.text.strip().split(",")))
    except Exception as e:
        print(f"Error extracting selling price: {e}")

    # Extract the original price of the product (if available)
    try:
        original_price = driver.find_element(
            By.XPATH, '//*[@id="corePriceDisplay_desktop_feature_div"]/div[2]/span/span[1]/span[2]/span/span[2]'
        ).text
        # Clean and convert the original price text to integer
        product_data["original price"] = int("".join(original_price.strip().split(",")))
    except Exception as e:
        print(f"Error extracting original price: {e}")

    # Extract discount or rating information
    try:
        discount = driver.find_element(
            By.XPATH, '//*[@id="corePriceDisplay_desktop_feature_div"]/div[1]/span[2]'
        )
        full_rating_text = discount.get_attribute("innerHTML").strip()

        # If the text contains "out of 5 stars", it's the rating; otherwise, it's the discount
        if " out of 5 stars" in full_rating_text.lower():
            product_data["rating"] = full_rating_text.lower().split(" out of")[0].strip()
        else:
            product_data["discount"] = full_rating_text
    except Exception as e:
        print(f"Error extracting discount: {e}")

    # Try to click on the rating popover to load reviews (if any)
    try:
        driver.find_element(By.CLASS_NAME, "a-icon-popover").click()
        time.sleep(1)
    except Exception as e:
        print(f"Error clicking rating popover: {e}")

    # Extract customer reviews
    try:
        reviews_link = driver.find_elements(
            By.XPATH, "//a[contains(text(), 'See customer reviews')]"
        )[-1].get_attribute("href")

        # Update product URL to the reviews link
        product_data["product_url"] = reviews_link.split("#")[0]

        driver.get(reviews_link)  # Open the reviews page
        time.sleep(3)  # Wait for the reviews to load

        reviews = driver.find_element(By.ID, "cm-cr-dp-review-list")
        reviews = reviews.find_elements(By.TAG_NAME, "li")

        # Collect each review and store it in the 'reviews' list
        for item in reviews:
            product_data["reviews"].append(item.get_attribute("innerText"))

        driver.back()  # Go back to the product page
        time.sleep(3)  # Wait before finishing
    except Exception as e:
        print(f"Error extracting reviews: {e}")

    # Close the web driver after finishing the scraping
    driver.quit()

    # Return the extracted product data
    return product_data


# Import necessary libraries
import pandas as pd  # For handling data in DataFrame format and saving to CSV
from datetime import datetime  # For getting the current date
import re  # For regular expressions to extract ratings from review text

# Function to extract rating from review text using regular expressions
def extract_rating_from_review(review_text):
    # Use regex to match a pattern like '4.5 out of 5 stars'
    match = re.search(r"(\d+\.\d+) out of 5 stars", review_text)
    # If a match is found, return the extracted rating as float
    if match:
        return float(match.group(1))
    return None  # If no match is found, return None

# Iterate through the links dictionary containing product names and URLs
for product_name, link in links.items():
    # Scrape product data using the scrape_product_data function
    product_data = scrape_product_data(link)

    # Try to read the reviews.csv file into a DataFrame, if exists
    try:
        reviews_df = pd.read_csv("reviews.csv")
    except FileNotFoundError:
        # If file not found, create an empty DataFrame with necessary columns
        reviews_df = pd.DataFrame(columns=["product_name", "review", "rating", "date"])

    new_reviews = []
    # Iterate through the reviews of the current product
    for review_text in product_data["reviews"]:
        # Extract rating from review text using the extract_rating_from_review function
        rating = extract_rating_from_review(review_text)
        # Append the review details (product_name, review, rating, and date) to the list
        new_reviews.append({
            "product_name": product_name,
            "review": review_text,
            "rating": rating,
            "date": datetime.now().strftime("%Y-%m-%d")  # Current date
        })

    # Convert the list of new reviews into a DataFrame
    new_reviews_df = pd.DataFrame(new_reviews)
    # Append the new reviews DataFrame to the existing reviews DataFrame
    reviews_df = pd.concat([reviews_df, new_reviews_df], ignore_index=True)
    # Save the updated reviews DataFrame to 'reviews.csv'
    reviews_df.to_csv("reviews.csv", index=False)

    # Try to read the competitor_data.csv file into a DataFrame
    try:
        competitor_df = pd.read_csv("competitor_data.csv")
    except FileNotFoundError:
        # If file not found, create an empty DataFrame with necessary columns
        competitor_df = pd.DataFrame(columns=["product_name", "price", "discount", "date"])

    # Create a new dictionary with product details for competitor data
    new_data = {
        "product_name": product_name,
        "price": product_data["selling price"],  # Extracted selling price
        "discount": product_data["discount"],    # Extracted discount
        "date": datetime.now().strftime("%Y-%m-%d")  # Current date
    }

    # Convert the new data dictionary into a DataFrame
    new_data_df = pd.DataFrame([new_data])
    # Append the new data DataFrame to the existing competitor DataFrame
    competitor_df = pd.concat([competitor_df, new_data_df], ignore_index=True)
    # Save the updated competitor DataFrame to 'competitor_data.csv'
    competitor_df.to_csv("competitor_data.csv", index=False)

# API keys
API_KEY = "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"  # Groq API Key
SLACK_WEBHOOK = "https://hooks.slack.com/services/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"  # Slack webhook URL
# Streamlit app setup
st.set_page_config(layout="wide")
# Create two columns
col1, col2 = st.columns(2)

# Add content to the first column
with col1:
    # Add a styled markdown text with a heading to the first column
    st.markdown(
        """
        <div style="font-size: 40px; text-align: left; width: 100%;">
            ❄️❄️❄️<strong>E-Commerce Competitor Strategy Dashboard</strong>❄️❄️❄️
        </div>
        """,
        unsafe_allow_html=True,  # Allow HTML content to be rendered
    )

# Add GIF to the second column
with col2:
    # Add an engaging GIF with right-alignment to the second column
    st.markdown(
        """
        <div style="text-align: right;">
            <img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExbzh4dXpuc2hpY3JlNnR1MDdiMXozMXlreHFoZjl0a2g5anJqNWxtMCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/hWe6YajFuxX41eV8I0/giphy.gif" alt="Engaging GIF" width="300">
        </div>
        """,
        unsafe_allow_html=True,  # Allow HTML content to be rendered
    )

# Utility function to truncate text
def truncate_text(text, max_length=512):
    """
    Truncate the input text to the specified maximum length.
    Args:
        text: The text to be truncated.
        max_length: The maximum allowed length of the text. Default is 512.
    Returns:
        Truncated text.
    """
    return text[:max_length]  # Return the truncated text

# Load competitor data
def load_competitor_data():
    """
    Load competitor data from a CSV file and display the first few rows.
    Returns:
        DataFrame: Loaded competitor data.
    """
    data = pd.read_csv("competitor_data.csv")  # Read the CSV file into a DataFrame
    st.write(data.head())  # Display the first 5 rows of the data for debugging purposes
    return data  # Return the loaded data

# Load reviews data
def load_reviews_data():
    """
    Load reviews data from a CSV file.
    Returns:
        DataFrame: Loaded reviews data.
    """
    reviews = pd.read_csv("reviews.csv")  # Read the reviews data from a CSV file
    return reviews  # Return the loaded reviews data

# Analyze customer sentiment
def analyze_sentiment(reviews):
    """
    Analyze the sentiment of customer reviews using a sentiment analysis pipeline.
    Args:
        reviews: List of reviews to analyze.
    Returns:
        List of sentiment analysis results for each review.
    """
    sentiment_pipeline = pipeline("sentiment-analysis")  # Initialize the sentiment analysis pipeline
    return sentiment_pipeline(reviews)  # Return the sentiment analysis results

# Train predictive model
def train_predictive_model(data):
    """
    Train a predictive model to forecast competitor pricing strategy based on price and discount data.
    Args:
        data: DataFrame containing competitor data including price and discount.
    Returns:
        model: A trained RandomForestRegressor model.
    """
    data["Discount"] = data["Discount"].str.replace("%", "").astype(float)  # Clean and convert discount to float
    data["Price"] = data["Price"].astype(float)  # Convert price to float
    data["Predicted_Discount"] = data["Discount"] + (data["Price"] * 0.05).round(2)  # Calculate a predicted discount

    X = data[["Price", "Discount"]]  # Select features for the model (Price and Discount)
    y = data["Predicted_Discount"]  # Select the target variable (Predicted_Discount)

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)  # Initialize a Random Forest Regressor model
    model.fit(X_train, y_train)  # Train the model using the training data
    return model  # Return the trained model


# Forecast discounts using ARIMA
def forecast_discounts_arima(data, future_days=5):
    """
    Forecast future discounts using ARIMA.
    :param data: DataFrame containing historical discount data (with a datetime index).
    :param future_days: Number of days to forecast.
    :return: DataFrame with historical and forecasted discounts.
    """
    # Ensure the data is sorted by index (date)
    data = data.sort_index()

    # Convert the 'Discount' column to numeric values, coerce errors to NaN
    data["Discount"] = pd.to_numeric(data["Discount"], errors="coerce")

    # Drop rows where the 'Discount' column is NaN
    data = data.dropna(subset=["Discount"])

    # Extract the 'Discount' column as a time series
    discount_series = data["Discount"]

    # Check if the DataFrame index is of type DatetimeIndex, and convert if necessary
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            # Attempt to convert the index to datetime
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            # Raise an error if the conversion fails
            raise ValueError("Index must be datetime or convertible to datetime.") from e

    # Define the ARIMA model with the order (p=5, d=1, q=0)
    model = ARIMA(discount_series, order=(5, 1, 0))

    # Fit the ARIMA model to the discount data
    model_fit = model.fit()

    # Forecast future discounts for the specified number of days
    forecast = model_fit.forecast(steps=future_days)

    # Generate future dates starting from the day after the last historical data point
    future_dates = pd.date_range(
        start=discount_series.index[-1] + pd.Timedelta(days=1),  # Next day after the last data point
        periods=future_days  # Forecast for the specified number of days
    )

    # Create a DataFrame with the future dates and forecasted discounts
    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Discount": forecast})

    # Set the 'Date' column as the index of the forecast DataFrame
    forecast_df.set_index("Date", inplace=True)

    # Return the forecasted discount data
    return forecast_df


# Send notifications to Slack
def send_to_slack(data):
    """
    Send the given data as a notification to Slack.
    :param data: Data to send to Slack (usually a string).
    """
    payload = {"text": data}  # Prepare the payload with the message text

    # Send a POST request to the Slack webhook URL with the payload
    response = requests.post(
        SLACK_WEBHOOK,  # The Slack webhook URL
        data=json.dumps(payload),  # Convert the payload to JSON format
        headers={"Content-Type": "application/json"}  # Set the content type header to JSON
    )

    # Check if the request was successful (HTTP status code 200)
    if response.status_code != 200:
        # Display an error message if the notification failed to send
        st.write(f"Failed to send notification to Slack: {response.status_code}")


# Generate strategy recommendations using an LLM
def generate_strategy_recommendation(product_name, competitor_data, sentiment):
    """Generate strategic recommendations using an LLM."""
    date = datetime.now()  # Get the current date and time

    # Construct the prompt for the language model
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

    # Data structure for sending the request to the API
    data = {
        "messages": [{"role": "user", "content": prompt}],  # Message content with user input (prompt)
        "model": "llama3-8b-8192",  # Specify the model to use
        "temperature": 0,  # Set the randomness of the model's response (0 means more deterministic)
    }

    # Headers including the API key for authorization
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

    # Send a POST request to the API for generating the strategy recommendations
    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",  # API endpoint for chat completions
        data=json.dumps(data),  # Convert data to JSON format for the request
        headers=headers,  # Include headers with authorization and content-type
    )

    # Parse the response JSON to extract the content of the generated message
    res = res.json()
    response = res["choices"][0]["message"]["content"]  # Extracting the LLM response from the API result

    return response  # Return the strategy recommendation response


# Streamlit UI

# Sidebar for selecting a product to analyze
st.sidebar.header("❄️Select a Product❄️")
products = [
    "Apple AirPods Pro (2nd Generation)",  # Product 1
    "Sony WH-1000XM4 Wireless Noise Cancelling Headphones",  # Product 2
    "Samsung Galaxy Buds2 Pro",  # Product 3
    "Jabra Elite 85t True Wireless Earbuds"  # Product 4
]

# User selects a product from the sidebar dropdown
selected_product = st.sidebar.selectbox("Choose a product to analyze:", products)

# Load competitor data and reviews data
competitor_data = load_competitor_data()  # Load competitor pricing and discount data
reviews_data = load_reviews_data()  # Load customer reviews data

# Filter competitor data for the selected product
product_data = competitor_data[competitor_data["product_name"] == selected_product]

# Filter reviews data for the selected product
product_reviews = reviews_data[reviews_data["product_name"] == selected_product]

# Display header for the selected product analysis
st.header(f"Competitor Analysis for {selected_product}")

# Display the competitor data for the selected product
st.subheader("Competitor Data")
st.table(product_data.tail(5))  # Show the last 5 rows of competitor data

# If there are reviews available for the selected product
if not product_reviews.empty:
    # Truncate reviews to a maximum length of 512 characters
    product_reviews["reviews"] = product_reviews["reviews"].apply(
        lambda x: truncate_text(x, 512)
    )
    reviews = product_reviews["reviews"].tolist()  # Convert reviews into a list of text
    sentiments = analyze_sentiment(reviews)  # Analyze the sentiment of the reviews

    # Display sentiment analysis results
    st.subheader("Customer Sentiment Analysis")
    sentiment_df = pd.DataFrame(sentiments)  # Convert sentiment results into a DataFrame
    fig = px.bar(sentiment_df, x="label",
                 title="Sentiment Analysis Results")  # Create a bar chart for sentiment analysis
    st.plotly_chart(fig)  # Display the bar chart in the Streamlit app
else:
    # If no reviews are available, display a message
    st.write("No reviews available for this product.")

# Preprocess competitor data for time series analysis
product_data["Date"] = pd.to_datetime(product_data["Date"], errors="coerce")  # Convert 'Date' to datetime format
product_data = product_data.dropna(subset=["Date"])  # Drop rows with missing 'Date'
product_data.set_index("Date", inplace=True)  # Set 'Date' as the index of the DataFrame

# Convert 'Discount' to numeric and handle errors (coerce invalid values)
product_data["Discount"] = pd.to_numeric(product_data["Discount"], errors="coerce")

# Drop rows with missing 'Discount' values
product_data = product_data.dropna(subset=["Discount"])

# Forecasting Model
product_data_with_predictions = forecast_discounts_arima(product_data)  # Forecast future discounts using the ARIMA model

# Display the most recent competitor data along with predicted discounts
st.subheader("Competitor Current and Predicted Discounts")
st.table(product_data_with_predictions.tail(10))  # Show the last 10 rows of the data, including predictions

# Generate strategic recommendations based on the competitor data and sentiment analysis
recommendations = generate_strategy_recommendation(
    selected_product,  # Pass the selected product's name
    product_data_with_predictions,  # Pass the competitor data with predicted discounts
    sentiments if not product_reviews.empty else "No reviews available",  # Use sentiment analysis or a fallback message if no reviews exist
)

# Display the generated strategic recommendations in the Streamlit app
st.subheader("Strategic Recommendations")
st.write(recommendations)  # Output the generated recommendations to the app interface

# Send the strategic recommendations to Slack for notification
send_to_slack(recommendations)  # Send the recommendations to Slack via a webhook
