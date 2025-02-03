# Import necessary libraries
import sys
sys.path.append("C:\\Users\\arush\\AppData\\Roaming\\Python\\Python39\\Scripts")  # Add Python script path to sys.path
import json  # For handling JSON data
import time  # For managing time delays
from datetime import datetime  # For working with dates and times
import pandas as pd  # For working with data frames (e.g., saving CSV files)
import requests  # For sending HTTP requests
import plotly.express as px  # For data visualization
import streamlit as st  # For building web apps
from selenium import webdriver  # For automating web browser interaction
from selenium.webdriver.chrome.options import Options  # To set Chrome options (e.g., headless mode)
from selenium.webdriver.chrome.service import Service  # To handle service for ChromeDriver
from selenium.webdriver.common.by import By  # To locate elements on the page
from selenium.webdriver.support import expected_conditions as EC  # For waiting until conditions are met
from selenium.webdriver.support.wait import WebDriverWait  # To set explicit wait
from webdriver_manager.chrome import ChromeDriverManager  # To manage ChromeDriver installation
import chromedriver_autoinstaller  # Automatically installs the ChromeDriver if needed
from transformers import pipeline  # For using pre-trained models (likely for NLP)
from sklearn.ensemble import RandomForestRegressor  # For machine learning
from sklearn.model_selection import train_test_split  # For splitting data into train/test sets
from statsmodels.tsa.arima.model import ARIMA  # For time series forecasting
from selenium.webdriver.chrome.service import Service  # To manage ChromeDriver service
from selenium import webdriver  # For interacting with the browser
from selenium.webdriver.chrome.options import Options  # For setting Chrome options
from selenium.webdriver.chrome.service import Service  # For managing ChromeDriver service

# Links to product pages on Amazon
links = {
    "Apple iPhone 13 (128GB) - Green": "https://www.amazon.in/Apple-iPhone-13-128GB-Green/dp/B09V4B6K53/ref=sr_1_1_sspa?crid=2XWF6OQBE9MW2&dib=eyJ2IjoiMSJ9.4Amcm6ymShwYf2cUNy6g87ZAmr160niWSMsGfJ6ktkhVvBfKClhwZifyFoyaaxp3p9CgrK4JD0kka6vg2gnarqoOb62duNBPCD13Tp0i69vRDmk4uzfDB-25bgoJNhIMNFEoNjBAjmfxVst_C0QmW8zulZt3XeCwXmXb04f26KHMlZ8v3WYOdj3IywjwNuQ1kRaqWcGGKYG5719prdWaQTuqcco0NBNjnzPCNlPyH_Y.GrzT8mZU2IyaErRyD0CZZeRLmD9_fnsrr95RqbZorhw&dib_tag=se&keywords=iphone&qid=1737998659&sprefix=iphone%2Caps%2C238&sr=8-1-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1",
    "Apple iPhone 14 (128 GB) - Midnight": "https://www.amazon.in/Apple-iPhone-14-128GB-Midnight/dp/B0BDHX8Z63/ref=sr_1_2_sspa?crid=2XWF6OQBE9MW2&dib=eyJ2IjoiMSJ9.4Amcm6ymShwYf2cUNy6g87ZAmr160niWSMsGfJ6ktkhVvBfKClhwZifyFoyaaxp3p9CgrK4JD0kka6vg2gnarqoOb62duNBPCD13Tp0i69vRDmk4uzfDB-25bgoJNhIMNFEoNjBAjmfxVst_C0QmW8zulZt3XeCwXmXb04f26KHMlZ8v3WYOdj3IywjwNuQ1kRaqWcGGKYG5719prdWaQTuqcco0NBNjnzPCNlPyH_Y.GrzT8mZU2IyaErRyD0CZZeRLmD9_fnsrr95RqbZorhw&dib_tag=se&keywords=iphone&qid=1737998659&sprefix=iphone%2Caps%2C238&sr=8-2-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1",
    "Apple iPhone 15 (128 GB) - Blue": "https://www.amazon.in/Apple-iPhone-15-128-GB/dp/B0CHX2F5QT/ref=sr_1_3?crid=2XWF6OQBE9MW2&dib=eyJ2IjoiMSJ9.4Amcm6ymShwYf2cUNy6g87ZAmr160niWSMsGfJ6ktkhVvBfKClhwZifyFoyaaxp3p9CgrK4JD0kka6vg2gnarqoOb62duNBPCD13Tp0i69vRDmk4uzfDB-25bgoJNhIMNFEoNjBAjmfxVst_C0QmW8zulZt3XeCwXmXb04f26KHMlZ8v3WYOdj3IywjwNuQ1kRaqWcGGKYG5719prdWaQTuqcco0NBNjnzPCNlPyH_Y.GrzT8mZU2IyaErRyD0CZZeRLmD9_fnsrr95RqbZorhw&dib_tag=se&keywords=iphone&qid=1737998659&sprefix=iphone%2Caps%2C238&sr=8-3&th=1"
}

# Function to set up the Selenium WebDriver
def get_driver():
    chrome_options = Options()  # Initialize Chrome options
    chrome_options.add_argument("--headless")  # Run Chrome in headless mode (no GUI)
    chrome_options.add_argument("--no-sandbox")  # Disable sandboxing (required for some environments)
    chrome_options.add_argument("--disable-dev-shm-usage")  # Disable shared memory usage

    # Automatically installs the correct version of ChromeDriver
    chromedriver_autoinstaller.install()

    # Create and return the WebDriver instance with the specified options
    driver = webdriver.Chrome(options=chrome_options)
    return driver

# Function to scrape product data from a given Amazon link
def scrape_product_data(link):
    driver = get_driver()  # Initialize the WebDriver
    driver.set_window_size(1920, 1080)  # Set the window size
    driver.get(link)  # Open the product page
    
    # Initialize an empty dictionary to store product data
    product_data = {
        "product_name": "",  # Add product name here
        "selling price": 0,  # Selling price of the product
        "original price": 0,  # Original price before any discounts
        "discount": 0,  # Discount on the product
        "rating": 0,  # Product rating
        "reviews": [],  # List to hold reviews
        "product_url": link,  # Product URL
    }

    retry = 0  # Retry counter for error handling
    while retry < 3:  # Try 3 times before failing
        try:
            driver.save_screenshot("screenshot.png")  # Take a screenshot
            wait = WebDriverWait(driver, 10)  # Set up an explicit wait
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "a-offscreen")))  # Wait for price to appear
            break  # Exit loop if successful
        except Exception as e:  # Handle exceptions
            print(f"Retrying... Error: {e}")
            retry += 1
            driver.get(link)  # Reload the page
            time.sleep(5)  # Wait before retrying

    # Extracting selling price
    try:
        price_elem = driver.find_element(
            By.XPATH, '//*[@id="corePriceDisplay_desktop_feature_div"]/div[1]/span[3]/span[2]/span[2]'
        )
        product_data["selling price"] = int("".join(price_elem.text.strip().split(",")))  # Convert price to integer
    except Exception as e:
        print(f"Error extracting selling price: {e}")  # Handle exception

    # Extracting original price
    try:
        original_price = driver.find_element(
            By.XPATH, '//*[@id="corePriceDisplay_desktop_feature_div"]/div[2]/span/span[1]/span[2]/span/span[2]'
        ).text
        product_data["original price"] = int("".join(original_price.strip().split(",")))  # Convert price to integer
    except Exception as e:
        print(f"Error extracting original price: {e}")

    # Extracting discount
    try:
        discount = driver.find_element(
            By.XPATH, '//*[@id="corePriceDisplay_desktop_feature_div"]/div[1]/span[2]'
        )
        full_rating_text = discount.get_attribute("innerHTML").strip()  # Get the text of the discount
        if " out of 5 stars" in full_rating_text.lower():  # Check if it's a rating
            product_data["rating"] = full_rating_text.lower().split(" out of")[0].strip()  # Extract rating
        else:
            product_data["discount"] = full_rating_text  # Otherwise, it's the discount
    except Exception as e:
        print(f"Error extracting discount: {e}")

    # Clicking on the rating popover to view reviews
    try:
        driver.find_element(By.CLASS_NAME, "a-icon-popover").click()  # Click the rating popover
        time.sleep(1)  # Wait for the reviews to load
    except Exception as e:
        print(f"Error clicking rating popover: {e}")

    # Extracting reviews
    try:
        reviews_link = driver.find_elements(
            By.XPATH, "//a[contains(text(), 'See customer reviews')]"
        )[-1].get_attribute("href")  # Get the link to customer reviews
        product_data["product_url"] = reviews_link.split("#")[0]  # Set the reviews URL
        driver.get(reviews_link)  # Open the reviews page
        time.sleep(3)  # Wait for reviews to load
        reviews = driver.find_element(By.ID, "cm-cr-dp-review-list")  # Find the review list
        reviews = reviews.find_elements(By.TAG_NAME, "li")  # Get all review items
        for item in reviews:  # Loop through each review and add to product data
            product_data["reviews"].append(item.get_attribute("innerText"))
        driver.back()  # Go back to the product page
        time.sleep(3)  # Wait before continuing
    except Exception as e:
        print(f"Error extracting reviews: {e}")

    driver.quit()  # Close the WebDriver
    return product_data  # Return the scraped data

# Loop through each product and scrape data
for product_name, link in links.items():
    product_data = scrape_product_data(link)  # Scrape product data
    reviews = json.loads(pd.read_csv("reviews.csv").to_json(orient="records"))  # Load existing reviews data
    price = json.loads(pd.read_csv("competitor_data.csv").to_json(orient="records"))  # Load existing price data
    price.append(  # Add new price data to the list
        {
            "product_name": product_name,  # Use the product name
            "Price": product_data["selling price"],  # Add selling price
            "Discount": product_data["discount"],  # Add discount
            "Date": datetime.now().strftime("%d-%m-%y"),  # Add the current date
        }
    )
    for i in product_data["reviews"]:  # Loop through each review and add to the reviews list
        reviews.append({"product_name": product_name, "reviews": i})
    
    # Save updated reviews and price data back to CSV
    pd.DataFrame(reviews).to_csv("reviews.csv", index=False)  # Save reviews to CSV
    pd.DataFrame(price).to_csv("competitor_data.csv", index=False)  # Save prices to CSV

# API keys
API_KEY = "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"  # Groq API Key
SLACK_WEBHOOK = "xoxe.xoxp-1-Mi0yLTgzNjMxNDY1MTEwMjgtODM3MzMxODc4NzI5Ny04Mzg1NTc0Mjg4ODUxLTgzODgxODkwNzUxMjQtOWVlODU0MzVhOWJiZjk3ZTAzM2JkNzdkNjVhNjE2MTViOTM3ZWRjMzc3MGRiYjI3ZDQ0MzhmM2FhNzNlYjkyZA"  # Slack webhook URL

# Streamlit app setup
st.set_page_config(layout="wide")  # Set the page layout to wide for better display

# Create two columns
col1, col2 = st.columns(2)  # Create two columns for content arrangement

# Add content to the first column
with col1:
     st.markdown(
        """
        <div style="font-size: 40px; text-align: left; width: 100%;">
            ❄️❄️❄️<strong>E-Commerce Competitor Strategy Dashboard</strong>❄️❄️❄️
        </div>
        """,
        unsafe_allow_html=True,  # Allow HTML content inside markdown
    )

# Add GIF to the second column
with col2:
    st.markdown(
        """
        <div style="text-align: right;">
            <img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExbzh4dXpuc2hpY3JlNnR1MDdiMXozMXlreHFoZjl0a2g5anJqNWxtMCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/hWe6YajFuxX41eV8I0/giphy.gif" alt="Engaging GIF" width="300">
        </div>
        """,
        unsafe_allow_html=True,  # Allow HTML content and display a GIF in the second column
    )

# Utility function to truncate text
def truncate_text(text, max_length=512):
    return text[:max_length]  # Return the text truncated to the specified max length

# Load competitor data
def load_competitor_data():
    """Load competitor data from a CSV file."""
    data = pd.read_csv("competitor_data.csv")  # Read the competitor data from a CSV file
    st.write(data.head())  # Display first few rows of the data for debugging
    return data  # Return the loaded data

# Load reviews data
def load_reviews_data():
    """Load reviews data from a CSV file."""
    reviews = pd.read_csv("reviews.csv")  # Read the reviews data from a CSV file
    return reviews  # Return the reviews data

# Analyze customer sentiment
def analyze_sentiment(reviews):
    """Analyze customer sentiment for reviews."""
    sentiment_pipeline = pipeline("sentiment-analysis")  # Initialize the sentiment analysis pipeline
    return sentiment_pipeline(reviews)  # Run the sentiment analysis on the reviews

# Train predictive model
def train_predictive_model(data):
    """Train a predictive model for competitor pricing strategy."""
    data["Discount"] = data["Discount"].str.replace("%", "").astype(float)  # Clean and convert discount to numeric
    data["Price"] = data["Price"].astype(float)  # Convert price to numeric
    data["Predicted_Discount"] = data["Discount"] + (data["Price"] * 0.05).round(2)  # Calculate predicted discount based on price

    X = data[["Price", "Discount"]]  # Features for the model
    y = data["Predicted_Discount"]  # Target variable for the model

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data into train and test sets

    model = RandomForestRegressor(random_state=42)  # Initialize a Random Forest model
    model.fit(X_train, y_train)  # Train the model on the training data
    return model  # Return the trained model

# Forecast discounts using ARIMA
def forecast_discounts_arima(data, future_days=5):
    """
    Forecast future discounts using ARIMA.
    :param data: DataFrame containing historical discount data (with a datetime index).
    :param future_days: Number of days to forecast.
    :return: DataFrame with historical and forecasted discounts.
    """
    data = data.sort_index()  # Ensure data is sorted by the date index.
    data["Discount"] = pd.to_numeric(data["Discount"], errors="coerce")  # Convert Discount to numeric, handle errors.
    data = data.dropna(subset=["Discount"])  # Drop rows where Discount is NaN.

    discount_series = data["Discount"]  # Extract the discount series from the data.

    if not isinstance(data.index, pd.DatetimeIndex):  # Check if index is a datetime object.
        try:
            data.index = pd.to_datetime(data.index)  # Convert the index to datetime if possible.
        except Exception as e:
            raise ValueError("Index must be datetime or convertible to datetime.") from e

    model = ARIMA(discount_series, order=(5, 1, 0))  # Initialize ARIMA model with specified order.
    model_fit = model.fit()  # Fit the ARIMA model on the discount data.

    forecast = model_fit.forecast(steps=future_days)  # Forecast the future discounts.
    future_dates = pd.date_range(
        start=discount_series.index[-1] + pd.Timedelta(days=1),  # Generate future dates for forecasting.
        periods=future_days
    )

    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Discount": forecast})  # Store the forecast results.
    forecast_df.set_index("Date", inplace=True)  # Set the Date column as index.
    return forecast_df

# Send notifications to Slack
def send_to_slack(data):
    payload = {"text": data}  # Create payload with message content.
    response = requests.post(
        SLACK_WEBHOOK,  # Send data to Slack via the webhook URL.
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"}
    )
    if response.status_code != 200:  # Check if notification failed.
        st.write(f"Failed to send notification to Slack: {response.status_code}")

# Generate strategy recommendations using an LLM
def generate_strategy_recommendation(product_name, competitor_data, sentiment):
    """Generate strategic recommendations using an LLM."""
    date = datetime.now()  # Get current date.
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

    data = {
        "messages": [{"role": "user", "content": prompt}],  # Prepare the prompt for the LLM.
        "model": "llama3-8b-8192",  # Specify model for generation.
        "temperature": 0,  # Set temperature to 0 for deterministic results.
    }

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",  # Send request to Groq API.
        data=json.dumps(data),
        headers=headers,
    )
    res = res.json()  # Get the response from the API.
    response = res["choices"][0]["message"]["content"]  # Extract the recommendation content from the response.
    return response

# Streamlit UI

st.sidebar.header("❄️Select a Product❄️")

products = [
    "Apple iPhone 15",
    "Apple 2023 MacBook Pro",
    "OnePlus Nord 4 5G",
    "Sony WH-1000XM5"
]
selected_product = st.sidebar.selectbox("Choose a product to analyze:", products)  # Create a dropdown for product selection.

competitor_data = load_competitor_data()  # Load competitor data.
reviews_data = load_reviews_data()  # Load reviews data.

product_data = competitor_data[competitor_data["product_name"] == selected_product]  # Filter competitor data for selected product.
product_reviews = reviews_data[reviews_data["product_name"] == selected_product]  # Filter reviews data for selected product.

st.header(f"Competitor Analysis for {selected_product}")  # Display product name on the dashboard.
st.subheader("Competitor Data")  # Display competitor data section.
st.table(product_data.tail(5))  # Show the latest 5 rows of competitor data.

if not product_reviews.empty:
    product_reviews["reviews"] = product_reviews["reviews"].apply(
        lambda x: truncate_text(x, 512)  # Truncate reviews to 512 characters.
    )
    reviews = product_reviews["reviews"].tolist()  # Get the list of reviews.
    sentiments = analyze_sentiment(reviews)  # Analyze sentiment of reviews.

    st.subheader("Customer Sentiment Analysis")  # Display sentiment analysis section.
    sentiment_df = pd.DataFrame(sentiments)  # Convert sentiment data to DataFrame.
    fig = px.bar(sentiment_df, x="label", title="Sentiment Analysis Results")  # Plot sentiment results as a bar chart.
    st.plotly_chart(fig)  # Display the sentiment chart.
else:
    st.write("No reviews available for this product.")  # Handle case where no reviews are available.

product_data["Date"] = pd.to_datetime(product_data["Date"], errors="coerce")  # Convert Date column to datetime.
product_data = product_data.dropna(subset=["Date"])  # Drop rows where Date is NaN.
product_data.set_index("Date", inplace=True)  # Set Date column as the index.
product_data["Discount"] = pd.to_numeric(product_data["Discount"], errors="coerce")  # Convert Discount to numeric.
product_data = product_data.dropna(subset=["Discount"])  # Drop rows where Discount is NaN.

# Forecasting Model
product_data_with_predictions = forecast_discounts_arima(product_data)  # Forecast future discounts using ARIMA.

st.subheader("Competitor Current and Predicted Discounts")  # Display forecasted discounts section.
st.table(product_data_with_predictions.tail(10))  # Show the last 10 rows of forecasted data.

recommendations = generate_strategy_recommendation(
    selected_product,
    product_data_with_predictions,
    sentiments if not product_reviews.empty else "No reviews available",  # Pass sentiment data if available.
)

st.subheader("Strategic Recommendations")  # Display strategic recommendations.
st.write(recommendations)  # Show the recommendations on the dashboard.

send_to_slack(recommendations)  # Send recommendations to Slack for team review.
