# Import necessary libraries
import json  # For working with JSON data
import time  # For time-related operations
from datetime import datetime  # For handling date and time
import pandas as pd  # For data manipulation and analysis
import requests  # For making HTTP requests
import plotly.express as px  # For creating interactive visualizations
import streamlit as st  # For building web applications
from selenium import webdriver  # For browser automation
from selenium.webdriver.chrome.options import Options  # For configuring Chrome options
from selenium.webdriver.chrome.service import Service  # For managing ChromeDriver service
from selenium.webdriver.common.by import By  # For locating elements on a webpage
from selenium.webdriver.support import expected_conditions as EC  # For explicit waits
from selenium.webdriver.support.wait import WebDriverWait  # For waiting for conditions
from webdriver_manager.chrome import ChromeDriverManager  # For managing ChromeDriver installation
import chromedriver_autoinstaller  # For automatically installing ChromeDriver
from transformers import pipeline  # For using pre-trained NLP models
from sklearn.ensemble import RandomForestRegressor  # For Random Forest regression
from sklearn.model_selection import train_test_split  # For splitting datasets into training and testing sets
from statsmodels.tsa.arima.model import ARIMA  # For ARIMA time series modeling
from selenium.webdriver.chrome.service import Service  # For managing ChromeDriver service (duplicate import)

# Dictionary containing product names as keys and their Amazon URLs as values
links = {
    "Apple iPhone 13 (128GB) - Green": "https://www.amazon.in/Apple-iPhone-13-128GB-Green/dp/B09V4B6K53/ref=sr_1_1_sspa?crid=2XWF6OQBE9MW2&dib=eyJ2IjoiMSJ9.4Amcm6ymShwYf2cUNy6g87ZAmr160niWSMsGfJ6ktkhVvBfKClhwZifyFoyaaxp3p9CgrK4JD0kka6vg2gnarqoOb62duNBPCD13Tp0i69vRDmk4uzfDB-25bgoJNhIMNFEoNjBAjmfxVst_C0QmW8zulZt3XeCwXmXb04f26KHMlZ8v3WYOdj3IywjwNuQ1kRaqWcGGKYG5719prdWaQTuqcco0NBNjnzPCNlPyH_Y.GrzT8mZU2IyaErRyD0CZZeRLmD9_fnsrr95RqbZorhw&dib_tag=se&keywords=iphone&qid=1737998659&sprefix=iphone%2Caps%2C238&sr=8-1-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1",
}

# Import chromedriver_autoinstaller again (duplicate import)
import chromedriver_autoinstaller
# Import Selenium webdriver and related modules again (duplicate imports)
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# Function to initialize and configure the Selenium WebDriver
def get_driver():
    # Create an instance of Chrome Options to configure ChromeDriver
    chrome_options = Options()
    # Run Chrome in headless mode (no GUI)
    chrome_options.add_argument("--headless")
    # Disable the sandbox for security (often used in CI/CD environments)
    chrome_options.add_argument("--no-sandbox")
    # Disable shared memory usage to avoid issues in Docker or limited environments
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Automatically install the ChromeDriver version that matches the installed Chrome version
    chromedriver_autoinstaller.install()

    # Create a WebDriver instance with the configured options
    driver = webdriver.Chrome(options=chrome_options)
    # Return the configured WebDriver instance
    return driver

# Function to scrape product data from a given Amazon product link
def scrape_product_data(link):
    # Initialize the Selenium WebDriver
    driver = get_driver()
    # Set the browser window size to 1920x1080 pixels
    driver.set_window_size(1920, 1080)
    # Navigate to the provided product link
    driver.get(link)

    # Dictionary to store scraped product data
    product_data = {
        "product_name": "",  # Placeholder for product name
        "selling price": 0,  # Placeholder for selling price
        "original price": 0,  # Placeholder for original price
        "discount": 0,  # Placeholder for discount
        "rating": 0,  # Placeholder for product rating
        "reviews": [],  # List to store customer reviews
        "product_url": link,  # Store the product URL
        "date": datetime.now().strftime("%Y-%m-%d"),  # Store the current date in YYYY-MM-DD format
    }

    # Retry mechanism to handle potential errors during scraping
    retry = 0
    while retry < 3:  # Retry up to 3 times
        try:
            # Take a screenshot of the current page (for debugging purposes)
            driver.save_screenshot("screenshot.png")
            # Wait for the presence of an element with the class "a-offscreen" (price element)
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "a-offscreen")))
            break  # Exit the retry loop if successful
        except Exception as e:
            # Print the error and retry
            print(f"Retrying... Error: {e}")
            retry += 1
            # Reload the page
            driver.get(link)
            # Wait for 5 seconds before retrying
            time.sleep(5)

    # Extract the selling price
    try:
        # Locate the selling price element using XPath
        price_elem = driver.find_element(
            By.XPATH, '//*[@id="corePriceDisplay_desktop_feature_div"]/div[1]/span[3]/span[2]/span[2]'
        )
        # Clean the price text (remove commas) and convert it to an integer
        product_data["selling price"] = int("".join(price_elem.text.strip().split(",")))
    except Exception as e:
        # Print an error message if the selling price cannot be extracted
        print(f"Error extracting selling price: {e}")

    # Extract the original price
    try:
        # Locate the original price element using XPath
        original_price = driver.find_element(
            By.XPATH, '//*[@id="corePriceDisplay_desktop_feature_div"]/div[2]/span/span[1]/span[2]/span/span[2]'
        ).text
        # Use a helper function (extract_price) to clean and convert the price
        product_data["original price"] = extract_price(original_price)
    except Exception as e:
        # Print an error message if the original price cannot be extracted
        print(f"Error extracting original price: {e}")

    # Extract the discount or rating (handles both cases)
    try:
        # Locate the discount or rating element using XPath
        discount = driver.find_element(
            By.XPATH, '//*[@id="corePriceDisplay_desktop_feature_div"]/div[1]/span[2]'
        )
        # Get the inner HTML of the element
        full_rating_text = discount.get_attribute("innerHTML").strip()
        # Check if the text contains a rating (e.g., "4.5 out of 5 stars")
        if " out of 5 stars" in full_rating_text.lower():
            # Extract the rating value
            product_data["rating"] = full_rating_text.lower().split(" out of")[0].strip()
        else:
            # If not a rating, assume it's a discount and store it
            product_data["discount"] = full_rating_text
    except Exception as e:
        # Print an error message if the discount or rating cannot be extracted
        print(f"Error extracting discount: {e}")

    # Extract the product rating (alternative method)
    try:
        # Locate the rating element using its class name
        rating_elem = driver.find_element(By.CLASS_NAME, "a-icon-star")
        # Get the inner text of the rating element
        product_data["rating"] = rating_elem.get_attribute("innerText").strip()
    except Exception as e:
        # Print an error message if the rating cannot be extracted
        print(f"Error extracting rating: {e}")

    # Extract customer reviews
    try:
        # Locate all "See customer reviews" links using XPath
        reviews_link_elements = driver.find_elements(
            By.XPATH, "//a[contains(text(), 'See customer reviews')]"
        )
        if reviews_link_elements:
            # Get the URL of the last "See customer reviews" link
            reviews_link = reviews_link_elements[-1].get_attribute("href")
            # Navigate to the reviews page
            driver.get(reviews_link)
            # Wait for 3 seconds to allow the page to load
            time.sleep(3)

            # Locate the reviews section using its ID
            reviews_section = driver.find_element(By.ID, "cm-cr-dp-review-list")
            # Locate all review elements within the reviews section
            review_elements = reviews_section.find_elements(By.TAG_NAME, "li")

            # Loop through each review element and append its text to the reviews list
            for review in review_elements:
                product_data["reviews"].append(review.text.strip())
        else:
            # Print a message if no customer reviews are found
            print("No customer reviews found.")
    except Exception as e:
        # Print an error message if the reviews cannot be extracted
        print(f"Error extracting reviews: {e}")

    # Close the WebDriver
    driver.quit()
    # Return the scraped product data
    return product_data


# Import the regular expressions module (used for text manipulation)
import re

# Function to extract and clean price from a string containing currency symbols or commas
def extract_price(price_text):
    """Extracts and converts price from a string with currency symbols or commas."""
    # Use regex to remove all non-digit characters (e.g., ₹, commas, etc.)
    price_text = re.sub(r"[^\d]", "", price_text)
    # Convert the cleaned string to an integer, or return 0 if the string is empty
    return int(price_text) if price_text else 0


# Function to extract the rating from a review text
def extract_rating_from_review(review_text):
    # Use regex to search for a pattern like "4.5 out of 5 stars"
    match = re.search(r"(\d+\.\d+) out of 5 stars", review_text)
    if match:
        # If a match is found, extract the rating and convert it to a float
        return float(match.group(1))
    # Return None if no rating is found
    return None


# Loop through each product name and its corresponding link in the `links` dictionary
for product_name, link in links.items():
    # Scrape product data using the `scrape_product_data` function
    product_data = scrape_product_data(link)

    # Update reviews.csv
    try:
        # Try to load the existing reviews CSV file into a DataFrame
        reviews_df = pd.read_csv("reviews.csv")
    except FileNotFoundError:
        # If the file doesn't exist, create a new DataFrame with the specified columns
        reviews_df = pd.DataFrame(columns=["product_name", "review", "rating", "date"])

    # Create a list to store new reviews
    new_reviews = []
    # Loop through each review text in the scraped reviews
    for review_text in product_data["reviews"]:
        # Extract the rating from the review text
        rating = extract_rating_from_review(review_text)
        # Append the review data to the new_reviews list
        new_reviews.append({
            "product_name": product_name,  # Product name
            "review": review_text,  # Review text
            "rating": rating,  # Extracted rating
            "date": datetime.now().strftime("%Y-%m-%d")  # Current date
        })

    # Convert the new_reviews list into a DataFrame
    new_reviews_df = pd.DataFrame(new_reviews)
    # Concatenate the existing reviews DataFrame with the new reviews DataFrame
    reviews_df = pd.concat([reviews_df, new_reviews_df], ignore_index=True)
    # Save the updated DataFrame back to the reviews CSV file
    reviews_df.to_csv("reviews.csv", index=False)

    # Update competitor_data.csv
    # The following block is commented out, but it shows an alternative approach
    # try:
    #     competitor_df = pd.read_csv("competitor_data.csv")
    # except FileNotFoundError:
    #     competitor_df = pd.DataFrame(columns=["product_name", "price", "discount", "date"])
    #
    # new_data = {
    #     "product_name": product_name,
    #     "price": product_data["selling price"],
    #     "discount": product_data["discount"],
    #     "date": datetime.now().strftime("%Y-%m-%d")
    # }
    #
    # new_data_df = pd.DataFrame([new_data])
    # competitor_df = pd.concat([competitor_df, new_data_df], ignore_index=True)
    # competitor_df.to_csv("competitor_data.csv", index=False)

    # Alternative approach for updating competitor_data.csv
    try:
        # Try to load the existing competitor data CSV file into a DataFrame
        competitor_df = pd.read_csv("competitor_data.csv")

        # Drop extra columns if they exist, keeping only the specified columns
        competitor_df = competitor_df[['product_name', 'price', 'discount', 'date']]
    except FileNotFoundError:
        # If the file doesn't exist, create a new DataFrame with the specified columns
        competitor_df = pd.DataFrame(columns=["product_name", "price", "discount", "date"])

    # Create a new data entry for the current product
    new_data = {
        "product_name": product_name,  # Product name
        "price": product_data["selling price"],  # Selling price
        "discount": product_data["discount"],  # Discount
        "date": datetime.now().strftime("%Y-%m-%d"),  # Current date
    }

    # Convert the new_data dictionary into a DataFrame
    # Ensure the DataFrame has the correct column order by explicitly specifying the columns
    new_data_df = pd.DataFrame([new_data], columns=["product_name", "price", "discount", "date"])

    # Append the new data to the existing competitor DataFrame
    # Use `pd.concat` to combine the DataFrames and reset the index to avoid duplicate indices
    competitor_df = pd.concat([competitor_df, new_data_df], ignore_index=True)

    # Save the updated DataFrame to the competitor_data.csv file
    # The `index=False` argument ensures that the DataFrame index is not written to the file
    competitor_df.to_csv("competitor_data.csv", index=False)

# API keys and configuration
API_KEY = "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"
# This is an API key for accessing the Groq API, which is likely used to interact with a machine learning or AI service.
SLACK_WEBHOOK = "https://hooks.slack.com/services/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"
# This is a Slack webhook URL, which is used to send automated messages or notifications to a specific Slack channel.

# Streamlit app setup
st.set_page_config(layout="wide")
# This configures the Streamlit app to use a wide layout, which means the app will take up more horizontal space on the screen
# making it easier to display content side by side.

# Create two columns in the Streamlit app
col1, col2 = st.columns(2)  # This creates two columns in the Streamlit app interface.
# These columns can be used to organize and display content (e.g., text, charts, inputs) side by side for better readability and layout design.

# Add content to the first column (col1)
with col1:
    # Use Streamlit's markdown function to display styled HTML content
    st.markdown(
        """
        <div style="font-size: 40px; text-align: left; width: 100%;">
            ❄️❄️❄️<strong>E-Commerce Competitor Strategy Dashboard</strong>❄️❄️❄️
        </div>
        """,
        unsafe_allow_html=True,  # Allow HTML rendering for custom styling
    )
    # This displays a large, bold heading for the dashboard with snowflake emojis for decoration.

# Add a GIF to the second column (col2)
with col2:
    # Use Streamlit's markdown function to display an image (GIF) aligned to the right
    st.markdown(
        """
        <div style="text-align: right;">
            <img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExbzh4dXpuc2hpY3JlNnR1MDdiMXozMXlreHFoZjl0a2g5anJqNWxtMCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/hWe6YajFuxX41eV8I0/giphy.gif" alt="Engaging GIF" width="300">
        </div>
        """,
        unsafe_allow_html=True,  # Allow HTML rendering for embedding the GIF
    )
    # This displays an engaging GIF on the right side of the dashboard to make the interface more visually appealing.


# Utility function to truncate text
def truncate_text(text, max_length=512):
    """Truncate text to a specified maximum length."""
    return text[:max_length]  # Return the first `max_length` characters of the text
    # This function is useful for limiting the length of text (e.g., reviews) to avoid overwhelming the UI.


# Load competitor data from a CSV file
def load_competitor_data():
    """Load competitor data from a CSV file."""
    data = pd.read_csv("competitor_data.csv")  # Read the CSV file into a DataFrame
    st.write(data.head())  # Display the first few rows of the data for debugging purposes
    return data  # Return the loaded data
    # This function loads competitor pricing and discount data for analysis.


# Load reviews data from a CSV file
def load_reviews_data():
    """Load reviews data from a CSV file."""
    reviews = pd.read_csv("reviews.csv")  # Read the CSV file into a DataFrame
    return reviews  # Return the loaded reviews
    # This function loads customer reviews for sentiment analysis.


# Analyze customer sentiment using a pre-trained NLP model
def analyze_sentiment(reviews):
    """Analyze customer sentiment for reviews."""
    sentiment_pipeline = pipeline("sentiment-analysis")  # Load a sentiment analysis pipeline
    return sentiment_pipeline(reviews)  # Analyze the sentiment of the reviews and return the results
    # This function uses Hugging Face's `transformers` library to determine if reviews are positive, negative, or neutral.

# Train a predictive model for competitor pricing strategy
def train_predictive_model(data):
    """Train a predictive model for competitor pricing strategy."""
    # Clean the "Discount" column by removing the percentage sign and converting to float
    data["Discount"] = data["Discount"].str.replace("%", "").astype(float)
    # Convert the "Price" column to float
    data["Price"] = data["Price"].astype(float)
    # Create a new column "Predicted_Discount" by adding 5% of the price to the current discount
    data["Predicted_Discount"] = data["Discount"] + (data["Price"] * 0.05).round(2)

    # Define features (X) and target (y) for the model
    X = data[["Price", "Discount"]]  # Features: Price and Discount
    y = data["Predicted_Discount"]  # Target: Predicted_Discount

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize a Random Forest Regressor model
    model = RandomForestRegressor(random_state=42)
    # Train the model on the training data
    model.fit(X_train, y_train)
    # Return the trained model
    return model


# Forecast future discounts using the ARIMA model
def forecast_discounts_arima(data, future_days=5):
    """
    Forecast future discounts using ARIMA.
    :param data: DataFrame containing historical discount data (with a datetime index).
    :param future_days: Number of days to forecast.
    :return: DataFrame with historical and forecasted discounts.
    """
    # Sort the data by its index (assumed to be dates)
    data = data.sort_index()
    # Convert the "discount" column to numeric, handling errors by coercing invalid values to NaN
    data["discount"] = pd.to_numeric(data["discount"], errors="coerce")
    # Drop rows with NaN values in the "discount" column
    data = data.dropna(subset=["discount"])

    # Extract the discount series for modeling
    discount_series = data["discount"]

    # Ensure the index is a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            # Convert the index to a DatetimeIndex
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            # Raise an error if the index cannot be converted to datetime
            raise ValueError("Index must be datetime or convertible to datetime.") from e

    # Initialize an ARIMA model with order (5, 1, 0)
    model = ARIMA(discount_series, order=(5, 1, 0))
    # Fit the ARIMA model to the data
    model_fit = model.fit()

    # Forecast future discounts for the specified number of days
    forecast = model_fit.forecast(steps=future_days)
    # Generate future dates for the forecasted values
    future_dates = pd.date_range(
        start=discount_series.index[-1] + pd.Timedelta(days=1),  # Start from the day after the last date
        periods=future_days  # Number of days to forecast
    )

    # Create a DataFrame to store the forecasted discounts and their corresponding dates
    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Discount": forecast})
    # Set the "Date" column as the index
    forecast_df.set_index("Date", inplace=True)
    # Return the forecast DataFrame
    return forecast_df


# Send notifications to Slack
def send_to_slack(data):
    """Send a notification to Slack using a webhook."""
    # Create the payload with the text to send
    payload = {"text": data}
    # Send a POST request to the Slack webhook URL
    response = requests.post(
        SLACK_WEBHOOK,  # Slack webhook URL
        data=json.dumps(payload),  # Convert payload to JSON
        headers={"Content-Type": "application/json"}  # Set content type to JSON
    )
    # Check if the request was successful (status code 200)
    if response.status_code != 200:
        # Display an error message if the notification failed
        st.write(f"Failed to send notification to Slack: {response.status_code}")


# Generate strategy recommendations using a Large Language Model (LLM)
def generate_strategy_recommendation(product_name, competitor_data, sentiment):
    """Generate strategic recommendations using an LLM."""
    # Get the current date
    date = datetime.now()
    # Create a detailed prompt for the LLM
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

    # Prepare the data for the API request
    data = {
        "messages": [{"role": "user", "content": prompt}],  # Include the prompt in the messages
        "model": "llama3-8b-8192",  # Specify the LLM model to use
        "temperature": 0,  # Set temperature to 0 for deterministic responses
    }

    # Set headers for the API request, including the API key for authentication
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    # Send a POST request to the Groq API
    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",  # API endpoint
        data=json.dumps(data),  # Convert data to JSON
        headers=headers,  # Include headers
    )
    # Parse the JSON response
    res = res.json()
    # Extract the generated response from the API
    response = res["choices"][0]["message"]["content"]
    # Return the generated response
    return response

# Streamlit UI

# Add a header to the sidebar for product selection
st.sidebar.header("❄️Select a Product❄️")


# Function to get a list of unique products from the competitor data
def get_product_list():
    try:
        # Load competitor data from the CSV file
        competitor_df = pd.read_csv("competitor_data.csv")
        # Extract unique product names and convert them to a list
        return competitor_df["product_name"].drop_duplicates().tolist()
    except FileNotFoundError:
        # Return an empty list if the file is not found
        return []


# Get the list of products
products = get_product_list()

# Add a dropdown in the sidebar to select a product
selected_product = st.sidebar.selectbox("Choose a product to analyze:", products)

# Load competitor and reviews data
competitor_data = load_competitor_data()
reviews_data = load_reviews_data()

# Filter data for the selected product
product_data = competitor_data[competitor_data["product_name"] == selected_product]
product_reviews = reviews_data[reviews_data["product_name"] == selected_product]

# Display the competitor analysis header
st.header(f"Competitor Analysis for {selected_product}")
# Display the competitor data section
st.subheader("Competitor Data")
# Show the last 5 rows of the competitor data in a table
st.table(product_data.tail(5))

# Check if there are reviews available for the selected product
if not product_reviews.empty:
    # Truncate review text to 512 characters for better display
    product_reviews.loc[:, "review"] = product_reviews["review"].apply(lambda x: truncate_text(x, 512))

    # Extract reviews and analyze their sentiment
    reviews = product_reviews["review"].tolist()
    sentiments = analyze_sentiment(reviews)

    # Display the sentiment analysis results
    st.subheader("Customer Sentiment Analysis")
    # Convert sentiment results to a DataFrame
    sentiment_df = pd.DataFrame(sentiments)
    # Create a bar chart to visualize sentiment distribution
    fig = px.bar(sentiment_df, x="label", title="Sentiment Analysis Results")
    # Display the chart in the Streamlit app
    st.plotly_chart(fig)
else:
    # Display a message if no reviews are available
    st.write("No reviews available for this product.")

# Prepare the data for forecasting
# Convert the "date" column to datetime format
product_data["date"] = pd.to_datetime(product_data["date"], errors="coerce")
# Set the index to a date range starting from the minimum date in the data
product_data.index = pd.date_range(start=product_data.index.min(), periods=len(product_data), freq="D")
# Convert the "discount" column to numeric, handling errors by coercing invalid values to NaN
product_data["discount"] = pd.to_numeric(product_data["discount"], errors="coerce")
# Drop rows with NaN values in the "discount" column
product_data = product_data.dropna(subset=["discount"])

# Forecasting Model
# Use the ARIMA model to forecast future discounts
product_data_with_predictions = forecast_discounts_arima(product_data)

# Display the current and predicted discounts
st.subheader("Competitor Current and Predicted Discounts")
# Show the last 10 rows of the predicted discounts in a table
st.table(product_data_with_predictions[["Predicted_Discount"]].tail(10))

# Generate strategic recommendations using the LLM
recommendations = generate_strategy_recommendation(
    selected_product,  # Selected product name
    product_data_with_predictions,  # Competitor data with predictions
    sentiments if not product_reviews.empty else "No reviews available",  # Sentiment analysis results
)

# Display the strategic recommendations
st.subheader("Strategic Recommendations")
st.write(recommendations)

# Send the recommendations to Slack
send_to_slack(recommendations)
