# import json
# import time
# from datetime import datetime

# import pandas as pd
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.support.wait import WebDriverWait
# from webdriver_manager.chrome import ChromeDriverManager

# links = {
#     "Apple iPhone 15": "https://www.amazon.in/dp/B0CHX3TW6X?ref=ods_ucc_kindle_B0CHX2WQLX&th=1",
#     "Apple 2023 MacBook Pro (16-inch, Apple M3 Pro chip with 12‑core CPU and 18‑core GPU, 36GB Unified Memory, 512GB) - Silver": "https://amzn.in/d/ib419CQ",
#     "OnePlus Nord 4 5G (Mercurial Silver, 8GB RAM, 256GB Storage)": "https://amzn.in/d/2KOJBxa",
#     "Sony WH-1000XM5 Best Active Noise Cancelling Wireless Bluetooth Over Ear Headphones with Mic for Clear Calling, up to 40 Hours Battery -Black": "https://amzn.in/d/dP5ATPJ",
# }


# def scrape_product_data(link):
#     options = Options()
#     options.add_argument("--headless")
#     options.add_argument("--no-sandbox")
#     options.add_argument("--disable-dev-shm-usage")
#     options.add_argument("--disable-gpu")
#     options.add_argument("--lang=en")
#     options.add_argument("--window-size=1920,1080")

#     driver = webdriver.Chrome(
#         service=Service(ChromeDriverManager().install()), options=options
#     )
#     driver.set_window_size(1920, 1080)
#     driver.get(link)
#     product_data, review_data = {}, {}
#     product_data["reviews"] = []
#     wait = WebDriverWait(driver, 10)
#     time.sleep(5)
#     retry = 0
#     while retry < 3:
#         try:
#             driver.save_screenshot("screenshot.png")
#             wait.until(EC.presence_of_element_located((By.CLASS_NAME, "a-offscreen")))
#             break
#         except Exception:
#             print("retrying")
#             retry += 1
#             driver.get(link)
#             time.sleep(5)

#     driver.save_screenshot("screenshot.png")
#     try:
#         price_elem = driver.find_element(
#             By.XPATH,
#             '//*[@id="corePriceDisplay_desktop_feature_div"]/div[1]/span[3]/span[2]/span[2]',
#         )
#         product_data["selling price"] = int("".join(price_elem.text.strip().split(",")))
#     except:
#         product_data["selling price"] = 0

#     try:
#         original_price = driver.find_element(
#             By.XPATH,
#             '//*[@id="corePriceDisplay_desktop_feature_div"]/div[2]/span/span[1]/span[2]/span/span[2]',
#         ).text
#         product_data["original price"] = int("".join(original_price.strip().split(",")))
#     except:
#         product_data["original price"] = 0

#     try:
#         discount = driver.find_element(
#             By.XPATH,
#             '//*[@id="corePriceDisplay_desktop_feature_div"]/div[1]/span[2]',
#         )
#         full_rating_text = discount.get_attribute("innerHTML").strip()
#         if " out of 5 stars" in full_rating_text.lower():
#             product_data["rating"] = (
#                 full_rating_text.lower().split(" out of")[0].strip()
#             )
#         else:
#             product_data["discount"] = full_rating_text
#     except:
#         product_data["discount"] = 0

#     try:
#         driver.find_element(By.CLASS_NAME, "a-icon-popover").click()
#         time.sleep(1)
#     except:
#         pass

#     try:
#         reviews_link = driver.find_elements(
#             By.XPATH, "//a[contains(text(),'See customer reviews')]"
#         )[-1].get_attribute("href")
#         product_data["product_url"] = reviews_link.split("#")[0]
#         driver.get(reviews_link)
#         time.sleep(3)
#         reviews = driver.find_element(By.ID, "cm-cr-dp-review-list")
#         reviews = reviews.find_elements(By.TAG_NAME, "li")
#         for item in reviews:
#             product_data["reviews"].append(item.get_attribute("innerText"))

#         driver.back()
#         time.sleep(3)
#     except Exception:
#         product_data["reviews"] = []

#     product_data["date"] = time.strftime("%Y-%m-%d")
#     review_data["date"] = time.strftime("%Y-%m-%d")
#     driver.quit()
#     return product_data


# for product_name, link in links.items():
#     product_data = scrape_product_data(link)
#     reviews = json.loads(pd.read_csv("reviews.csv").to_json(orient="records"))
#     price = json.loads(pd.read_csv("competitor_data.csv").to_json(orient="records"))
#     price.append(
#         {
#             "product_name": product_name,
#             "Price": product_data["selling price"],
#             "Discount": product_data["discount"],
#             "Date": datetime.now().strftime("%Y-%m-%d"),
#         }
#     )
#     for i in product_data["reviews"]:
#         reviews.append({"product_name": product_name, "reviews": i})

#     pd.DataFrame(reviews).to_csv("reviews.csv", index=False)
#     pd.DataFrame(price).to_csv("competitor_data.csv", index=False)

import requests  # Library for making HTTP requests to fetch web pages
from bs4 import BeautifulSoup  # Library for parsing HTML and XML documents
import pandas as pd  # Library for data manipulation and analysis
from datetime import datetime  # Library for handling date and time
import time  # Library for time-related functions (e.g., delays)
import random  # Library for generating random numbers


def scrape_product_data(url):
    """Scrape product data from e-commerce website."""

    # Define headers to mimic a real browser request
    #Headers for Web Requests: The headers dictionary is used to mimic a real browser request. This helps avoid being blocked by the website for scraping.
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # Send a GET request to the product URL
        response = requests.get(url, headers=headers)

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract product name
        product_name = soup.find('span', {'id': 'productTitle'}).text.strip()

        # Extract product price and convert it to a float (remove currency symbol and commas)
        price = float(soup.find('span', {'class': 'a-price-whole'}).text.replace('₹', '').replace(',', '').strip())

        # Extract discount percentage and convert it to a float (remove percentage symbol)
        discount = float(soup.find('span', {'class': "a-size-large a-color-price savingPriceOverride aok-align-center reinventPriceSavingsPercentageMargin savingsPercentage"}).text.replace('%', '').strip())

        # Get reviews if available
        reviews = []
        review_elements = soup.find_all('div', {'class': 'a-expander-content reviewText review-text-content a-expander-partial-collapse-content'})

        # Loop through the first 5 reviews and extract review text and rating
        for review in review_elements[:5]:  # Get latest 5 reviews for simplicity
            review_text = review.find('p').text.strip()
            rating = float(review.find('span', {'class': 'rating'}).text.strip())

            # Append each review to the reviews list
            reviews.append({
                'review_text': review_text,
                'rating': rating,
                'date': datetime.now().strftime('%Y-%m-%d')  # Add current date to the review
            })

        # Return the scraped product data and reviews
        return {
            'product_data': {
                'product_name': product_name,
                'price': price,
                'discount': discount,
                'date': datetime.now().strftime('%Y-%m-%d')  # Add current date to the product data
            },
            'reviews': reviews
        }

    except Exception as e:
        # Print any errors that occur during scraping
        print(f"Error scraping data: {str(e)}")
        return None


def update_csv_files(product_data, reviews):
    """Update CSV files with new data."""

    try:
        # Convert product data to a DataFrame and append it to 'competitor_data.csv'
        #The update_csv_files function appends the scraped product data and reviews to existing CSV files (competitor_data.csv and reviews.csv).
        # The mode='a' parameter in to_csv ensures that new data is appended to the existing files without overwriting them.

        df_products = pd.DataFrame([product_data])
        df_products.to_csv('competitor_data.csv', mode='a', header=False, index=False)

        # Convert reviews data to a DataFrame and append it to 'reviews.csv'
        if reviews:
            df_reviews = pd.DataFrame(reviews)
            df_reviews.to_csv('reviews.csv', mode='a', header=False, index=False)

        return True
    except Exception as e:
        # Print any errors that occur during CSV file updates
        print(f"Error updating CSV files: {str(e)}")
        return False


def main():
    # List of competitor URLs to monitor
    competitor_urls = [
        'https://www.amazon.in/Apple-AirPods-Generation-MagSafe-USB%E2%80%91C/dp/B0CHX719JD/ref=sr_1_3?crid=98YV2CHWS3P6&dib=eyJ2IjoiMSJ9.eAsYHuN12gq3JlC28Fid-130szldDdEcc_yNkk6ksXOdUeT_QK0qt2rfbIvKpcP4uj2Zjtg-OjVVgxiFUfA4rtNT0Tz-VQTa8udIJMZDXpBTxvRCWcQl2KV2jHqGg7DXEf_deHjoDmIa2Lnm-BOnnkt5Dj6r59vqdbH-gw8GjZNmgDOq6ETeswvPfzS2kFy3CH01d2FHjhHZ_p4Zm1GNsFeQJ1WzL5AmApX_pSdenDg.Nj4Au_NQPT2usO3DyJQzOLdEo1s1UxHeL_PiKg_-bPg&dib_tag=se&keywords=apple+airpods+2nd+generation&nsdOptOutParam=true&qid=1738388720&sprefix=apple+airpods+%2Caps%2C250&sr=8-3'
        'https://www.amazon.in/Bose-QuietComfort-Cancelling-World-Class-Cancellation/dp/B0CD2FSRDD/ref=sr_1_4?crid=3K56VNLUIVADS&dib=eyJ2IjoiMSJ9.gdiXMxJTGVgPQI_U2BdWauUYa3wflRhI8xADHIIxbbKA-i5SkGSziLTwGsg5CfUEaTyPB3qcxUJKMTz2TQ1-27-tH79iVmdiG-w7QuqFPlnaEXlxFtLND_-kBW3SYVhD6D1lu_w6TLePal491t5-eKR9PsvaMjH6eLYyb8sITydjkDrNyb0IDEhvsjKNADsTmrRccl3b_sizeGNbOIKzIQmYjwChKusGHEty5rYAi6s.U_KOdn9S9GpbrYca2V7AnTUMQS0CBbnnt84Azw7NTmA&dib_tag=se&keywords=bose%2Bquiet%2Bcomfort%2B2&nsdOptOutParam=true&qid=1738388787&sprefix=bose%2Bquiet%2Bcomfort%2B2%2Caps%2C286&sr=8-4&th=1'
        # We can add more URLs as needed
    ]

    # Infinite loop to continuously monitor the competitor URLs
    #The main function contains an infinite loop that continuously monitors the competitor URLs.
    # It scrapes data from each URL, updates the CSV files, and then waits for a random delay (2-5 seconds) before moving to the next URL.
    # After completing one cycle of scraping all URLs, it waits for 1 hour before starting the next cycle.

    while True:
        for url in competitor_urls:
            # Scrape product data from the current URL
            data = scrape_product_data(url)

            # If data is successfully scraped, update the CSV files
            if data:
                update_csv_files(data['product_data'], data['reviews'])

            # Add a random delay between requests (2-5 seconds) to avoid being blocked and detected as a bot by the website.
            time.sleep(random.uniform(2, 5))

        # Wait for 1 hour before the next update cycle
        time.sleep(3600)


if __name__ == "__main__":
    # Run the main function when the script is executed
    main()