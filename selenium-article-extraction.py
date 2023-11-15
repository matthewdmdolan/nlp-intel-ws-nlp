from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

# Load your DataFrame
news_df = pd.read_csv('news_data.csv')  # Replace with your file path or DataFrame source

# Add a new column for the article body
news_df['Article Body'] = ''

# Setup Selenium WebDriver
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Uncomment if you don't need a browser UI
driver = webdriver.Chrome(options=options)  # Replace with the browser you're using
wait = WebDriverWait(driver, 10)  # Adjust the timeout as necessary

# CSS selector for the article body, update this to match the website you're scraping
article_body_selector = '#ctl00_ctl00_body_mainContent_NewsActicle_pnlBody'  # Update this selector

# Loop over the first 10 rows using .iloc
for index, row in news_df.iloc[:1000].iterrows():
    link = row['Link']
    if pd.notnull(link):
        try:
            # Navigate to the link
            driver.get(link)

            # Wait for the article body to load and become visible
            article_body_element = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, article_body_selector)))
            article_body_text = article_body_element.text  # Extract the text

            # Update the DataFrame with the article body
            news_df.at[index, 'Article Body'] = article_body_text
        except (NoSuchElementException, TimeoutException):
            print(f"Couldn't find the article body for the link: {link}")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print(f"No link available at index {index}")

# Save the DataFrame with the article bodies back to a CSV file for the first 10 rows
news_df.to_csv('news_article_text.csv', index=False)

# Close the Selenium WebDriver
driver.quit()
