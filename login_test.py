import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Initialize the WebDriver (make sure you have the right driver installed, e.g., ChromeDriver)
driver = webdriver.Chrome()

try:
    # Step 1: Open the login page
    driver.get("https://practicetestautomation.com/practice-test-login/")  # Replace with your login page URL
    driver.maximize_window()

    # Step 2: Locate username and password fields
    #username_field = driver.find_element(By.ID, "username")  # Adjust locator as needed
    #password_field = driver.find_element(By.ID, "password")
    
    username_field = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.ID, "username"))
)
    password_field = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.ID, "password"))
)

    # Step 3: Enter credentials
    username_field.send_keys("student")
    password_field.send_keys("Password123")

    # Step 4: Submit the form
    #login_button = driver.find_element(By.ID, "loginBtn")  # Adjust locator as needed
    login_button = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.ID, "submit"))
)
    login_button.click()

    # Step 5: Wait for page to load
    time.sleep(3)
    
finally:
    # Close the browser
    driver.quit()