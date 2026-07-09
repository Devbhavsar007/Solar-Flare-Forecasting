import os
from playwright.sync_api import sync_playwright
import time
from dotenv import load_dotenv

load_dotenv()

USERNAME = os.getenv("PRADAN_USERNAME")
PASSWORD = os.getenv("PRADAN_PASSWORD")

def explore():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            ignore_https_errors=True,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        
        print("Navigating to PRADAN...")
        page.goto("https://pradan.issdc.gov.in/pradan/")
        page.wait_for_load_state("networkidle")
        
        os.makedirs("artifacts_temp", exist_ok=True)
        
        # Check if we are on the login page (Keycloak)
        if "idp.issdc.gov.in" in page.url:
            print(f"Redirected to login: {page.url}")
            page.fill("#username", USERNAME)
            page.fill("#password", PASSWORD)
            page.click("input[type='submit'], button[type='submit'], #kc-login")
            page.wait_for_load_state("networkidle")
            print(f"URL after login: {page.url}")
        
        # Dump HTML
        with open("artifacts_temp/pradan_home.html", "w", encoding="utf-8") as f:
            f.write(page.content())
            
        print("Done. Check artifacts_temp/ folder.")
        browser.close()

if __name__ == "__main__":
    explore()
