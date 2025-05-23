# 🧠 Enhancing Web Scraping with Selenium (CostcoInsider Coupon Scraper)

## 🚀 Why Use Selenium?

The current scraper uses `requests` to fetch HTML via `get_page()`, which works for **static** pages but fails on **JavaScript-heavy** pages.

The issue:  
The `find_all_coupon_links()` function often returns empty results because `https://www.costcoinsider.com/category/coupons/` loads content **dynamically with JavaScript** after the initial load.

✅ **Solution: Use Selenium**  
Selenium renders the entire page including JavaScript, providing accurate HTML for parsing with BeautifulSoup.

---

## 🧩 Integration Plan

### 1. Install Selenium and Set Up WebDriver

#### ✅ Install Selenium

```bash
pip install selenium

✅ Download ChromeDriver (Manual Option)
Check your Chrome version (visit chrome://version in Chrome).

Go to: https://sites.google.com/chromium.org/driver/

Download the matching ChromeDriver version.

Extract and place the chromedriver.exe in a known directory or add it to your system PATH.

🔁 Optional: Use webdriver-manager (Recommended)
bash
Copy
Edit
pip install webdriver-manager
Let Selenium automatically manage your ChromeDriver — no manual downloads required.

