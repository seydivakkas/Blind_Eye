from playwright.sync_api import sync_playwright

tabs = ["overview", "pipeline", "training", "analysis", "academic", "clinical", "deploy"]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto("http://localhost:8080")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(2000)

    for tab in tabs:
        page.click(f'button[data-tab="{tab}"]')
        page.wait_for_timeout(800)
        page.screenshot(path=f"screenshot_{tab}.png", full_page=False)
        print(f"Captured: {tab}")

    browser.close()
    print("All screenshots captured successfully")
