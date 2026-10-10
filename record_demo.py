"""
Automated Video Demonstration Recorder for Gnani.ai Chatbot.
Uses Playwright to record a high-definition video walkthrough of the Streamlit application.
"""

import os
import shutil
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

ARTIFACT_DIR = Path(r"C:\Users\vamseeswara reddy\.gemini\antigravity-ide\brain\42c227f6-ca9d-4386-909a-6b57dc3bbf8b")
VIDEO_TEMP_DIR = ARTIFACT_DIR / "temp_video"
VIDEO_TEMP_DIR.mkdir(parents=True, exist_ok=True)


def record_walkthrough():
    print("Starting Playwright Chromium browser recorder...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            record_video_dir=str(VIDEO_TEMP_DIR),
            record_video_size={"width": 1280, "height": 800},
            viewport={"width": 1280, "height": 800},
        )
        page = context.new_page()

        print("Navigating to Streamlit app at http://localhost:8501...")
        page.goto("http://localhost:8501", timeout=30000)

        # Wait for Streamlit app to render
        page.wait_for_selector(".gnani-header", timeout=20000)
        time.sleep(3)

        print("Highlighting sidebar and headers...")
        page.mouse.move(200, 300)
        time.sleep(2)

        # Click on Suggested Question: "What is Gnani.ai?"
        print("Clicking suggested question: 'What is Gnani.ai?'...")
        sugg_button = page.locator("button:has-text('What is Gnani.ai?')").first
        if sugg_button.is_visible():
            sugg_button.click()
        else:
            page.locator("button").first.click()

        # Wait for response generation
        time.sleep(6)

        # Scroll down smoothly to show answer and sources
        print("Scrolling down to reveal grounded answer and clickable sources...")
        for y in range(0, 500, 50):
            page.mouse.wheel(0, 50)
            time.sleep(0.1)
        time.sleep(3)

        # Type in the chat input
        print("Typing question: 'What is Armour365?'...")
        chat_input = page.locator("textarea[data-testid='stChatInputTextArea'], input[data-testid='stChatInputTextArea']").first
        if chat_input.is_visible():
            chat_input.click()
            chat_input.fill("What is Armour365?")
            time.sleep(1)
            page.keyboard.press("Enter")
            print("Submitted question, waiting for response...")
            time.sleep(6)

        # Final smooth overview scroll
        print("Capturing full conversation...")
        for y in range(0, 400, 50):
            page.mouse.wheel(0, 50)
            time.sleep(0.1)
        time.sleep(3)

        page.close()
        context.close()
        browser.close()

    print("Browser closed. Locating recorded video...")
    video_files = list(VIDEO_TEMP_DIR.glob("*.webm"))
    if video_files:
        latest_video = max(video_files, key=os.path.getctime)
        target_dest = ARTIFACT_DIR / "gnani_chatbot_demo.webm"
        shutil.copy(str(latest_video), str(target_dest))
        print(f"Video saved to: {target_dest} (Size: {target_dest.stat().st_size:,} bytes)")
        return str(target_dest)
    else:
        print("No video file found in temp dir.")
        return None


if __name__ == "__main__":
    record_walkthrough()
