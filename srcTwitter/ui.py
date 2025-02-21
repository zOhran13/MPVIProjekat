from nicegui import ui
from TwitterService import predict_profile
from TwitterService import login_to_twitter
import threading

from nicegui import ui
from TwitterService import predict_profile
import threading
import asyncio

async def process_prediction():
    username = username_input.value.strip()
    if not username:
        result_html.set_content("<b style='color:red;'>Please enter a username.</b>")
        return

    result_html.set_content("<b style='color:blue;'>Processing...</b>")

    async def run():
        try:
            await login_to_twitter()
            result = await predict_profile(username)
            profile = result["profile_data"]
            fake_probability = result["fake_probability"]
            output = (
                      f"<b>Followers:</b> {profile.get('followers_count', 'N/A')}<br>"
                      f"<b>Following:</b> {profile.get('friends_count', 'N/A')}<br>"
                      f"<b>Biography Length:</b> {profile.get('description_length', '')}<br>"
                      f"<b>Username Digit Count:</b> {profile.get('username_digit_count', '0')}<br>"
                      f"<b>Username Length:</b> {profile.get('name_length', '')}<br>"
                      f"<b>Account age:</b> {profile.get('account_age', '')} days<br>"
                      f"<b>Real profile Probability:</b> {fake_probability:.2f}%")

            result_html.set_content(output)
        except Exception as e:
            result_html.set_content(f"<b style='color:red;'>Error:</b> {str(e)}")

    thread = threading.Thread(target=lambda: asyncio.run(run()))
    thread.start()

# UI Elements
ui.label("Twitter Profile Analysis").classes("text-lg font-bold")
username_input = ui.input("Enter Twitter Username").classes("w-64")
ui.button("Submit", on_click=process_prediction).classes("bg-blue-500 text-white px-4 py-2 rounded")
result_html = ui.html("Result will appear here.").classes("mt-4 text-lg")

ui.run(title="Twitter Analysis", port=8080)
