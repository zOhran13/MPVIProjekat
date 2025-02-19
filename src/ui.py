from nicegui import ui
from be import predict_profile
import threading

def process_prediction():
    username = username_input.value.strip()
    if not username:
        result_html.set_content("<b style='color:red;'>Please enter a username.</b>")
        return

    result_html.set_content("<b style='color:blue;'>Processing...</b>")

    def run():
        try:
            result = predict_profile(username)
            profile = result["profile_data"]
            fake_probability = result["fake_probability"]

            output = (f"<b>Username:</b> {profile['username']}<br>"
                      f"<b>Followers:</b> {profile['followers']}<br>"
                      f"<b>Following:</b> {profile['followees']}<br>"
                      f"<b>Biography Length:</b> {profile['biography_length']}<br>"
                      f"<b>Media Count:</b> {profile['mediacount']}<br>"
                      f"<b>Has Profile Picture:</b> {'Yes' if profile['has_profile_pic'] else 'No'}<br>"
                      f"<b>Is Private:</b> {'Yes' if profile['is_private'] else 'No'}<br>"
                      f"<b>Username Digit Count:</b> {profile['username_digit_count']}<br>"
                      f"<b>Username Length:</b> {profile['username_length']}<br>"
                      f"<b>Fake Probability:</b> {fake_probability}%")

            result_html.set_content(output)
        except Exception as e:
            result_html.set_content(f"<b style='color:red;'>Error:</b> {str(e)}")

    thread = threading.Thread(target=run)
    thread.start()

ui.label("Instagram Profile Analysis").classes("text-lg font-bold")
username_input = ui.input("Enter Instagram Username").classes("w-64")
ui.button("Submit", on_click=process_prediction).classes("bg-blue-500 text-white px-4 py-2 rounded")
result_html = ui.html("Result will appear here.").classes("mt-4 text-lg")

ui.run(title="Instagram Analysis", port=8080)
