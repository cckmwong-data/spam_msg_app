import requests

# Replace YOUR_STREAMLIT_APP_URL with your real app URL
url = "https://spammsgcheck.streamlit.app/"

try:
    response = requests.get(url)
    if response.status_code == 200:
        print(f"Successfully pinged {url}")
    else:
        print(f"Failed to ping {url}. Status code: {response.status_code}")
except Exception as e:
    print(f"Error pinging {url}: {e}")