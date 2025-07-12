from flask import Flask, render_template, request
import requests
import os
import joblib
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import urllib.request
import random

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

# === Download models from Google Drive ===
import gdown

def download_from_gdrive(file_id, dest_path):
    if not os.path.exists(dest_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {dest_path}...")
        gdown.download(url, dest_path, quiet=False)


# Replace these IDs with your actual Google Drive file IDs
download_from_gdrive("1KOCXpD-MQkagIjz58G_sLukv4WKo73Ep", "model/rain_classifier.pkl")
download_from_gdrive("1XGoC5oRjZj6hygzv4-kBYweohsdBymWj", "model/rain_regressor.pkl")
download_from_gdrive("1SmS4vWWEsqQ0O5Ty7r299FKEzX1BMVjY", "model/scaler.pkl")

# Load the models after download
clf = joblib.load("model/rain_classifier.pkl")
reg = joblib.load("model/rain_regressor.pkl")
scaler = joblib.load("model/scaler.pkl")

# === Flask app setup ===
app = Flask(__name__)

def get_advice(weather):
    sunny_tips = [
        "Wear light, breathable clothing.",
        "Use sunscreen to protect from UV rays.",
        "Wear sunglasses to shield your eyes.",
        "Stay hydrated—carry a water bottle.",
        "Plan outdoor activities like cycling or walking.",
        "Wear a hat or cap to protect your head.",
        "Avoid your curious tasks during peak sun (12–3 PM).",
        "Eat light meals to stay cool.",
        "Charge your devices—solar power works best today!",
        "Great day for a picnic or photoshoot!"
    ]

    rain_tips = [
        "Carry an umbrella or raincoat.",
        "Wear waterproof shoes or boots.",
        "Avoid slippery roads and surfaces.",
        "Check traffic before you travel.",
        "Stay indoors if lightning is expected.",
        "Protect electronics in waterproof bags.",
        "Enjoy hot beverages like tea or coffee.",
        "Read a book or watch a movie indoors.",
        "Avoid riding two-wheelers in heavy rain.",
        "Great time to journal or do indoor hobbies!"
    ]

    if weather == "rain":
        return random.choice(rain_tips)
    elif weather == "sunny":
        return random.choice(sunny_tips)
    else:
        return "Have a nice day!"

def cloud_description(cloud_percent):
    if cloud_percent <= 10:
        return "Clear sky"
    elif cloud_percent <= 30:
        return "Mostly clear"
    elif cloud_percent <= 60:
        return "Partly cloudy"
    elif cloud_percent <= 84:
        return "Mostly cloudy"
    else:
        return "Overcast"

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        city = request.form["city"]
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        res = requests.get(url).json()

        if res.get("cod") == 200:
            main = res["main"]
            wind = res["wind"]["speed"]
            clouds = res["clouds"]["all"]
            weather = res["weather"][0]["main"].lower()
            timestamp = res["dt"] + res["timezone"]
            dt = datetime.utcfromtimestamp(timestamp)

            features = pd.DataFrame([{
                "Humidity3pm": main["humidity"],
                "Pressure9am": main["pressure"],
                "Temp3pm": main["temp"],
                "Cloud3pm": clouds,
                "WindSpeed3pm": wind,
                "RainToday": 1 if "rain" in weather else 0
            }])
            scaled = scaler.transform(features)
            prob = clf.predict_proba(scaled)[0][1]
            amt = reg.predict(scaled)[0] if prob > 0.5 else 0

            cloud_desc = cloud_description(clouds)
            wind_kmph = wind * 3.6

            data = {
                "city": city.title(),
                "temp": f"{main['temp']}°C",
                "humidity": f"{main['humidity']}%",
                "pressure": f"{main['pressure']} hPa",
                "clouds": cloud_desc,
                "wind": f"{wind_kmph:.1f} km/hr",
                "rain_prob": f"{prob*100:.1f}%",
                "rain_amt": f"{amt:.2f} mm"
            }

            if prob > 0.5 and amt > 0.5:
                return render_template("rain.html", data=data, advice=get_advice("rain"))
            else:
                return render_template("sunny.html", data=data, advice=get_advice("sunny"))
        else:
            return render_template("index.html", error="City not found.", hour=datetime.now().hour)
    return render_template("index.html", hour=datetime.now().hour)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=10000)
