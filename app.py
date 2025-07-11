from flask import Flask, render_template, request
import requests
import os
import joblib
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

clf = joblib.load("model/rain_classifier.pkl")
reg = joblib.load("model/rain_regressor.pkl")
scaler = joblib.load("model/scaler.pkl")

app = Flask(__name__)

import random

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

            data = {
                "city": city.title(),
                "temp": f"{main['temp']}°C",
                "humidity": f"{main['humidity']}%",
                "pressure": f"{main['pressure']} hPa",
                "clouds": f"{clouds}%",
                "wind": f"{wind} m/s",
                "rain_prob": f"{prob*100:.1f}%",
                "rain_amt": f"{amt:.2f} mm"
            }

            if prob > 0.5:
                return render_template("rain.html", data=data, advice=get_advice("rain"))
            else:
                return render_template("sunny.html", data=data, advice=get_advice("sunny"))
        else:
            return render_template("index.html", error="City not found.")
    return render_template("index.html", hour=datetime.now().hour)

if __name__ == "__main__":
    app.run(debug=True)
