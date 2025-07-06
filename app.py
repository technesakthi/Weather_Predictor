from flask import Flask, render_template, request
import requests
import joblib
import pandas as pd
import os
import random
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

app = Flask(__name__)

# Load models
clf = joblib.load("model/rain_classifier.pkl")
reg = joblib.load("model/rain_regressor.pkl")
scaler = joblib.load("model/scaler.pkl")

CLOTHES = [
    "Carry an umbrella.", "Wear a raincoat.", "Use waterproof boots.",
    "Wear a light shirt.", "Put on a jacket.", "Use sunglasses.", "Wear breathable cotton clothes."
]

ACTIVITIES = [
    "You can enjoy a walk outside.", "Consider indoor activities today.",
    "Good day for a coffee indoors.", "Try light exercise indoors.",
    "Watch a movie or read a book.", "Avoid outdoor events if possible."
]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        city = request.form["city"]
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

        try:
            response = requests.get(url).json()
            if response.get("cod") != 200:
                raise ValueError("City not found")

            weather_main = response["weather"][0]["main"].lower()
            main = response["main"]
            wind = response["wind"]
            clouds = response["clouds"]
            current_hour = int(response["dt"] + response["timezone"]) // 3600 % 24

            # Features for prediction
            data = {
                "Humidity3pm": main["humidity"],
                "Pressure9am": main["pressure"],
                "Temp3pm": main["temp"],
                "Cloud3pm": clouds["all"],
                "WindSpeed3pm": wind["speed"],
                "RainToday": 1 if "rain" in weather_main else 0
            }

            X_input = pd.DataFrame([data])
            X_scaled = scaler.transform(X_input)

            # Rain prediction
            rain_pred = clf.predict(X_scaled)[0]
            rain_prob = clf.predict_proba(X_scaled)[0][1]
            rainfall_amount = round(reg.predict(X_scaled)[0], 2) if rain_pred else 0.0

            # Background & sound
            if rain_pred:
                bg = "rainy"
                sound = "rain.mp3"
            elif current_hour >= 19 or current_hour <= 5:
                bg = "night"
                sound = "sunny.mp3"
            else:
                bg = "day"
                sound = "sunny.mp3"

            result = {
                "city": city.title(),
                "temp": data["Temp3pm"],
                "humidity": data["Humidity3pm"],
                "pressure": data["Pressure9am"],
                "cloud": data["Cloud3pm"],
                "wind": data["WindSpeed3pm"],
                "probability": round(rain_prob * 100, 2),
                "amount": rainfall_amount if rain_pred else None,
                "advice": random.choice(CLOTHES) + " " + random.choice(ACTIVITIES),
                "bg_class": bg,
                "sound": sound
            }

        except Exception as e:
            print("Error:", e)
            result = {"error": "City not found or API error."}

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
