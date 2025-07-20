
# WEATHER_PREDICTOR

This project is a Flask-based web application that provides intelligent weather predictions and personalized advice using real-time data from the OpenWeatherMap API and trained machine learning models. It predicts the likelihood of rainfall, classifies rainfall intensity, gives clothing recommendations, and issues rain alerts to users.


## 📚 General Information
The Smart Weather-Based Rain Predictor & Advisor is designed to help individuals plan their daily activities based on accurate weather forecasts powered by real-time API inputs and ML models. The user enters a city name, and the system analyzes current weather features like humidity, temperature, wind, and cloud cover to provide a useful forecast and advisory.

What problem does it solve?

* Helps people prepare for rainfall or possible floods in advance.

* Avoids manual search of weather sites—quick, tailored predictions and advice.

* Gives automated clothing suggestions (e.g., carry umbrella, wear raincoat).

Could be integrated with agriculture, travel, or smart city systems.
## Tech Stack

**Frontend:** HTML, CSS

**Backend:** Python, Flask

**Machine Learning:** Scikit-learn, Pandas, NumPy, joblib

**API:** OpenWeatherMap API

**Utilities:** python-dotenv, requests

**Tools & IDEs:** Pycharm

 

## Setup

The Weather Predictor app is developed in Python and requires the following libraries and tools:

Python (3.6+)

Flask – Web application framework

Requests – For making HTTP requests to the weather API

Pandas – For data manipulation and preprocessing

Joblib – For loading serialized (pre-trained) ML models

Scikit-learn – For machine learning classification and regression

python-dotenv – For handling environment variables securely

📄 All dependencies are listed in the requirements.txt file


## Installation

Follow these steps to install and run the Weather Predictor application locally:

📥 1. Clone the repository
```bash
git clone https://github.com/technesakthi/Weather_Predictor.git
cd Weather_Predictor
```
🧪 2. Set up a virtual environment (recommended)
```bash
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```
📦 3. Install dependencies
```bash
pip install -r requirements.txt
```
🔐 4. Configure environment variables

Create a .env file in the root directory and add your OpenWeatherMap API key:
```bash
OPENWEATHER_API_KEY=your_openweather_api_key_here
```

🚀 5. Run the application
```bash
python app.py
```
The app will be live at:

🌐 http://localhost:3000

## Features

🌦 Rain Prediction
Uses a trained machine learning model to predict whether it will rain or not based on real-time weather inputs.

🌧 Rainfall Intensity Classification
Classifies rainfall into categories like None, Light, Moderate, or Heavy, giving users a better understanding of what to expect.

🧥 Clothing Recommendations
Suggests what to wear or carry based on current weather and predicted rain (e.g., umbrella, raincoat).


🔄 Live Weather Data Integration
Fetches and displays current weather conditions using the OpenWeatherMap API.

📈 Dual Model Support

Classification Model: For rain prediction and intensity level.

Regression Model: For estimating rainfall in mm
## Screenshots
```bash
Day View
```
![App Screenshot](https://ik.imagekit.io/zcwxbddch/Screenshot%202025-07-20%20172533.png?updatedAt=1753012944589)



```bash
Night View (After 6 PM)
```

![App Screenshot](https://ik.imagekit.io/zcwxbddch/Screenshot%202025-07-20%20181011.png?updatedAt=1753015363603)
```bash 
Sunny Weather Display	
Displays sunny conditions with no rain forecast and light outfit suggestions.
```

![App Screenshot](https://ik.imagekit.io/zcwxbddch/Screenshot%202025-07-20%20172739.png?updatedAt=1753013082975)

```bash
Rainy Weather Alert	
Triggers rain alerts and recommends carrying an umbrella or wearing a raincoat.
```

![App Screenshot](https://ik.imagekit.io/zcwxbddch/Screenshot%202025-07-20%20181026.png?updatedAt=1753015364007)


## Author

- [sakthitechne](https://www.github.com/technesakthi)


## Feedback

If you have any feedback, please reach out to me at sakthi.techne@gmail.com

