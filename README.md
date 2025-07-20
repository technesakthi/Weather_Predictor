
# Weather Predictor

## Introduction

**Weather Predictor** is a smart weather forecasting application built with Python and Flask. It provides real-time weather predictions by fetching data from an external weather API (OpenWeatherMap) and then processing this data through machine learning models that have been trained to predict rain probability and rainfall amounts. The application uses a classifier to determine the likelihood of rain and a regression model to estimate the expected rainfall in millimeters. With dynamic templates and multimedia backgrounds, the app offers an engaging user experience where the displayed content—whether for rainy or sunny conditions—is automatically adjusted based on the forecast outcome. fileciteturn0file1

## Usage

To run the Weather Predictor application, perform the following steps:

1. Clone the repository to your local machine.
2. Install the required dependencies (see Requirements section).
3. Set up your environment variables, including a valid OpenWeatherMap API key.
4. Run the application using the provided command.

For example, if you have set up everything in your virtual environment, you can start the server with the command:

  $ python app.py

This will start the Flask server on port 3000 (hosted on 0.0.0.0 by default), allowing you to access the application in your web browser at http://localhost:3000. The main page displays a form where you can input the name of any city, and based on the retrieved weather data, the appropriate results page (rainy or sunny) will be displayed. fileciteturn0file1

## Features

- **Real-time Weather Data:**  
  The application retrieves current weather conditions such as temperature, humidity, pressure, cloud cover, and wind speed using the OpenWeatherMap API.
  
- **Machine Learning Predictions:**  
  Two pre-trained models are used:
  • A classifier predicts the likelihood of rain (using a RandomForestClassifier).  
  • A regressor estimates the actual rainfall (using a RandomForestRegressor) when rain is expected.  
  These models were trained and optimized using historical weather data. fileciteturn0file2

- **Dynamic User Interface:**  
  Depending on the forecast, the app renders a sunny page or a rainy page with distinctive multimedia elements:
  • Video backgrounds change to reflect the current weather (e.g., rain.mp4 versus sunny.mp4).  
  • Audio elements such as rain sounds or light background tracks enhance user immersion.
  
- **User Guidance:**  
  Helpful advice is provided based on the predicted weather. For instance, in a rainy scenario the app suggests carrying an umbrella, while on a sunny day it offers tips like wearing sunscreen. fileciteturn0file1

- **Responsive Design:**  
  The front-end templates are designed to automatically adjust visual content based on the time of day and weather conditions.

## Requirements

The Weather Predictor app is developed in Python and requires the following libraries and tools:

- **Python (3.6+)**
- **Flask** – for web application framework
- **Requests** – for HTTP requests to the weather API
- **Pandas** – for handling data sets and feature processing
- **Joblib** – for loading pre-trained machine learning models
- **Scikit-learn** – for machine learning components (classification and regression)
- **python-dotenv** – for managing environment variables
- **gdown** – for possible file downloads

A complete list of dependencies is provided in the requirements.txt file. fileciteturn0file6

## Installation

Follow these steps to install and run the Weather Predictor application:

1. **Clone the repository:**

   ```
   $ git clone https://github.com/technesakthi/Weather_Predictor.git
   $ cd Weather_Predictor
   ```

2. **Set up a virtual environment (optional but recommended):**

   ```
   $ python -m venv venv
   $ source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```
   $ pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**

   Create a `.env` file in the project root directory and add your OpenWeatherMap API key:

   ```
   OPENWEATHER_API_KEY=your_openweather_api_key_here
   ```

5. **Run the Application:**

   Start the Flask server with the command below:

   ```
   $ python app.py
   ```

   The application will run on port 3000. Open your browser and navigate to http://localhost:3000 to start using the Weather Predictor. fileciteturn0file1

---

This README provides a comprehensive overview of the Weather Predictor project and should assist both in running the application and understanding its functionality. Enjoy seamless weather forecasting with a smart interface that combines real-time data and machine learning insights!
