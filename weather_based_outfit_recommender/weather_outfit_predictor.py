import streamlit as st
import pandas as pd
import requests
import joblib
from sklearn.preprocessing import LabelEncoder

# OpenWeather API Key (Hardcoded for now)
API_KEY = "7da29f9618d83beb8bc1187fb0e9ff82"

# Load trained model
model = joblib.load("outfit_recommender.pkl")

# Load dataset to get label encoder and outfit recommendations
df = pd.read_csv("weather_clothing.csv")
label_encoder = LabelEncoder()
df['Outfit'] = label_encoder.fit_transform(df['Outfit'])

# Function to fetch weather data
def get_weather_data(city):
    URL = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(URL)
    data = response.json()

    if response.status_code == 200:
        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        rainfall = data.get("rain", {}).get("1h", 0)  # Default to 0 if no rain
        weather_main = data["weather"][0]["main"]

        return [temperature, humidity, wind_speed, rainfall, weather_main]
    else:
        return None

# Function to fetch an outfit recommendation based on occasion
def get_occasion_outfit(temp, occasion):
    if occasion == "Outdoor":
        return df.loc[df['Temperature'].sub(temp).abs().idxmin(), "Outdoor Outfit"]
    elif occasion == "Formal":
        return df.loc[df['Temperature'].sub(temp).abs().idxmin(), "Formal Outfit"]
    elif occasion == "Party":
        return df.loc[df['Temperature'].sub(temp).abs().idxmin(), "Party Outfit"]
    elif occasion == "Casual":
        return df.loc[df['Temperature'].sub(temp).abs().idxmin(), "Casual Outfit"]
    return "No outfit found"

# Streamlit UI
st.title("🌦 Weather-Based Outfit Recommender 👕👗")
st.write("Enter your details to receive a **personalized** outfit suggestion.")

# User inputs
city = st.text_input("🌍 Enter City Name", "")
gender = st.radio("👤 Select Gender", ["Male", "Female"])
age = st.number_input("🎂 Enter Age", min_value=1, max_value=100, step=1)
occasion = st.selectbox("🎭 Select Occasion", ["Casual", "Formal", "Party", "Outdoor"])

if st.button("Get Recommendation"):
    if city:
        weather_features = get_weather_data(city)

        if weather_features:
            temp, humidity, wind_speed, rainfall, weather_main = weather_features

            # Display weather details
            st.subheader(f"🌤 Weather in {city}")
            st.write(f"🌡 **Temperature:** {temp}°C")
            st.write(f"💧 **Humidity:** {humidity}%")
            st.write(f"💨 **Wind Speed:** {wind_speed} km/h")
            st.write(f"🌧 **Rainfall:** {rainfall} mm")
            st.write(f"☁ **Condition:** {weather_main}")

            # Predict outfit using ML model
            new_weather = pd.DataFrame([[temp, humidity, wind_speed, rainfall]],
                                       columns=['Temperature', 'Humidity', 'WindSpeed', 'Rainfall'])
            predicted_outfit = model.predict(new_weather)
            outfit_name = label_encoder.inverse_transform(predicted_outfit)[0]

            # Get occasion-based outfit recommendation
            occasion_outfit = get_occasion_outfit(temp, occasion)

            # Display recommendations
            st.success(f"👗 **AI Recommended Outfit:** {outfit_name}")
            st.info(f"🎯 **Occasion-Specific Outfit Suggestion:** {occasion_outfit}")

        else:
            st.error("❌ Error fetching weather data. Please check the city name.")
    else:
        st.warning("⚠️ Please enter a city name.")
