from flask import Flask, render_template, request
import pickle
import re
import string

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form["email_text"]
    cleaned_text = clean_text(email_text)
    prediction = model.predict([cleaned_text])
    result = "Spam" if prediction[0] == 1 else "Not Spam"
    return render_template("index.html", prediction=result, email_text=email_text)

if __name__ == "__main__":
    app.run(debug=True)
