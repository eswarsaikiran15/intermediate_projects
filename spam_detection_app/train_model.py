import pandas as pd
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import ComplementNB  # Better for imbalanced data
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "message"]

# Convert labels to binary (ham = 0, spam = 1)
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Improved Text Cleaning Function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r'\b\d+\b', '', text)  # Remove standalone numbers (keep if part of words)
    return text.strip()

# Apply text cleaning
df["message"] = df["message"].apply(clean_text)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label"], test_size=0.2, random_state=42)

# Build a more accurate pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1, 2), stop_words='english')),  # Consider word pairs
    ('tfidf', TfidfTransformer()),  # Normalize text features
    ('classifier', ComplementNB())  # Complement Naïve Bayes (better for spam detection)
])

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as model.pkl")

# Function to classify messages
def classify_messages(messages):
    cleaned_messages = [clean_text(msg) for msg in messages]
    predictions = model.predict(cleaned_messages)
    
    spam_messages = []
    not_spam_messages = []
    
    for msg, pred in zip(messages, predictions):
        if pred == 1:
            spam_messages.append(msg)
        else:
            not_spam_messages.append(msg)
    
    return spam_messages, not_spam_messages

# Example test messages
example_messages = [
    "Congratulations! You've won a free gift card. Click here to claim now.",
    "Hey, are we still meeting for coffee tomorrow?",
    "Urgent! Your bank account is compromised. Click the link to secure it now.",
    "Hi, just checking in. How are you?",
    "Win a brand-new iPhone now! Limited offer, hurry up!",
    "Get 50% off on all products today only. Visit our website!",
    "Your PayPal account has been suspended. Click to verify now.",  # Should be Spam
    "Are you free this weekend? Let's plan something fun."
]

# Classify test messages
spam, not_spam = classify_messages(example_messages)

# Display classified messages
print("\n=== SPAM MESSAGES ===")
for msg in spam:
    print(f"- {msg}")

print("\n=== NOT SPAM MESSAGES ===")
for msg in not_spam:
    print(f"- {msg}")

# Allow user to enter unlimited messages
while True:
    user_input = input("\nEnter a message to check (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Exiting spam detector.")
        break
    
    prediction = "Spam" if model.predict([clean_text(user_input)])[0] == 1 else "Not Spam"
    print(f"Prediction: {prediction}")
