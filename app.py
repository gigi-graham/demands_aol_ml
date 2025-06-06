from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        inventory = int(request.form["inventory"])
        sold = int(request.form["sold"])
        ordered = int(request.form["ordered"])
        price = float(request.form["price"])
        discount = int(request.form["discount"])
        holiday = int(request.form["holiday"])
        competitor = float(request.form["competitor"])

        categories = ["Clothing", "Electronics", "Furniture", "Groceries", "Toys"]
        category_input = request.form["category"]
        category = [1.0 if category_input == c else 0.0 for c in categories]

        regions = ["East", "North", "South", "West"]
        region_input = request.form["region"]
        region = [1.0 if region_input == r else 0.0 for r in regions]

        seasons = ["Autumn", "Spring", "Summer", "Winter"]
        season_input = request.form["season"]
        season = [1.0 if season_input == s else 0.0 for s in seasons]

        weathers = ["Cloudy", "Rainy", "Snowy", "Sunny"]
        weather_input = request.form["weather"]
        weather = [1.0 if weather_input == w else 0.0 for w in weathers]

        features = [inventory, sold, ordered, price, discount, holiday, competitor]
        features += category + region + season + weather

        columns = joblib.load("columns.pkl")

        df = pd.DataFrame([features], columns=columns)

        numeric_cols = ['Inventory Level', 'Units Sold', 'Units Ordered', 'Price', 'Discount', 'Competitor Pricing']
        df[numeric_cols] = scaler.transform(df[numeric_cols])

        prediction = model.predict(df)[0]

        return render_template("predict.html", prediction_text=f"Predicted result: {round(prediction, 2)} units!")

    return render_template("predict.html")

if __name__ == "__main__":
    app.run(debug=True)
