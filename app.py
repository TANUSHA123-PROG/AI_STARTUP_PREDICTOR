from flask import Flask, render_template, request
import pandas as pd
import csv
import joblib

app = Flask(__name__)

# Load Trained Model
model = joblib.load("model/startup_model.pkl")


@app.route('/')
def home():
    return render_template(
        'index.html',
        funding_score=0,
        employee_score=0,
        founder_score=0,
        age_score=0,
        investor_score=0
    )
@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/features')
def features():
    return render_template('features.html')
@app.route('/predict', methods=['POST'])
def predict():

    try:

        funding = float(request.form['funding'])
        employees = int(request.form['employees'])
        founders = int(request.form['founders'])
        age = int(request.form['age'])
        investors = int(request.form['investors'])

        data = [[
            funding,
            employees,
            founders,
            age,
            investors
        ]]

        prediction = model.predict(data)

        probability = model.predict_proba(data)

        success_probability = round(
            probability[0][1] * 100,
            2
        )

        # Save history
        with open(
            "history.csv",
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                funding,
                employees,
                founders,
                age,
                investors,
                success_probability
            ])

        # Quote
        if success_probability >= 50:

            quote = """
            🚀 Your startup shows promising potential for success.
            Focus on scaling operations, attracting investors,
            and maintaining sustainable growth.
            """

        else:

            quote = """
            ⚠️ Your startup may face challenges in its current stage.
            Consider improving funding, team strength,
            and investor support to increase success chances.
            """

        # Result Text
        if success_probability >= 80:

            result_text = "High Chance of Success 🚀"
            risk = "Low Risk"
            risk_color = "#00ff88"

        elif success_probability >= 60:

            result_text = "Moderate Chance 📈"
            risk = "Medium Risk"
            risk_color = "#ffcc00"

        else:

            result_text = "Low Chance ⚠"
            risk = "High Risk"
            risk_color = "#ff4d4d"
        # Save history
        # Analytics Scores
        funding_score = min(
            round((funding / 5000000) * 100, 2),
            100
        )

        employee_score = min(
            round((employees / 100) * 100, 2),
            100
        )

        investor_score = min(
            round((investors / 20) * 100, 2),
            100
        )

        founder_score = min(
            round((founders / 10) * 100, 2),
            100
        )

        age_score = min(
            round((age / 10) * 100, 2),
            100
        )

        return render_template(
            'index.html',
            probability=success_probability,
            prediction=prediction[0],
            risk=risk,
            risk_color=risk_color,
            result_text=result_text,
            funding_score=funding_score,
            employee_score=employee_score,
            investor_score=investor_score,
            founder_score=founder_score,
            age_score=age_score,
            quote=quote
        )

    except Exception as e:

        return render_template(
            'index.html',
            error=str(e)
        )
@app.route('/dashboard')
def dashboard():

    try:

        df = pd.read_csv(
            "history.csv",
            header=None
        )
        recent_predictions = df.tail(5).values.tolist()
        probabilities = df.iloc[:,5]
        total_funding = df.iloc[:,0].sum()

        total_predictions = len(df)

        average_probability = round(
            probabilities.mean(),
            2
        )

        max_probability = round(
            probabilities.max(),
            2
        )

        min_probability = round(
            probabilities.min(),
            2
        )

        return render_template(
            "dashboard.html",
            total_predictions=total_predictions,
            average_probability=average_probability,
            max_probability=max_probability,
            min_probability=min_probability,
            total_funding=total_funding,
            recent_predictions=recent_predictions
        )

    except:

        return render_template(
            "dashboard.html",
            total_predictions=0,
            average_probability=0,
            max_probability=0,
            min_probability=0
        )

if __name__ == '__main__':
    app.run(
        debug=True
    )