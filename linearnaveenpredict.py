from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Total Tip Bill Prediction.com</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f7f7f7;
            margin: 0;
            padding: 0;
        }
        h1 {
            text-align: center;
            color: #333;
            margin-top: 20px;
        }
        h2 {
            color: #007bff;
        }
        form {
            max-width: 400px;
            margin: 20px auto;
            padding: 20px;
            background-color: #fff;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        label {
            color: #333;
            font-weight: bold;
        }
        input[type="number"],
        input[type="text"],
        select {
            width: 100%;
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            box-sizing: border-box;
        }
        input[type="submit"] {
            width: 100%;
            padding: 10px;
            background-color: #007bff;
            color: #fff;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        input[type="submit"]:hover {
            background-color: #0056b3;
        }
        #result {
            max-width: 400px;
            margin: 20px auto;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        #result h2 {
            color: #333;
        }
        #result p {
            color: #666;
        }
    </style>
</head>
<body>
    <h1>Total Tip Bill Prediction </h1>

    <h2>developed by Naveen S yelloti</h2>
    <form action="/predict_tip" method="post">
        <label for="total_bill">Total Bill Amount:</label><br>
        <input type="number" id="total_bill" name="total_bill" required><br>

        <label for="sex">Sex:</label><br>
        <select id="sex" name="sex" required>
            <option value="Male">Male</option>
            <option value="Female">Female</option>
        </select><br>

        <label for="smoker">Smoker:</label><br>
        <select id="smoker" name="smoker" required>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
        </select><br>

        <label for="day">Day:</label><br>
        <input type="text" id="day" name="day" required><br>

        <label for="time">Time:</label><br>
        <select id="time" name="time" required>
            <option value="Lunch">Lunch</option>
            <option value="Dinner">Dinner</option>
        </select><br>

        <label for="size">Party Size:</label><br>
        <input type="number" id="size" name="size" required><br>

        <input type="submit" value="Predict Tip">
    </form>

    <div id="result">
        <h2>Predicted Tip:</h2>
        <p>{{ prediction }}</p>
    </div>
</body>
</html>
"""

# Load the tips dataset
import seaborn as sns

tips = sns.load_dataset("tips")
df = tips.copy()

# Preprocessing
df = pd.get_dummies(df, columns=["sex", "smoker"], drop_first=True)
df["day"] = df["day"].map({"Sun": 1, "Mon": 2, "Tue": 3, "Wed": 4, "Thu": 5, "Fri": 6, "Sat": 7})

# Define features (x) and target variable (y)
x = df.drop(["tip", "time"], axis=1)
y = df["tip"]
x["day"].fillna(x["day"].mode()[0],inplace=True)
# Split data into train and test sets
a, b, c, d = train_test_split(x, y, test_size=0.2, random_state=0)

# Instantiate and fit the LinearRegression model
obj = LinearRegression()
obj.fit(a, c)


@app.route('/')
def index():
    return render_template_string(html_template)


@app.route('/predict_tip', methods=['POST'])
def predict_tip():
    total_bill = float(request.form['total_bill'])
    sex_female = 1 if request.form['sex'] == 'Female' else 0
    smoker_no = 1 if request.form['smoker'] == 'No' else 0
    day = request.form['day']
    size = int(request.form['size'])

    # Preprocess the input data
    day_map = {"Sun": 1, "Mon": 2, "Tue": 3, "Wed": 4, "Thu": 5, "Fri": 6, "Sat": 7}
    day_numeric = day_map.get(day, 0)  # Get the corresponding numerical value, 0 if not found

    # Make prediction
    prediction = obj.predict([[total_bill, day_numeric, size, sex_female, smoker_no]])

    # Render the result back to the UI
    return render_template_string(html_template, prediction=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)
