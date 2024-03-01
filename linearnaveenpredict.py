from flask import Flask, render_template, request

# Import necessary libraries for preprocessing and prediction
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__,template_folder='template1')

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
print(x.isnull().sum())

x["day"].fillna(x["day"].mode()[0],inplace=True)
# Split data into train and test sets

a, b, c, d = train_test_split(x, y, test_size=0.2, random_state=0)
print(x.isnull().sum())
# Instantiate and fit the LinearRegression model
obj = LinearRegression()
obj.fit(a, c)


# Render the index.html template for the root URL
@app.route('/')
def index():
    return render_template('linearnaveenpredict.html')


# Handle form submission and prediction
@app.route('/predict_tip', methods=['POST'])
def predict_tip():
    # Extract the form data
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
    return render_template('linearnaveenpredict.html', prediction=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)