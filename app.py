from flask import Flask, render_template, request
import joblib
import pandas as pd


# ==================================
# FLASK APP
# ==================================

app = Flask(__name__)


# ==================================
# LOAD MODEL
# ==================================

model = joblib.load(
    "models/best_model.pkl"
)

encoders = joblib.load(
    "models/label_encoders.pkl"
)


print("Model Loaded Successfully")



# ==================================
# HOME PAGE
# ==================================

@app.route("/")
def splash():
    return render_template("splash.html")


@app.route("/home")
def home():
    return render_template("index.html")



# ==================================
# PREDICT LOAN
# ==================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Getting values from form

        data = pd.DataFrame({

            "Gender": [
                request.form["Gender"]
            ],

            "Married": [
                request.form["Married"]
            ],

            "Dependents": [
                request.form["Dependents"]
            ],

            "Education": [
                request.form["Education"]
            ],

            "Self_Employed": [
                request.form["Self_Employed"]
            ],

            "ApplicantIncome": [
                float(request.form["ApplicantIncome"])
            ],

            "CoapplicantIncome": [
                float(request.form["CoapplicantIncome"])
            ],

            "LoanAmount": [
                float(request.form["LoanAmount"])
            ],

            "Loan_Amount_Term": [
                float(request.form["Loan_Amount_Term"])
            ],

            "Credit_History": [
                float(request.form["Credit_History"])
            ],

            "Property_Area": [
                request.form["Property_Area"]
            ]

        })



        # ==================================
        # ENCODE INPUT DATA
        # ==================================

        categorical_columns = [

            "Gender",

            "Married",

            "Dependents",

            "Education",

            "Self_Employed",

            "Property_Area"

        ]


        for col in categorical_columns:

            data[col] = encoders[col].transform(
                data[col].astype(str)
            )



        # ==================================
        # PREDICTION
        # ==================================

        prediction = model.predict(
            data
        )


        probability = model.predict_proba(
            data
        )


        confidence = round(
            max(probability[0]) * 100,
            2
        )



        if prediction[0] == 1:

            result = "Loan Approved ✅"

        else:

            result = "Loan Rejected ❌"



        return render_template(

            "result.html",

            prediction=result,

            confidence=confidence

        )



    except Exception as e:

        return f"Error: {str(e)}"




# ==================================
# START SERVER
# ==================================

if __name__ == "__main__":

    app.run(
        debug=True
    )