import pandas as pd
import random
import os

random.seed(42)

rows = []

for i in range(1000):
    gender = random.choice(["Male", "Female"])
    married = random.choice(["Yes", "No"])
    dependents = random.choice(["0", "1", "2", "3+"])
    education = random.choice(["Graduate", "Not Graduate"])
    self_emp = random.choice(["Yes", "No"])

    applicant_income = random.randint(1500, 25000)
    co_income = random.randint(0, 10000)

    loan_amount = random.randint(50, 400)
    loan_term = random.choice([120, 180, 240, 300, 360])
    credit_history = random.choice([0, 1])
    property_area = random.choice(["Urban", "Semiurban", "Rural"])

    # Simple approval logic
    if (
        credit_history == 1
        and applicant_income >= 4000
        and loan_amount <= 250
    ):
        loan_status = "Y"
    else:
        loan_status = random.choice(["Y", "N"])

    rows.append([
        f"LP{i+100000}",
        gender,
        married,
        dependents,
        education,
        self_emp,
        applicant_income,
        co_income,
        loan_amount,
        loan_term,
        credit_history,
        property_area,
        loan_status
    ])

columns = [
    "Loan_ID",
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
    "Loan_Status"
]

df = pd.DataFrame(rows, columns=columns)

os.makedirs("dataset", exist_ok=True)

df.to_csv("dataset/loan_data.csv", index=False)

print("✅ Dataset created successfully!")
print("Location: dataset/loan_data.csv")
print(df.head())
print("\nShape:", df.shape)