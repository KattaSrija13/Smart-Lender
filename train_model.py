print("SMART LENDER MODEL TRAINING STARTED")


import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



# =====================================
# LOAD DATASET
# =====================================

print("\nLoading dataset...")


df = pd.read_csv(
    "dataset/loan_data.csv",
    low_memory=False
)


print("Dataset Loaded Successfully!")

print(df.head())



# Remove extra spaces from column names

df.columns = df.columns.str.strip()



print("\nColumns:")
print(df.columns)



# =====================================
# FIX DATASET COLUMN PROBLEM
# =====================================


# If Property_Area and Loan_Status are merged

if "Property_Area" in df.columns:

    df["Property_Area"] = (
        df["Property_Area"]
        .astype(str)
        .str.strip()
    )


if "Loan_Status" in df.columns:

    df["Loan_Status"] = (
        df["Loan_Status"]
        .astype(str)
        .str.strip()
    )



# =====================================
# REMOVE LOAN ID
# =====================================


if "Loan_ID" in df.columns:

    df.drop(
        "Loan_ID",
        axis=1,
        inplace=True
    )


print("\nLoan_ID Removed")



# =====================================
# HANDLE MISSING VALUES
# =====================================


print("\nHandling Missing Values...")


for col in df.columns:

    if df[col].dtype == "object":

        if df[col].isnull().sum() > 0:

            df[col] = df[col].fillna(
                df[col].mode()[0]
            )


    else:

        if df[col].isnull().sum() > 0:

            df[col] = df[col].fillna(
                df[col].median()
            )


print("Missing Values Completed")



# =====================================
# ENCODING
# =====================================


print("\nEncoding Started...")


encoders = {}


categorical_columns = [

    "Gender",

    "Married",

    "Dependents",

    "Education",

    "Self_Employed",

    "Property_Area",

    "Loan_Status"

]



for col in categorical_columns:

    if col in df.columns:

        encoder = LabelEncoder()


        df[col] = encoder.fit_transform(

            df[col]
            .astype(str)
            .str.strip()

        )


        encoders[col] = encoder



print("Encoding Completed")


print("\nData Types:")

print(df.dtypes)



# =====================================
# FEATURES AND TARGET
# =====================================


X = df.drop(
    "Loan_Status",
    axis=1
)


y = df["Loan_Status"]



print("\nFeature Shape:")
print(X.shape)


print("Target Shape:")
print(y.shape)



# =====================================
# TRAIN TEST SPLIT
# =====================================


print("\nSplitting Data...")


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)



# =====================================
# RANDOM FOREST TRAINING
# =====================================


print("\nTraining Random Forest Model...")


model = RandomForestClassifier(

    n_estimators=100,

    random_state=42,

    n_jobs=-1

)



model.fit(

    X_train,

    y_train

)



print("Training Completed")



# =====================================
# ACCURACY
# =====================================


print("\nTesting Model...")


prediction = model.predict(

    X_test

)



accuracy = accuracy_score(

    y_test,

    prediction

)



print(
    "Accuracy:",
    round(accuracy*100,2),
    "%"
)



# =====================================
# SAVE MODEL
# =====================================


print("\nSaving Model...")


os.makedirs(

    "models",

    exist_ok=True

)



joblib.dump(

    model,

    "models/best_model.pkl"

)



joblib.dump(

    encoders,

    "models/label_encoders.pkl"

)



print("\n==============================")

print("MODEL SAVED SUCCESSFULLY")

print("==============================")


print(
    "models/best_model.pkl"
)

print(
    "models/label_encoders.pkl"
)