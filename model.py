import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.metrics import f1_score
import streamlit as st


#All columns: RowNumber,CustomerId,Surname,CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited,Complain,Satisfaction Score,Card Type,Point Earned

#data cleaning
data_path = "Customer-Churn-Records.csv"
df = pd.read_csv(data_path)
# print("Original Class Distribution:")
# print(df['Exited'].value_counts(normalize=True))



df = df.drop(['RowNumber','CustomerId', 'Surname', 'Complain'], axis=1)


# print(df.head())

#converting these to numerical form
categorical_columns_to_convert = ['Geography', 'Gender', 'Card Type']

df = pd.get_dummies(df, columns=categorical_columns_to_convert, drop_first=True)

x = df.drop(columns=['Exited'])
y = df['Exited']

#train test model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
# print("Test set size:", x_test.shape[0])
# print("Train set size:", x_train.shape[0])


numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'HasCrCard', 'IsActiveMember', 'Satisfaction Score', 'Point Earned']



scaler = StandardScaler()
x_train[numeric_features] = scaler.fit_transform(x_train[numeric_features])
x_test[numeric_features] = scaler.transform(x_test[numeric_features]) 

#model
model = RandomForestClassifier(random_state=42,class_weight='balanced',n_estimators=200,max_depth=None)
model.fit(x_train, y_train)
y_pred=model.predict(x_test)

# print("\n--- Model Performance ---")
# acurracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", acurracy)

# cf=confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:\n", cf)

# print("Classification Report:\n", classification_report(y_test, y_pred))


st.title("💳 Customer Churn Prediction")
st.write("Enter customer details below to predict whether they are likely to churn:")

# Collect inputs interactively
credit_score = st.number_input("Credit Score", min_value=300.0, max_value=900.0, value=600.0)
age = st.slider("Age", min_value=18, max_value=100, value=35)
tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=20, value=5)
balance = st.number_input("Account Balance", min_value=0.0, max_value=300000.0, value=50000.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_crcard = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
is_active_member = st.selectbox("Is Active Member", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=300000.0, value=70000.0)
satisfaction_score = st.slider("Satisfaction Score (1–5)", 1, 5, 3)
points_earned = st.number_input("Reward Points Earned", min_value=0, max_value=10000, value=500)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
card_type = st.selectbox("Card Type", ["SILVER", "GOLD", "PLATINUM", "DIAMOND"])

new_customer = pd.DataFrame([{
    'CreditScore': credit_score,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_crcard,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary,
    'Satisfaction Score': satisfaction_score,
    'Point Earned': points_earned,
    'Geography': geography,
    'Gender': gender,
    'Card Type': card_type
}])

# print("\n--- Test New Customer ---")
# new_customer = {
#     'CreditScore': float(input("Credit Score: ")),
#     'Age': int(input("Age: ")),
#     'Tenure': int(input("Tenure (years with bank): ")),
#     'Balance': float(input("Account Balance: ")),
#     'NumOfProducts': int(input("Number of Products: ")),
#     'HasCrCard': int(input("Has Credit Card (1 = Yes, 0 = No): ")),
#     'IsActiveMember': int(input("Is Active Member (1 = Yes, 0 = No): ")),
#     'EstimatedSalary': float(input("Estimated Salary: ")),
#     'Satisfaction Score': int(input("Satisfaction Score (1–5): ")),
#     'Point Earned': int(input("Reward Points Earned: ")),
#     'Geography': input("Geography (France / Germany / Spain): ").strip(),
#     'Gender': input("Gender (Male / Female): ").strip(),
#     'Card Type': input("Card Type (SILVER / GOLD / PLATINUM / DIAMOND): ").strip()
# }

testdf = pd.DataFrame(new_customer)

testdf_encoded = pd.get_dummies(testdf, columns=categorical_columns_to_convert, drop_first=True)

final_testdf = testdf_encoded.reindex(columns=x.columns, fill_value=0)
final_testdf[numeric_features] = scaler.transform(final_testdf[numeric_features])
# Predict when button clicked
if st.button("Predict Churn"):
    prediction = model.predict(final_testdf)
    probability = model.predict_proba(final_testdf)

    churn_prob = probability[0][1] * 100
    stay_prob = probability[0][0] * 100

    if prediction[0] == 1:
        st.error(f"Prediction: **Customer might leave**")
        st.metric("Churn Probability", f"{churn_prob:.2f} %")
        st.metric("Stay Probability", f"{stay_prob:.2f} %")
    else:
        st.success(f"Prediction: **Customer will stay**")
        st.metric("Stay Probability", f"{stay_prob:.2f} %")
        st.metric("Churn Probability", f"{churn_prob:.2f} %")
