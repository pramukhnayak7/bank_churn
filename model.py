import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

#data cleaninga ahh stuff
data_path = "Customer-Churn-Records.csv"
df = pd.read_csv(data_path)

df = df.drop(['RowNumber','CustomerId', 'Surname'], axis=1)
print(df)

#converting these to numerical form
categorical_columns_to_convert = ['Geography', 'Gender', 'Card Type']

df = pd.get_dummies(df, columns=categorical_columns_to_convert, drop_first=True)

x = df.drop(columns=['Exited'])
y = df['Exited']

#train test model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
print(x_test.shape[0])
print(x_train.shape[0])


numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'HasCrCard', 'IsActiveMember', 'Complain', 'Satisfaction Score', 'Point Earned']

#scaler lets model doent give importance to one value which is larger than other
scaler = StandardScaler()
x_train[numeric_features] = scaler.fit_transform(x_train[numeric_features])
x_test[numeric_features] = scaler.transform(x_test[numeric_features]) 

#model
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)
y_pred=model.predict(x_test)

acurracy = accuracy_score(y_test, y_pred)
print(acurracy)

cf=confusion_matrix(y_test, y_pred)
print(cf)


#-------test-------balls
new_customer = {
    'CreditScore': 600,
    'Age': 45,
    'Tenure': 1,
    'Balance': 120000,
    'NumOfProducts': 1,
    'HasCrCard': 1,
    'IsActiveMember': 0,
    'EstimatedSalary': 100000,
    'Complain': 1,
    'Satisfaction Score': 1,
    'Point Earned': 200,
    'Geography': 'Germany',
    'Gender': 'Male',
    'Card Type': 'DIAMOND'
}

testdf = pd.DataFrame([new_customer])
#encoding lets words to numerical form
testdf_encoded = pd.get_dummies(testdf, columns=categorical_columns_to_convert, drop_first=True)
final_testdf = testdf_encoded.reindex(columns=x.columns, fill_value=0)
final_testdf[numeric_features] = scaler.transform(final_testdf[numeric_features])

prediction = model.predict(final_testdf)
probability = model.predict_proba(final_testdf)

churn_pb = probability[0][1]
stay_pb = probability[0][0]

if prediction[0] ==1: 
   print(churn_pb * 100)
else:
  print(stay_pb * 100)