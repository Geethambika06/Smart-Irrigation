import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("data.csv")

df = df.drop(columns=["id", "date", "time"])

df["altitude"] = df["altitude"].astype(str)
df["altitude"] = df["altitude"].str.extract(r'(\d+\.?\d*)', expand=False)

df["altitude"] = pd.to_numeric(df["altitude"], errors="coerce")

df["altitude"] = df["altitude"].fillna(df["altitude"].median())


df = df.rename(columns={"soilmiosture": "soil_moisture"})

X = df[["temperature", "pressure", "altitude", "soil_moisture"]]
y = df["class"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=len(label_encoder.classes_),
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\n ACCURACY:")
print(accuracy_score(y_test, y_pred))

print("\n CLASSIFICATION REPORT:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))

print("\n CONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))


joblib.dump(model, "smart_irrigation_xgboost.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\n XGBoost model saved as smart_irrigation_xgboost.pkl")
