from pathlib import Path
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from src.data import load_train, add_rul, add_features, feature_columns

Path("models").mkdir(exist_ok=True)

df = add_rul(load_train())
df = add_features(df)
features = feature_columns()

X, y = df[features], df["RUL"]
groups = df["unit"]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(X, y, groups))

model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    random_state=42,
)

model.fit(X.iloc[train_idx], y.iloc[train_idx])
pred = model.predict(X.iloc[val_idx])

print("MAE :", round(mean_absolute_error(y.iloc[val_idx], pred), 3))
print("RMSE:", round(mean_squared_error(y.iloc[val_idx], pred) ** 0.5, 3))
print("R2  :", round(r2_score(y.iloc[val_idx], pred), 3))

joblib.dump({"model": model, "features": features}, "models/medpredict.joblib")
print("Saved models/medpredict.joblib")
