import pandas as pd
import numpy as np
import joblib
import math
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("C:/Users/JoovGoaD/Desktop/final_dataset.csv")
df = df.drop(columns=['date'])

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

df[['squareMeters', 'centreDistance']] = scaler_x.fit_transform(
    df[['squareMeters', 'centreDistance']]
)

df['centreDistance'] *= 0.25

df[['price']] = scaler_y.fit_transform(df[['price']])


city_categories = [df['city'].unique()]
encoder = OneHotEncoder(
    categories=city_categories,
    handle_unknown='ignore',
    sparse_output=False
)

city_ohe = encoder.fit_transform(df[['city']])

city_ohe_df = pd.DataFrame(
    city_ohe,
    columns=encoder.get_feature_names_out(['city']),
    index=df.index
)

df = df.drop(columns=['city'])
df = pd.concat([df, city_ohe_df], axis=1)


y = df['price']
X = df.drop(columns=['price'])

featured_columns = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, shuffle=True, random_state=42
)

model = MLPRegressor()
model = MLPRegressor(
    hidden_layer_sizes=(256, 64, 16),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=2000,
    early_stopping=True,
    n_iter_no_change=22,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

name = "Multiple Layer Perceptron"
paramaters_number = len(X.columns) * 256 + 256 * 64 + 64 * 16 + 16 * 1 + (256+64+16+1)
df_len = len(df)
r2 = r2_score(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = math.sqrt(mse)

metrics = {
    "Model": name,
    "Number of learnable parameters": paramaters_number,
    "Dataset samples amount": df_len,
    "R squared error": r2,
    "Mean squared error": mse,
    "Root mean squared error": rmse
}

#joblib.dump(metrics, "metrics.joblib")
# joblib.dump(model, 'model.joblib')
# joblib.dump(encoder, 'encoder.joblib')
# joblib.dump(scaler_x, 'scaler_x.joblib')
# joblib.dump(scaler_y, 'scaler_y.joblib')
# joblib.dump(featured_columns, 'featured_columns.joblib')