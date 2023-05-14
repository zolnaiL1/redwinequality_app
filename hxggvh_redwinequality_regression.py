import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import streamlit as st

# Adathalmaz betöltése
df = pd.read_csv('https://raw.githubusercontent.com/zolnaiL1/redwinequality_csv/main/winequality-red.csv')

# Statisztikai jellemzők kiszámítása
st.set_page_config(page_title="Wine Quality App", page_icon=":wine_glass:", layout="wide")
st.title("Wine Quality App")
st.subheader("Statisztikai információk:")
st.write(df.describe())

# Train-test felosztás
X_train, X_test, y_train, y_test = train_test_split(df[['citric acid']], df['quality'], test_size=0.2, random_state=42)

# Szórásdiagram ábrázolása a tanítóadathalmazon
st.subheader("Regresziós modell:")
fig, ax = plt.subplots()
ax.scatter(X_train, y_train)
ax.set_xlabel('Citric acid')
ax.set_ylabel('Quality')
st.pyplot(fig)

# Lineáris regresszió modell létrehozása és illesztése a tanítóadathalmazon
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

# Illesztési egyenes ábrázolása a tanítóadathalmazon
fig, ax = plt.subplots()
ax.scatter(X_train, y_train)
ax.plot(X_train, lin_model.predict(X_train), color='red')
ax.set_xlabel('Citric acid')
ax.set_ylabel('Quality')
st.pyplot(fig)

# A lineáris modell teljesítményének értékelése a tesztadathalmazon
y_pred = lin_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write('Lineáris regresszió MSE: ', mse)

# Polinomiális regresszió modell létrehozása és illesztése a tanítóadathalmazon
poly_model = LinearRegression()
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)
poly_model.fit(X_train_poly, y_train)

# Illesztési görbe ábrázolása a tanítóadathalmazon
fig, ax = plt.subplots()
ax.scatter(X_train, y_train)
ax.plot(X_train, poly_model.predict(X_train_poly), color='red')
ax.set_xlabel('Citric acid')
ax.set_ylabel('Quality')
st.pyplot(fig)

# A polinomiális modell teljesítményének értékelése a tesztadathalmazon
y_pred_poly = poly_model.predict(X_test_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
st.write('Polinomiális regresszió MSE: ', mse_poly)

# Logisztikus regresszió model létrehozása és illesztése a tanítóadathalmazon
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Az illesztési egyenes ábrázolása a tanítóadathalmazon
fig, ax = plt.subplots()
ax.scatter(X_train, y_train)
ax.plot(np.sort(X_train.values, axis=0), log_model.predict_proba(np.sort(X_train, axis=0))[:, 1], color='red')
ax.set_xlabel('Citric acid')
ax.set_ylabel('Probability')
st.pyplot(fig)

# A logisztikus modell teljesítményének értékelése a tesztadathalmazon
y_pred = log_model.predict(X_test)
accuracy = log_model.score(X_test, y_test)
st.write('Logisztikus regresszió pontosság: ', accuracy)

# Az alkalmazás felületén megjelenő beviteli mező
st.subheader("Predikciók:")
citric_acid = st.number_input('Enter citric acid value')

# Az előrejelzett minőségi pontszám kiszámítása a beviteli mező alapján
predicted_quality = lin_model.predict([[citric_acid]])
st.write('Predicted quality: ', predicted_quality[0])