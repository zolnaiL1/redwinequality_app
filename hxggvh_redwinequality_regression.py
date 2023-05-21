import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Adatok beolvasása
url = "https://raw.githubusercontent.com/zolnaiL1/redwinequality_csv/main/winequality-red.csv"
data = pd.read_csv(url, delimiter=",")

# Az oszlopnevek külön változóba helyezése
column_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

# Oszlopnevek módosítása az adathalmazban
data.columns = column_names

# Eltávolítjuk a sorokat, amelyekben van hiányzó adat
data = data.dropna()

# Bemeneti (X) és kimeneti (y) változók meghatározása
X = data.drop("citric acid", axis=1)
y = data["citric acid"]

# Adatok felosztása tanító és tesztelő adatokra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lineáris regressziós modell létrehozása és illesztése
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Lasso regressziós modell létrehozása és illesztése
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

# Ridge regressziós modell létrehozása és illesztése
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)

# Polinomiális regressziós modell létrehozása és illesztése
poly_features = PolynomialFeatures(degree=2)  # Polinomiális fokszám beállítása
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Lineáris regresszió predikciók a tesztelő adatokon
linear_predictions = linear_model.predict(X_test)

# Lasso regresszió predikciók a tesztelő adatokon
lasso_predictions = lasso_model.predict(X_test)

# Ridge regresszió predikciók a tesztelő adatokon
ridge_predictions = ridge_model.predict(X_test)

# Polinomiális regresszió predikciók a tesztelő adatokon
poly_predictions = poly_model.predict(X_test_poly)

# Lineáris regresszió MSE kiértékelése
linear_mse = mean_squared_error(y_test, linear_predictions)

# Lasso regresszió MSE kiértékelése
lasso_mse = mean_squared_error(y_test, lasso_predictions)

# Ridge regresszió MSE kiértékelése
ridge_mse = mean_squared_error(y_test, ridge_predictions)

# Polinomiális regresszió MSE kiértékelése
poly_mse = mean_squared_error(y_test, poly_predictions)

# Streamlit app
st.set_page_config(page_title="Wine Quality App", page_icon=":wine_glass:")
st.title("Wine Quality App")

# Statisztikai jellemzők kiszámítása
st.subheader("Dataset")
df = pd.read_csv('https://raw.githubusercontent.com/zolnaiL1/redwinequality_csv/main/winequality-red.csv')
st.write(df.describe())

# Display MSE for each model
st.subheader("Mean Squared Error (MSE)")
st.write("Linear Regression MSE:", linear_mse)
st.write("Lasso Regression MSE:", lasso_mse)
st.write("Ridge Regression MSE:", ridge_mse)
st.write("Polynomial Regression MSE:", poly_mse)

# Lineáris regresszió ábrázolása
fig_linear = plt.figure()
ax1 = fig_linear.add_subplot(2, 2, 1)
ax1.scatter(y_test, linear_predictions)
ax1.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
ax1.set_xlabel('Valós értékek')
ax1.set_ylabel('Predikciók')
ax1.set_title('Lineáris regresszió: Valós vs. Predikció')

# Lasso regresszió ábrázolása
fig_lasso = plt.figure()
ax2 = fig_lasso.add_subplot(2, 2, 2)
ax2.scatter(y_test, lasso_predictions)
ax2.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
ax2.set_xlabel('Valós értékek')
ax2.set_ylabel('Predikciók')
ax2.set_title('Lasso regresszió: Valós vs. Predikció')

# Ridge regresszió ábrázolása
fig_ridge = plt.figure()
ax3 = fig_ridge.add_subplot(2, 2, 3)
ax3.scatter(y_test, ridge_predictions)
ax3.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
ax3.set_xlabel('Valós értékek')
ax3.set_ylabel('Predikciók')
ax3.set_title('Ridge regresszió: Valós vs. Predikció')

# Polinomiális regresszió ábrázolása
fig_poly = plt.figure()
ax4 = fig_poly.add_subplot(2, 2, 4)
ax4.scatter(y_test, poly_predictions)
ax4.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
ax4.set_xlabel('Valós értékek')
ax4.set_ylabel('Predikciók')
ax4.set_title('Polinomiális regresszió: Valós vs. Predikció')

# Streamlit alkalmazás felépítése
st.title('Regresszió ábrák')

col1, col2 = st.columns(2)

# Elhelyezzük az ábrákat az oszlopokban
with col1:
    st.pyplot(fig_linear)
    st.pyplot(fig_lasso)

with col2:
    st.pyplot(fig_ridge)
    st.pyplot(fig_poly)

#####
# Adatok beolvasása
url = "https://raw.githubusercontent.com/zolnaiL1/redwinequality_csv/main/winequality-red.csv"
data = pd.read_csv(url, delimiter=",")

# Az oszlopnevek külön változóba helyezése
column_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

# Oszlopnevek módosítása az adathalmazban
data.columns = column_names

# Eltávolítjuk a sorokat, amelyekben van hiányzó adat
data = data.dropna()

# Bemeneti (X) és kimeneti (y) változók meghatározása
X = data.drop("citric acid", axis=1)
y = data["citric acid"]

# Adatok felosztása tanító és tesztelő adatokra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge regressziós modell létrehozása és illesztése
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)

# Streamlit alkalmazás létrehozása
st.title("Ridge regresszió MSE számítása")

# Csúszka létrehozása az érték beállításához
input_value = st.slider("Adjon meg egy értéket:", min_value=0.0, max_value=2.0, step=0.01)

# Funkció a Ridge regresszió MSE számításához az általad megadott érték alapján
def calculate_mse(input_value):
    input_data = X_test.iloc[0].copy()  # Másolatot készítünk a tesztadatok egy soráról
    input_data["citric acid"] = input_value  # Az általad megadott értéket helyettesítjük a "citric acid" oszlopban
    input_data = input_data.values.reshape(1, -1)  # A megfelelő alakúvá alakítjuk az input adatot
    prediction = ridge_model.predict(input_data[:, :-1])  # Predikció a modell segítségével, csak az első 11 oszlopot használjuk
    mse = mean_squared_error([input_value], prediction)  # MSE számítása az általad megadott érték és a predikció alapján
    return mse

# MSE számítása és eredmény kiírása
mse_result = calculate_mse(input_value)
st.write("Ridge regresszió MSE a(z)", input_value, "érték alapján:", mse_result)
