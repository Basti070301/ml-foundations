import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Daten laden
wage_data = pd.read_csv("Model/linear_regression/Data/Raw/wage_data.csv")

# Outputvariable definieren
y = wage_data["wage"]

# Eingangsvariablen definieren
x = wage_data[["educ", "exper", "tenure", "nonwhite", "female", "married"]]

# Model erstellen
linear_regression = LinearRegression().fit(x, y)

# Koeffizienten
coeffs = pd.DataFrame({
    "Feature": x.columns,
    "Koeffizient": linear_regression.coef_
})

# Modellgüte (R²)
from sklearn.metrics import r2_score
y_pred = linear_regression.predict(x)

# Residuen berechnen
residuals = y - linear_regression.predict(x)

#Grafische Darstellung
#plt.figure(figsize=(6,4))
#sns.scatterplot(x=y_pred, y=residuals)
#plt.axhline(0, color="red", linestyle="--")
#plt.xlabel("Vorhergesagte Werte")
#plt.ylabel("Residuen")
#plt.title("Residuen vs. Vorhersage")
#plt.show()

#Histogramm der Residuen
#plt.figure(figsize=(6,4))
#sns.histplot(residuals, kde=True)
#plt.title("Histogramm der Residuen")
#plt.show()

# Vorhersage für neue Daten
new_data = pd.DataFrame({
    "educ": [12],
    "exper": [5],
    "tenure": [3],
    "nonwhite": [0],
    "female": [1],
    "married": [0]
})
predictions = linear_regression.predict(new_data)
print(predictions)
#Ergebnis: ca. 3,83 Dollar pro Stunde