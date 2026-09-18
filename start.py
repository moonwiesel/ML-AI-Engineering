# main-script zur Projektaufgabe
# import-block - erläuterungen zu den Importen siehe Readme.md
import pandas as pd
# scikit-learn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
# ende import block

df = pd.read_csv("datasource/data.csv")   # erzeugt ein dataframe von data.csv
print("Shape:", df.shape) # zeilen, spalten
df.dropna() # spalten mit fehlenden werten entfernen
print("Shape:", df.shape) # zeilen, spalten
print(df["target"].value_counts()) # Aufteilung der Vorkommen von Werten in Feld target
#print(df.head(10))

# teilen von Features und Target-Daten in zwei DataFrames
X = df.drop(columns=["target"])
y = df["target"]

# numeric_features: nur numerische spalten für "num" pipeline mit StandardScaler
numeric_features = X.drop(columns=["gender", "hand"]).columns

#im beispiel - aber nicht verwendet
#item_cols = [c for c in X.columns if c not in ["age", "gender", "hand"]]
#numeric_features = item_cols + ["age"]

# Spalten für "cat" Pipeline mit OneHotEncoder
categorical_features = ["gender", "hand"]

# definieren des preprozessors; 
# strategy des Imputers bei Num-Werten ist "median" weil einzelne werte nach sichtung 
# stark abweichen. "num" steht für die Verwendung numerischer Werte,
# "cat" für die Verwendung klassifizierter Werte
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")), # ersetzt NaN durch median
        ("scale", StandardScaler()), # bringt alle Values auf annähernt gleiches Niveau
    ]), numeric_features),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")), # ersetzt NaN durch häufigsten Value
        ("onehot", OneHotEncoder(handle_unknown="ignore")), # sofern Testdaten Kategoriene liefern die unbekannt sind 
        # weil im Training nicht vorhanden (gelernt), wird das Programm nicht abgebrochen und in allen spalten des merkmals eine 0 notiert
    ]), categorical_features),
])

print(numeric_features)

# lernen (fit), transformieren (transform) und rückgabe der Daten
X_prepared = preprocessor.fit_transform(X)
print("\n(Rows, Cols) Before Transf.:", X.shape, " After Transf.:", X_prepared.shape, "\n")
# 2x2 Columns kommen aus dem OneHotEncoder hinzu (After)

# model definieren und testen

# zunächst werden trainings und testdaten im verhältnis 80/20 
# nach dem mischen (random_state) getrennt.
# stratify=y sorgt dafür das die Trainings und Testdaten ausgewogen sind (zB gleich viele
# Female/Male im Verhältnis zueinander)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



# model als pipe erstellen
# daten werden im preprozess aufgearbeitet, als classifier-model RandomForestClassifier genutzt
# random_state = durch zahl definierte repruduzierbarkeit, n_jobs=-1(nutze alle CPU-Kerne, 1 = nutze einen)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1)),
    #("model", LogisticRegression()),
])

# Training
model.fit(X_train, y_train)
# Vorhersage anhand Testdaten
predictions = model.predict(X_test)
# vergleich bekannte Testergebnisse mit den vom Model errechneten in %
print("Genauigkeit\nAccuracy:", round(accuracy_score(y_test, predictions) * 100, 2), " %")

"""
Fragen zum Projekt:

Die Klassen sind nicht gleich groß. Warum könnte Accuracy hier eine irreführende Kennzahl sein? (Das beheben wir in Live-Session 2.)
    Die Verteilung Female/Male ist bei der Trennung von Train/Test-Daten eventuell unverhältnismäßig,
    selbiges gilt für Links/Rechtshänder. Im Script sollte "stratify=y" dies aber berücksichtigen.

Was würde passieren, wenn ein neuer Nutzer eine Frage unbeantwortet lässt? Kommt die Pipeline damit zurecht?
    Dies gibt es leider in den Testdaten nicht, zudem werden betreffende Zeilen vorher mit dropna() eliminiert.

Versuch, RandomForestClassifier gegen LogisticRegression auszutauschen. Ändert sich die Accuracy?
    Die Genauigkeit steigt um knapp 1% von 84.03 auf 84.99%
"""

