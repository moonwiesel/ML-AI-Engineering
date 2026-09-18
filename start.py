import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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
# strategy des Imputers bei Num-Werten ist "median weil" einzelne werte nach sichtung stark abweichen
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numeric_features),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
])

print(numeric_features)

# lernen (fit), transformieren (transform) und rückgabe der Daten
X_prepared = preprocessor.fit_transform(X)
print("Before:", X.shape, " After:", X_prepared.shape)
# 2x2 Columns kommen aus dem OneHotEncoder hinzu (After)


