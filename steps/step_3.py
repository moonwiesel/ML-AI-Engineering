import pandas as pd
import streamlit as st
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib


def step_3():

    
    # Laden der bereinigten Daten und Ergebnisse von Features trennen 
    df = pd.read_csv("data_clear.csv")
    X = df.drop(columns="target")
    y = df["target"]

    #auswahl set der zu verwenden model's
    models = {
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Classification": SVC(random_state=42),
        "KNN": KNeighborsClassifier()
    }


    # modeling transformer und preprozessor

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    # preprozessor beinhaltet transformer, der prozessor ist später bestandteil der pipe
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, make_column_selector(dtype_include='number')),
            ('cat', categorical_transformer, ['gender', 'hand'])
        ]
    )

    #hyperparameter definitionen je Model
    param_grid_RandomForest = {
        "classifier__n_estimators": [50, 100, 200],  # Anzahl der Bäume
        "classifier__max_depth": [1, 10, 20],  # Maximale Tiefe der Bäume
        "classifier__min_samples_split": [2, 5],  # Mindestanzahl für Splits (Probanden je Knoten) bevor nicht weiter gemacht wird
    }  

    param_grid_AdaBoost = {
        "classifier__n_estimators": [50, 100, 200],  # Anzahl der Bäume
        "classifier__learning_rate": [0.01, 0.1, 1.0, 10.0],  # Lernrate (kleiner = exakter, größer = schneller aber gröbere Sprünge)
        #"classifier__max_depth": [1,2,3],  # Tiefe, ehr flache Bäume
    } 

    param_grid_LogistischeRegression = {
        "classifier__C":[0.01, 0.1, 1.0, 10.0] #Stärke der "Bestrafung von Falschaussagen" Hohe Wert mehr Freirraum
    }    

    param_grid_SVC = {
         "classifier__C":[0.01, 0.1, 1.0, 10.0] # hoher Wert = Härtere Vorgehendsweise gegen Fehler, dadurch Aufwendiger
    }

    param_grid_KNN = {
         "classifier__n_neighbors":[2, 5, 7, 10]
    }

    # df für Auswertungsdaten
    result_df = pd.DataFrame(columns=["Model", "f1_macro", "accuracy", "best_params"])
    i = 0 #result grid index

    # Fester Splitter für absolute Gleichheit auf allen PCs
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # 5 Folds, random_state überall gleich

    #build pipes in einer for-Schleife
    # das model-Set durchlaufen
    for name, model in models.items():

        # zuordnung hyperparams
        hyperparams = ""
        match name:
            case "Random Forest":
                hyperparams = param_grid_RandomForest
            case "AdaBoost":
                hyperparams = param_grid_AdaBoost
            case "Logistic Regression":
                hyperparams = param_grid_LogistischeRegression
            case "Support Vector Classification":
                hyperparams = param_grid_SVC
            case "KNN":
                hyperparams = param_grid_KNN
            case _:
                continue

        # Pipeline zusammensetzen
        pipeline = Pipeline(
        steps=[("preprocessor", preprocessor),
                ("classifier", model)]
        )

        # wo bin ich?
        dyn_text = st.empty()
        dyn_text.write(f"Model lernt: {name} -> {model}")

        # starten in 5 Folds, alle Kombinationen der Hyperparameter = GridSearch
        grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=hyperparams,
        cv=cv_splitter,  # 5-fache Kreuzvalidierung und random_state=42
        scoring={ # score metriken definieren
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
        },
        refit="f1_macro", # berechne f1_macro
        n_jobs=-1,  # Nutze alle CPU-Kerne deines PCs für die Suche
        )

        grid_search.fit(X, y) # execute GridSearch

        # Index der besten Parameterkombination
        best_index = grid_search.best_index_
        # Die dazugehörige Accuracy aus den CV-Ergebnissen fischen
        best_accuracy = grid_search.cv_results_["mean_test_accuracy"][best_index]
        # Beste Hyper-Parameter (Kombi)
        best_params = grid_search.cv_results_["params"][best_index]
        result_df.loc[i] = [model, grid_search.best_score_, best_accuracy, best_params]
        i = i + 1

        # dient zur ermittlung des besten scores
        if (
            "best_overall_score" not in st.session_state
            or grid_search.best_score_ > st.session_state["best_overall_score"]
        ):
          st.session_state["best_overall_score"] = grid_search.best_score_
          st.session_state["best_model"] = grid_search.best_estimator_
          st.session_state["best_model_name"] = name

        dyn_text.markdown(f"**{name} ist bereit ..**")

    # serialisieren (speichern) des besten model
    joblib.dump(st.session_state["best_model"], "best_model.joblib")

    st.write("Das serialisierte Model liegt im Hauptverzeichnis unter best_model.joblib")

    # anzeigen der einzelergebnisse
    st.dataframe(result_df.sort_values(by="f1_macro", ascending=False))

    # Ausgabe der Auswertung
    st.write(f"## And the winner is: {st.session_state["best_model_name"]}" )

    st.balloons() # Freude und Heiterkeit
        
    return ''