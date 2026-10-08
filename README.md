PROJEKT-UEBERBLICK UND EINRICHTUNGSANLEITUNG

1. UEBERBLICK

Was das Projekt tut:
Dieses System sagt den Persönlichkeitstyp einer Person anhand eines kurzen 
Fragebogens voraus und stellt die Ergebnisse in einer Streamlit-Web-Anwendung bereit.
Zuvor werden die Rohdaten eingespielt, eine EDA durchgeführt und eine Auswahl von 
Models auf die durch die Analyse bereinigten Daten trainiert.

Das Problem:
Es handelt sich um eine überwachte Klassifikationsaufgabe (Supervised Classification). 
Ziel ist es, Personen basierend auf psychometrischen Antworten und demografischen Merkmalen 
einer von vier spezifischen Persönlichkeitszielgruppen zuzuordnen.

Der Datensatz:
Die Daten basieren auf dem bekannten Big-Five-Persönlichkeitstest (OCEAN) von Kaggle. 
Jede Zeile repräsentiert einen Probanden. 
Der optimierte Datensatz enthält 19 Fragen und Angaben wie Alter, Geschlecht und Schreibhand. 
Zudem als "target"-Feld vier Zielklassen: Moderat, Resilient, Überkontrolliert und Unterkontrolliert.

Der Ansatz:
Der Workflow folgt einem ML-Prozess:

* Import und Bereinigung der Rohdaten
* Exploratorische Datenanalyse (EDA)
* Vorverarbeitungspipeline inklusive Imputation und Skalierung
* Modelltraining und Hyperparameter-Tuning mittels GridSearchCV über verschiedene Algorithmen hinweg 
(Random Forest, AdaBoost, Logistic Regression, SVC, KNN)
* Abspeicherung des besten Modells (Siegermodell) via Joblib
* Bereitstellung einer Streamlit-Oberfläche für Vorhersagen


Das Ergebnis:
Das beste Modell wurde anhand fester Validierungsmetriken (wie F1-Score Macro und Accuracy) 
ermittelt und gespeichert.
Benutzer können anhand der App bzw. des "Siegermodels" Persönlichkeitstypen testen.

So verwendest du die App:
Der Benutzer beantwortet 19 Fragen im Browser und gibt demografische Daten an. 
Die App verarbeitet diese Eingaben live über das trainierte Modell und zeigt den vorhergesagten 
Persönlichkeitstyp sofort an. Du kannst der App-Reiterstruktur nach dem Starten der App von 
Links nach Rechts folgend alle Schritte der App durchgehen.

2. EINRICHTUNG UND REPRODUZIERBARKEIT

Voraussetzung: Du hast das Repository in ein Projektverzeichnis geklont und besitzt eine lokale Python-Installation. 
Führe die folgenden Schritte bitte in der exakten Reihenfolge aus:

Schritt 1: Daten beschaffen
Die App bietet die Möglichkeit die Rohdaten direkt zu downloaden. Alternativ kannst du die Daten selbst über:
https://drive.google.com/uc?export=download&id=1pj6sCZKSmn0hptGgSvIWmlcIfCvkwR0G 
laden und in der App einspielen (Schritt1)

Schritt 2: Umgebung erstellen
Erstelle eine virtuelle Python-Umgebung und aktiviere sie:
python -m venv venv

* Aktivierung unter Windows: venv\Scripts\activate
* Aktivierung unter macOS / Linux: source venv/bin/activate

Schritt 3: Abhängigkeiten installieren
Installriere alle erforderlichen Bibliotheken über die im Repository enthaltene Datei:
pip install -r requirements.txt
Achte darauf das du dich im Projektverzeichnis dabei befindest. (das ist da wo die gitignore liegt)

Kontrolliere ob du auch wirklich Module aus der requirements.txt geladen hast. Führe 
pip list
.. aus und prüfe beispielweise ob sich Pandas in der Liste befindet.

Schritt 4: Streamlit-App lokal starten
Starte die Anwendung im Terminal mit folgendem Befehl, um die Benutzeroberfläche lokal im Browser zu öffnen:
streamlit run app.py

3. PROJEKTSTRUKTUR

ML-AI-Engineering-main/
│
├── app.py                  # Hauptdatei der Streamlit-Anwendung
├── requirements.txt        # Python-Abhängigkeiten
├── .streamlit/
│   └── config.toml         # Streamlit-Konfiguration
├── helpers/
│   └── download_raw_data.py # Skript zum Laden der Rohdaten
└── steps/
    ├── step_0.py           # Import der Rohdaten
    ├── step_1.py           # EDA (Exploratorische Datenanalyse)
    ├── step_2.py           # Preprocessing & Feature Engineering
    ├── step_3.py           # Modelltraining & Grid Search (Siegermodell ermitteln)
    └── step_4.py           # App für Vorhersagen


Im dem Order befinden sich Code-Dateien für folgende Zuständigkeiten:
* python steps/step_0.py (Import der Rohdaten)
* python steps/step_1.py (EDA-Analyse)
* python steps/step_2.py (Preprocessing & Features)
* python steps/step_3.py (Modelltraining, GridSearch & Speicherung des best_model.joblib)
* python steps/step_4.py (Umfrage/Vorhersagen)

4. Repository-Url : https://github.com/moonwiesel/ML-AI-Engineering.git

5. TECH-STACK

Tech Stack

* Programming Language: Python
* Data Manipulation & Analysis: Pandas, NumPy
* Machine Learning: Scikit-Learn (RandomForest, AdaBoost, LogisticRegression, SVC, KNeighborsClassifier, GridSearchCV, Pipelines)
* Web-App Framework: Streamlit
* Model Serialization: Joblib

5. LIZENZ

Dieses Projekt ist für Ausbildungs- und Demonstrationszwecke im Bereich Machine Learning und AI Engineering konzipiert.