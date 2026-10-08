import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from helpers.download_raw_data import download_project_raw_data
from steps.step_0 import step_0 as step_0 # startseite, download data.csv
from steps.step_1 import step_1 as step_1 # eda
from steps.step_2 import step_2 as step_2 # erzeuge clean data file
from steps.step_3 import step_3 as step_3 # model fight
from steps.step_4 import step_4 as step_4 # app
from pathlib import Path # datei exisits check


#header anpassen (weniger padding-top, ausserdem Nutzung der gesamten Browser-Breite)
st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    /* Zielt ausschließlich auf Buttons mit type="primary" ab */
    button[kind="primary"] {
        background-color: #28a745 !important; /* Ein frisches Grün */
        color: white !important;
        border: none !important;
    }
    
    /* Hover-Effekt für den Primär-Button */
    button[kind="primary"]:hover {
        background-color: #218838 !important; /* Dunkleres Grün beim Drüberfahren */
        color: white !important;
    }
    </style>
    <style>
        .block-container { padding-top: 0.1rem; }
        header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Callback-Funktion: Wird ausgeführt, BEVOR die Seite neu gezeichnet wird um switch zurück zu ermöglichen
def switch_to_step(step):
    st.session_state.step = step
    st.session_state.radio_nav = tab_options[step] 

# Funktion, falls der Nutzer manuell auf den Radio-Button klickt 
def update_step_from_radio():
    st.session_state.step = tab_options.index(st.session_state.radio_nav)

fig, ax = plt.subplots(figsize=(6, 4))

# Anlegen und initialisieren von Session Variablen
# Anlegen des Radio-Indexes für Header-Navigation
if "step" not in st.session_state:
  st.session_state.step = 0

# Speicherort für Rohdaten: hier werden die geladenen CSV RAW Daten gespeichert
# und stehen der Session zur Verfügung 
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None

tab_options = ["Start - Rohdaten laden", 
               "EDA - Datenanalyse", 
               "EDA - saubere Daten erzeugen", 
               "Models (Training und Wettkampf)",
               "(App) - Teste das Sieger-Model", 
               "Noch Meer App ;-)"]

# Erstelle den Radio-Button und steuere ihn über den 'index'-Parameter
selected = st.radio(
    "Navigation",
    tab_options,
    index=st.session_state.step,  # Nutzt den aktuellen Schritt als Startindex
    horizontal=True,
    label_visibility="collapsed",
    key="radio_nav",
    on_change=update_step_from_radio
)

# Synchronisiere, falls der Nutzer manuell auf einen Reiter klickt
st.session_state.step = tab_options.index(selected)

st.markdown("---")

######################################## st.session_state.step = 0 = Welcome
if st.session_state.step == 0:
    step_0()
#END ####################################### st.session_state.step = 0 = Welcome

######################################## st.session_state.step = 1 = EDA
if st.session_state.step == 1:
    if st.session_state.df_raw is None:
        st.info('Up`s, du hast vergessen Daten zu laden :-)')
        st.button("Zurück zu Welcome und lade bitte Daten!", on_click=switch_to_step, args=(0,))
        st.stop()
    step_1()
#END ####################################### st.session_state.step = 1 = EDA


######################################## st.session_state.step == 2
if st.session_state.step == 2:
    if st.session_state.df_raw is None:
        st.info('Up`s, du hast vergessen Daten zu laden :-)')
        st.button("Zurück zu Welcome und lade bitte Daten!", on_click=switch_to_step, args=(0,))
        st.stop()

    st.subheader("Bereinigte Daten erstellen")
    st.write("Klick auf den Button um, aus den Erkenntnissen der EDA, einen 'sauberen' Datensatz zu erzeugen.")
    if st.button("Mit 'Klick' auf mich erzeugst du eine saubere " \
    "Daten-File als CSV, welche im Projektordner gespeichert wird.", type="primary"):
        st.write(step_2())

#END ####################################### st.session_state.step == 2


######################################## st.session_state.step == 3
if st.session_state.step == 3:
    if Path("data_clear.csv").is_file() is False:
        st.info('Up`s, du hast vergessen Daten zu laden :-)')
        st.button("Zurück zu Welcome und lade bitte Daten!", on_click=switch_to_step, args=(0,))
        st.stop()

    st.subheader("Finde das Supermodel")
    st.write("Folgende Models werden am Wettbewerb teinehmen: AdaBoostClassifier, RandomForestClassifier, LogisticRegression, Support Vector Classification sowie KNN")
    st.write('Die Models nutzen die zuvor erstellten, bereinigten Daten der Datein data_clear.csv')
    if st.button("Starte den Wettkampf ..", type="primary"):
        st.write(step_3())
#END ####################################### st.session_state.step == 3


######################################## st.session_state.step == 4
if st.session_state.step == 4:
    if Path("best_model.joblib").is_file():
        st.write(step_4())
    else:
        st.info("Tja, leider kein Model zur Umfrage gefunden ..")
#END ####################################### st.session_state.step == 4


######################################## st.session_state.step == 4
if st.session_state.step == 5:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.video("https://www.youtube.com/watch?v=PXQh9jTwwoA&list=RDPXQh9jTwwoA&start_radio=1")
        st.write("Und vielen Dank für deinen Unterricht!")
#END ####################################### st.session_state.step == 4
