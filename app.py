import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from helpers.download_raw_data import download_project_raw_data
from steps.step_0 import step_0 as step_0
from steps.step_1 import step_1 as step_1
import time;

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

tab_options = ["Welcome", "EDA - Datenanalyse", "Preprozessing", "Train the Models"]

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


######################################## st.session_state.step == 1
if st.session_state.step == 10:
    if st.session_state.df_raw is None:
        st.info('Up`s, du hast vergessen Daten zu laden :-)')
        st.button("Zurück zu Welcome und lade bitte Daten!", on_click=switch_to_step, args=(0,))
        st.stop()

    with st.expander("📁 Äußere Ebene (Hier auf-/zuklappen)", expanded=True):
        st.write("Das ist der Inhalt der ersten Ebene.")
        col_grid, col_ctrl = st.columns([7, 3], border=False)
    
        with col_grid:
        # Ein auf- und zuklappbarer Bereich (Expander)
            st.dataframe(st.session_state.df_raw, use_container_width=True)

        with col_ctrl:
            python_code = """
            import pandas as pd

            df = pd.DataFrame({
            'Produkt': ['A', 'B', 'C'],
            'Wert': [10, 20, 30]
            })
            """
            st.code(python_code, language="python")

            st.write("Klicke hier, um zurück zu Schritt 1 zu springen:")

            # Hier ändern wir nur den Integer-Index und erzwingen einen Neuaufbau
        
            st.button("Zurück zu Step 1", on_click=switch_to_step, args=(2,))

    with st.expander("📁 Äußere Ebene (Hier auf-/zuklappen)", expanded=True):
        col_grid2,col_grid3, c4, c5 = st.columns([2,2,2,3],border=False)
        with col_grid2:
            st.pyplot(fig)
            #st.dataframe(df, use_container_width=True)

        with col_grid3:
            st.pyplot(fig)
            #st.dataframe(df, use_container_width=True)

#END ####################################### st.session_state.step == 1


######################################## st.session_state.step == 2
if st.session_state.step == 2:
  st.subheader("Schritt 3: Diagramm")
  st.write("Inhalt von Schritt 3.")
#END ####################################### st.session_state.step == 2


######################################## st.session_state.step == 3
if st.session_state.step == 3:
  st.subheader("Schritt 4: Diagramm")
  st.write("Inhalt von Schritt 4.")
#END ####################################### st.session_state.step == 3