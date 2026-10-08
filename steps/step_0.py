import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from helpers.download_raw_data import download_project_raw_data
import time;


def step_0():
    st.subheader("Start - Rohdaten laden")
    st.write("Willkommen du :-) , zur ML-Präsentation. Aufgabe ist es, " \
      "das beste Model zu finden, welches anhand vorhandener Umfragedaten " \
      "Persönlichkeitstypen vorhersagt.")
    st.write('Bitte klicke auf den Button "Import Rohdaten" um die unbehandelten Umfragedaten online für dieses Projekt direkt zu laden.')
  
    if st.button("Import Rohdaten ..", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        st.session_state.df_raw = None
    
        for status, *data in download_project_raw_data():
            time.sleep(0.05)
            if status == "progress":
                progress, downloaded, total = data
                # Fortschrittsbalken im Hauptskript aktualisieren
                progress_bar.progress(progress)
            if total > 0:
                status_text.text(
                f"Heruntergeladen: {downloaded / 1024:.1f} KB von"
                f" {total / 1024:.1f} KB"
                )
            else:
                status_text.text(f"Heruntergeladen: {downloaded / 1024:.1f} KB")
            if status == "done":
                st.session_state.df_raw = data[0]
                progress_bar.empty()
                status_text.success("Download erfolgreich! Wenn du nun unten ein Datengrid " \
                "mit den ersten 20 Zeilen siehst ist alles ok. ")

    st.write('Die Quell-URL der Daten-CSV Rohdaten lautet: https://drive.google.com/uc?export=download&id=1pj6sCZKSmn0hptGgSvIWmlcIfCvkwR0G')
    st.write("Alternativ, falls du die Quelldaten zunächst über den oben genannten Link auf " \
      "deine Platte geladen hast, kannst du die Datei hier auswählen und verwenden ..")
    
    hochgeladene_datei = st.file_uploader(
        "Wähle eine CSV-Datei aus", type=["csv"]
        )

    if hochgeladene_datei is not None:
    # Pandas liest das hochgeladene File-Objekt direkt ein
        if hochgeladene_datei.name.endswith(".csv"):
            st.session_state.df_raw = pd.read_csv(hochgeladene_datei)
            st.success("Datei erfolgreich von der Platte importiert!")
    
    
      # DataFrame anzeigen, wenn es da ist
    if st.session_state.df_raw is not None:
        st.dataframe(st.session_state.df_raw.head(20), use_container_width=True, hide_index=False)
    