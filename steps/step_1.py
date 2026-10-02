import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from helpers.download_raw_data import download_project_raw_data
import io
import time



def step_1():
    st.subheader("EDA")
    st.write("Okay. Schauen wir uns die Daten der geladenen CSV näher an und " \
    "wenden wir verschiedene Methoden an unsere Daten zu beurteilen")

############### Sichtprüfung #################

    with st.expander("📁 Sichtprüfung der Daten - der erste Eindruck", expanded=False):
        col_grid, col_ctrl = st.columns([7, 3], border=True)
    
        with col_grid:
        # Ein auf- und zuklappbarer Bereich (Expander)
            st.dataframe(st.session_state.df_raw, use_container_width=False, hide_index=True)

        with col_ctrl:
            python_code = """
            import pandas as pd

            df = pd.DataFrame({
            'Produkt': ['A', 'B', 'C'],
            'Wert': [10, 20, 30]
            })
            """
            st.subheader('Ergebnisse')
            st.write('* Auf den ersten Blick macht das Grid einen geordneten Eindruck, " \
            "Zeilen und Spalten scheinen intakt, es gibt keine "zerschossenen" Spalten.')
            st.write('* Es fällt auf das es sowohl numerische also auch Klassifizierungsdaten gibt.')
            st.write('* Unser zu prognostizierendes Feld "target" ist klar sichtbar')
            st.write('* Schon jetzt, bei den ersten 20 Zeilen fällt auf, dass das Feld "hand" ' \
            'mehrheitlich, erwartbar von Rechtshändern belegt ist.')
            #st.code(python_code, language="python")

            #st.write("Klicke hier, um zurück zu Schritt 1 zu springen:")
            #st.button("Zurück zu Step 1", on_click=switch_to_step, args=(2,))

############### Auswertung mit Python #################
    
    with st.expander("📁 Ausertung mit Python - DataFrame.info()", expanded=False):
        col_grid1, col_grid2, col_grid3 = st.columns([3, 3, 3], border=True)
        
        with col_grid1:
            python_code = """ /*Blick in Code*/
            buffer = io.StringIO()
            st.session_state.df_raw.info(buf=buffer)
            s = buffer.getvalue()"""

            st.subheader('Python-Analyse - fehlerhafte Werte')
            st.write('Prüfung des Datasets mit "DataFrame.info()"')
            st.code(python_code, language="python")
            buffer = io.StringIO()
            st.session_state.df_raw.info(buf=buffer)
            s = buffer.getvalue()

            st.subheader('Ausgabe der Funktion')
            st.text(s)
            
            st.subheader('Auswertung und Erkenntnisse')
            st.write('* Als Ergebnis sehen wir die Anzahl der verfügbaren Spalten (23) und Zeilen (19718) im DataFrame,'
            'bestehend, hauptsächtlich nummerischen Werten, aber auch aus Strings.')
            st.write('* Die Zeilenzahl ist bei num. Feldern ohne Werten bei 0, in den Feldern mit Klassifizierungsdaten ' \
            'kann man erkennen das es Zeilen mit fehlenden Werten gibt da die ' \
            'Anzahl der non-null Zeilen von der Gesamtzahl aller Zeilen abweicht.')

    
        with col_grid2:
            python_code = """            buffer = io.StringIO()
            st.session_state.df_raw.info(buf=buffer)
            s = buffer.getvalue()"""
            st.subheader('Python-Analyse - Ausreißer')
            st.write('Da es sich bei den Feldern N*, E* und C* um Umfragewerte ' \
            'die von 1 bis max. 5 gehen und lt. DataFrame.info() in diesen Feldern keine Nan-Werte vorhanden sind ' \
            'sollte eine Prüfung der betreffenden Felder auf Gültigkeit innerhalb dieser Range genügen.')

            st.dataframe(get_outliers(), use_container_width=True, hide_index=True, height=700)

            st.text('Keine der Werte überschreitet die zulässige Range')

            st.subheader("Alterssuche & Ausreißer-Analyse (Boxplot)")

        # 1. Matplotlib Figure & Axis erstellen
            fig, ax = plt.subplots(figsize=(8, 3))

            # 2. Seaborn Boxplot für die Spalte 'age' zeichnen
            df_filtered = st.session_state.df_raw[st.session_state.df_raw["age"] < 100]
            sns.boxplot(
                data=df_filtered,
                x="age",
                ax=ax,
                color="skyblue",
                showfliers=True,  # Korrekt: True zeigt Ausreißer als Punkte
            )

            # 3. Diagramm-Layout anpassen
            #ax.set_title("Boxplot der Altersverteilung")
            ax.set_xlabel("Alter")

            # 4. In Streamlit anzeigen
            st.pyplot(fig)

def get_outliers():
    # Felder die nicht dazu gehören dropen
    drop_cols = ["gender", "hand", "target", "age"]
    df_features = st.session_state.df_raw.drop(columns=drop_cols)

    # Min und Max berechnen (alle spalten)
    summary_df = pd.DataFrame(
        {"Minimum": df_features.min(), "Maximum": df_features.max()}
    ).reset_index()

    # Umbenennen der Spalte mit den Feldnamen
    summary_df.rename(columns={"index": "Feldname"}, inplace=True)

    return summary_df

    
