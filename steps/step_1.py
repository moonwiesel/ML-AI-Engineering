import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from helpers.download_raw_data import download_project_raw_data
import io



def step_1():
    st.subheader("EDA")
    st.write("Okay. Schauen wir uns die Daten der geladenen CSV näher an und " \
    "wenden wir verschiedene Methoden an unsere Daten zu beurteilen")

############### Sichtprüfung #################

    with st.expander("📁 Sichtprüfung der Daten - der erste Eindruck", expanded=False):
        col_grid1, col_ctrl1 = st.columns([7, 3], border=True)
    
        with col_grid1:
        # Ein auf- und zuklappbarer Bereich (Expander)
            st.dataframe(st.session_state.df_raw, use_container_width=False, hide_index=True)

        with col_ctrl1:
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
            st.write("* Bei der Sortierung der Daten fällt auf das 'age' ungültige Werte (>100) besitzt")
            #st.code(python_code, language="python")

            #st.write("Klicke hier, um zurück zu Schritt 1 zu springen:")
            #st.button("Zurück zu Step 1", on_click=switch_to_step, args=(2,))

############### Auswertung mit Python Trainingsdaten #################
    
    with st.expander("📁 Auswertung mit Python Trainingsdaten", expanded=False):
        col_grid1, col_grid2, col_grid3 = st.columns([2, 4, 3], border=True)
        
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

            # Anzahl der Zeilen mit NaN
            is_nan = st.session_state.df_raw.isna().any(axis=1).sum()
            
            st.subheader('Auswertung und Erkenntnisse')
            st.write('* Als Ergebnis sehen wir die Anzahl der verfügbaren Spalten (23) und Zeilen (19718) im DataFrame,'
            'bestehend, hauptsächtlich nummerischen Werten, aber auch aus Strings.')
            st.write('* Die Zeilenzahl ist bei num. Feldern ohne Werten bei 0, in den Feldern mit Klassifizierungsdaten ' \
            'kann man erkennen das es Zeilen mit fehlenden Werten gibt da die ' \
            'Anzahl der non-null Zeilen von der Gesamtzahl aller Zeilen abweicht.')
            st.write(f"* Die Anzahl der Zeilen welche NaN beinhalten {is_nan}")

    
        with col_grid2:
            python_code = """# Felder die nicht dazu gehören dropen
    drop_cols = ["gender", "hand", "target", "age"]
    df_features = st.session_state.df_raw.drop(columns=drop_cols)

    # Min und Max berechnen (alle spalten)
    summary_df = pd.DataFrame(
        {"Minimum": df_features.min(), "Maximum": df_features.max()}
    ).reset_index()

    # Umbenennen der Spalte index in Feldname
    summary_df.rename(columns={"index": "Feldname"}, inplace=True)"""
            st.subheader('Python-Analyse - Ausreißer Umfragewerte')
            st.write('Da es sich bei den Feldern N*, E* und C* um Umfragewerte ' \
            'die von 1 bis max. 5 gehen und lt. DataFrame.info() in diesen Feldern keine Nan-Werte vorhanden sind ' \
            'sollte eine Prüfung der betreffenden Felder auf Gültigkeit innerhalb dieser Range genügen.')
            st.code(python_code, language="python")

            st.dataframe(get_outliers(), use_container_width=True, hide_index=True, height=700)

            st.write('Keine der Maximum-Werte überschreitet den Maximalwert 5. In den Minimumwerten stehen Nullen.')
            st.write('Eine Sichprüfung (Sortierung) ergab das lediglich eine Zeile 0 enthält. Die Zeile wird gelöscht.')
        
        
        with col_grid3:  
            st.subheader("Überprüfung der Altersangaben")
            st.write(f"Eine Sichtprüfung (Sortierung) ergab das unwahrscheinliche \
            Altersangaben enthalten sind. Es betrifft {(st.session_state.df_raw["age"] > 100).sum()} \
            Zeilen. Diese werden gelöscht. Altersangaben bis 100 werden akzeptiert.")

            st.code('(st.session_state.df_raw["age"] > 100).sum()', language='python')


        # 1. Matplotlib Figure & Axis erstellen
            fig, ax = plt.subplots(figsize=(8, 3))

            # 2. Seaborn Boxplot für die Spalte 'age' zeichnen
            df_filtered = st.session_state.df_raw[st.session_state.df_raw["age"] < 100]

            sns.histplot(
                data=df_filtered,
                x="age",
                kde=True,
                  # Färbt die Punkte nach dem Target ein (optional)
                #palette="viridis",
                ax=ax,
                bins=30,
                alpha=0.7,  # Leichte Transparenz bei vielen Punkten
            )

            # 3. Diagramm-Layout anpassen
            ax.set_title("Histplot der Altersverteilung")
            ax.set_xlabel("Alter")

            # 4. In Streamlit anzeigen
            st.pyplot(fig)
            st.write("Die Altersverteilung zeigt einen deutlichen Überhang der Personen \
                     von ungefähr 18-30 Jahren. Dies könnte eine Einschränkung der Korrektheit " \
                     "der Vorhersage für Personen ab 30 Jahre bedeuten")

            fig, ax = plt.subplots(figsize=(8, 3))
            sns.histplot(
                data=df_filtered,
                x="hand",
                ax=ax,
                bins=3,
                alpha=0.7,  # Leichte Transparenz bei vielen Punkten
            )

            ax.set_title("Histplot der Schreibhand")
            ax.set_xlabel("Rechts/Linkshänder")

            st.pyplot(fig)
            st.write('Sofern die Ausprägung der Schreibhand irgendeine Relevanz / Bezug ' \
            'zu den gegebenen Antworten hat gilt hier Ähnliches wie bei der Altersverteilung.')



############### Auswertung der Daten - Correlation #################

    with st.expander("📁 Prüfen der Daten auf mgl. Korrelationen", expanded=False):
        col_grid3, col_ctrl3 = st.columns([7, 3], border=True)
    
        with col_grid3:

            # Entnehmen ausschließlich nummerische Werte
            numeric_df = st.session_state.df_raw.select_dtypes(include=[np.number])

            # Compute correlation matrix
            corr_matrix = numeric_df.corr()

            df_temp = corr_matrix.copy()

            # Werte der Selbstkorrelation ausschließen
            #np.fill_diagonal(df_temp.values, np.nan)

            # 3. Die Diagonale (Selbstkorrelationen) sicher auf NaN setzen
            for col in corr_matrix.columns:
                corr_matrix.loc[col, col] = np.nan

            #st.write(corr_matrix)
            # Plot heatmap
            fig, ax = plt.subplots(figsize=(16, 10))
            sns.heatmap(corr_matrix, 
                        annot=True, 
                        cmap='coolwarm', 
                        fmt='.2f',
                        vmin=-1, 
                        vmax=1, 
                        center=0, 
                        square=True, 
                        linewidths=0.5, 
                        ax=ax)
            plt.title('Correlation Matrix Heatmap')
            plt.tight_layout()
            st.pyplot(fig)

            st.write(f"Stärkste Korrelation (ohne Selbstbezug): Min = {corr_matrix.min().min():.2f}, Max ="f" {corr_matrix.max().max():.2f}")

        with col_ctrl3:
            python_code = """
            import pandas as pd

            df = pd.DataFrame({
            'Produkt': ['A', 'B', 'C'],
            'Wert': [10, 20, 30]
            })
            """
            st.subheader('Erläuterungen / Auswertung')
            st.write('* Starke Korrelationen (zB: > 0.8 .o. < -0.8) decken starke Zusammenhänge auf, ' \
            'die dauf hindeuten das es sich mglw. um gleiche Inhalte handeln könnte, die zu Redunanz ' \
            'und damit zur Instabilität des Models führen könnten. Prinzipiell handelt es sich um doppelte Informationen. " \
            "Möglicherweise können bei der Auswertung Spalten "gespart" werden.')

            st.write(f"* Die Corr. bewegt sich im Bereich von Min = {corr_matrix.min().min():.2f}, \
            Max = {corr_matrix.max().max():.2f}. Die Werte geben aufgrund der ehr schwachen Relationen \
            keinen Anlass zur Änderung.")



    ############### Target Analyse #################

    with st.expander("📁 Target Analyse", expanded=False):
        col_grid4, col_ctrl4 = st.columns([7, 3], border=True)

        with col_grid4:
            st.subheader("Target Gruppen - eine Liste vorhandener Gruppen:")
            for target_name, group_df in st.session_state.df_raw.groupby("target"):
                st.write(f"Gruppe: {target_name}, Anzahl Zeilen: {len(group_df)}")

        # 1. Figure & Axis erstellen
            fig, ax = plt.subplots(figsize=(8, 3))
            fontsize =6
            # 2. Seaborn Countplot mit 'data=' und 'ax=' aufrufen
            sns.boxplot(
                data=st.session_state.df_raw[st.session_state.df_raw["age"].between(1, 100)],
                x="target",
                y="age",
                palette="viridis",
                width=0.6,
                ax=ax,
            )

            # 3. Titel und Achsenbeschriftungen anpassen
            ax.set_title("Verteilung der Zielgruppen (Target)",fontsize=fontsize)
            ax.set_xlabel("Zielgruppe", fontsize=fontsize)
            ax.set_ylabel("Alter", fontsize=fontsize)
            ax.tick_params(axis="x", labelsize=fontsize)
            ax.tick_params(axis="y", labelsize=fontsize)

            # Optional: X-Achsen-Beschriftungen leicht drehen, falls sie lang sind
            plt.xticks(rotation=15)
            plt.tight_layout()

            # 4. In Streamlit anzeigen
            st.pyplot(fig)


        with col_ctrl4:
            st.subheader('Erläuterungen')
            st.write("* Moderate und Resiliente " \
            "Personen sind in den betreffenden Daten deutlich mehrheitlich vertreten, " \
            "die Streuung des Alters is bei Resilienten stärker ausgeprägt. " \
            "Vornehmlich nehmen mehrheitlich Jüngere an der Umfrage teil.")
            st.write("* Der Boxplot wertet Personen ab etwa 60 Jahren (über dem obersten Whisker) als " \
            "Ausreisser, welche als Punkte dargestellt werden. Für das Ergebnis der Einschätzung älterer " \
            "Teilnehmer der Umfrage kann das Ergebnis daher ungenauer sein")
            st.write("* Ältere Menschen existieren aber nunmal, daher schließen wir aus diese Datensätze zu entfernen.")
            st.write("* Das Diagramm beinhaltet Daten von Personen im Alter von 1-100. Datensätze älterer Art " \
            "oder mit NaN sind nicht enthalten ")
            st.write("* Da das Alter der Teilnehmenden Personen sich mehrheitlich im Bereich " \
            "von etwa 20-35 Jahren zu bewegen scheint ist eventuell die Anwendung als Hyperparameter als ein ehr wenig scharf trennender Parameter zu sehen")

        col_grid5, col_ctrl5 = st.columns([7, 3], border=True)

        with col_grid5:

            # Daten bereinigen, Class-Felder bis auf traget raus
            numeric_df = st.session_state.df_raw.drop(columns=["age", "hand", "gender"]).dropna().copy()

            # Gruppieren nach 'target' und Mittelwerte berechnen
            grouped = numeric_df.groupby("target").mean()

            # Matplotlib Figure & Axis erstellen (Größe anpassen für gute Lesbarkeit)
            fig, ax = plt.subplots(figsize=(12, 4))

            # 5. Seaborn Heatmap direkt auf das gruppierte DataFrame anwenden
            sns.heatmap(
                grouped,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                cbar=True,
                linewidths=0.5,
                ax=ax,
            )

            # 6. Beschriftungen und Layout
            ax.set_title("Durchschnittswerte der Features je Zielgruppe", fontsize=14)
            ax.set_xlabel("Features (Fragen & Alter)", fontsize=11)
            ax.set_ylabel("Zielgruppe", fontsize=11)

            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            # 7. WICHTIG: In Streamlit anzeigen statt plt.show()
            st.pyplot(fig)


        with col_ctrl5:
            st.subheader('Erläuterungen: target by mean')
            st.write("* Auf diesem Plot ist gut erkennbar das in vielen Fällen auffallende Trennungen" \
            "zwischen den einzelnen Gruppen und deren avg-Antworten herrscht. Dies ist ein positiver Indikator" \
            "für ein erfogreiches ML-Model.")
            st.write("* Beispiel: Resilient -> N3. Bekommt man eine Wert unter 3 ist die Wahrscheinlichkeit " \
            "gegeben einen Resilienten zu treffen höher als bei einem aus übrigen Gruppen. " \
            "Kommt zB noch N8 < 1.6 dazu ist es fast ein Treffer.")
            st.write("* Hier zeigt sich auch das die zuvor untersuchten Korrelation keine oder nur schwache Ausprägungen hatten.")



def get_outliers():
    # Felder die nicht dazu gehören dropen
    drop_cols = ["gender", "hand", "target", "age"]
    df_features = st.session_state.df_raw.drop(columns=drop_cols)

    # Min und Max berechnen (alle spalten)
    summary_df = pd.DataFrame(
        {"Minimum": df_features.min(), "Maximum": df_features.max()}
    ).reset_index()

    # Umbenennen der Spalte index in Feldname
    summary_df.rename(columns={"index": "Feldname"}, inplace=True)

    return summary_df

    
