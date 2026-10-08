import pandas as pd
import streamlit as st
import random
import joblib

def step_4():
    st.session_state["best_model"] = joblib.load("best_model.joblib")
    render_input_form()

    return ''

def render_input_form():
  st.subheader("📝 Probanden-Eingabemaske")
  st.write(
      "Geben Sie hier die Frageinhalte ein und bewerten Sie die Items (1 bis"
      " 5):"
  )
  st.write("Die Vorhersage nutzt das zuvor ermittelte 'Supermodel' aus dem Hauptverzeichnis der App, 'best_model.joblib'.")


  # Vordefinierte deutsche Beispielfragen
  questions = {
    # Neurotizismus (N)
    "N1": "Ich mache mir oft Sorgen über Dinge, die passieren könnten.",
    "N2": "Ich fühle mich oft niedergeschlagen und traurig.",
    "N3": "Ich reagiere in stressigen Situationen schnell nervös.",
    "N4": "Ich fühle mich oft angespannt und unruhig.",
    "N5": "Ich reagiere leicht empfindlich auf Kritik.",
    "N6": "Ich mache mir zu viele Gedanken über Kleinigkeiten.",
    "N7": "Ich habe oft das Gefühl, die Kontrolle zu verlieren.",
    "N8": "Ich ärgere mich schnell über alltägliche Hindernisse.",
    "N9": "Ich neige dazu, pessimistisch in die Zukunft zu blicken.",
    "N10": "Ich brauche lange, um mich von emotionalen Rückschlägen zu erholen.",
    # Extraversion (E)
    "E1": "Ich bin ein geselliger und kontaktfreudiger Mensch.",
    "E3": "Ich strahle viel Energie aus und bin gerne aktiv.",
    "E4": "Ich knüpfe leicht neue Kontakte und spreche Fremde an.",
    "E5": "Ich bin ein eher optimistischer und fröhlicher Typ.",
    "E7": "Ich mag es, unter vielen Menschen zu sein.",
    "E9": "Ich rede gerne viel und halte mich ungern im Hintergrund.",
    "E10": "Ich sprühe oft vor Tatendrang und Unternehmungslust.",
    # Verträglichkeit (A)
    "A4": "Ich versuche, rücksichtsvoll zu sein und Streit zu vermeiden.",
    # Gewissenhaftigkeit (C)
    "C4": "Ich arbeite sehr gründlich, zuverlässig und zielgerichtet.",
  }
  
  # Alle psychometrischen Felder aus deinen Daten
  personality_fields = [  
      "N1",
      "N2",
      "N3",
      "N4",
      "N5",
      "N6",
      "N7",
      "N8",
      "N9",
      "N10",
      "E1",
      "E3",
      "E4",
      "E5",
      "E7",
      "E9",
      "E10",
      "A4",
      "C4",
  ]

  # Dictionary zum Sammeln der Werte für das DataFrame
  input_data = {}

  with st.form("prediction_form"):
    st.markdown("### Fragen von wenig zutreffend = 1 bis hoch = 5")

    for field in personality_fields:
      # Zwei Spalten: Links das Freifeld für die Frage, rechts die Radiobuttons
      col_text, col_radio = st.columns([1.5, 2.5])

      with col_text:
        # Freifeld für den Inhalt der Frage
        st.text_input(
            f"Fragetext für {field}",
            value=f"{questions[field]}",
            key=f"question_text_{field}",
            label_visibility="collapsed",  # Platzsparend, optional
        )

      with col_radio:
        # Radiobutton von 1 bis 5 (horizontal angeordnet)
        input_data[field] = st.radio(
            f"Wert für {field}",
            options=[1, 2, 3, 4, 5],
            horizontal=True,
            key=f"radio_{field}",
        )

    st.markdown("---")
    st.markdown("### Demografische Daten")

    col_a, col_g, col_h = st.columns(3)

    with col_a:
      input_data["age"] = st.selectbox(
          "Alter", options=list(range(1, 101)), index=45
      )

    with col_g:
      input_data["gender"] = st.selectbox("Geschlecht", options=["Male", "Female"])

    with col_h:
      input_data["hand"] = st.selectbox("Händigkeit", options=["Right", "Left", "Both"])

    # Absende-Button für das Formular
    submitted = st.form_submit_button("🚀 Vorhersage berechnen")

    if submitted:
      # In DataFrame (1 Zeile) umwandeln
      df_input = pd.DataFrame([input_data])

      # Im Session State speichern, falls andere Schritte es brauchen
      st.session_state["user_input_df"] = df_input

      st.success("Daten erfolgreich erfasst!")
      st.dataframe(df_input)

      # Wenn bereits ein trainiertes Modell vorhanden ist, direkte Vorhersage starten:
      if "best_model" in st.session_state:
        prediction = st.session_state["best_model"].predict(df_input)
        st.info(f"🏆 Vorhergesagte Zielgruppe: **{prediction[0]}**")
      else:
        st.warning(
            "Es wurde noch kein Modell trainiert. Bitte führe zuerst das"
            " Modelltraining (Schritt 3) aus."
        )
