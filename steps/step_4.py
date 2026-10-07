import pandas as pd
import streamlit as st
import random
import joblib

def step_4():
    st.session_state["best_model"] = joblib.load("best_model.joblib")
    render_input_form()

    return ''
def get_random_values():
    data = {
      "N1": random.randint(1, 5),
      "N2": random.randint(1, 5),
      "N3": random.randint(1, 5),
      "N4": random.randint(1, 5),
      "N5": random.randint(1, 5),
      "N6": random.randint(1, 5),
      "N7": random.randint(1, 5),
      "N8": random.randint(1, 5),
      "N9": random.randint(1, 5),
      "N10": random.randint(1, 5),
      "E1": random.randint(1, 5),
      "E3": random.randint(1, 5),
      "E4": random.randint(1, 5),
      "E5": random.randint(1, 5),
      "E7": random.randint(1, 5),
      "E9": random.randint(1, 5),
      "E10": random.randint(1, 5),
      "A4": random.randint(1, 5),
      "C4": random.randint(1, 5),
      "age": random.randint(10, 100),  # z.B. sinnvolleres Alter
      "gender": get_rand_gender_string(),
      "hand": get_rand_hand_string(),  # Klammern ergänzt!
  }

  # In ein DataFrame mit genau einer Zeile umwandeln
    df = pd.DataFrame([data])
    return df

def get_rand_gender_string():
    g = random.randint(0,1)
    gender = ""
    match g:
        case 0: gender = "male"
        case 1: gender = "female"
    return gender

def get_rand_hand_string():
    g = random.randint(0,2)
    hand = ""
    match g:
        case 0: hand = "right"
        case 1: hand = "left"
        case 2: hand = "both"
    return hand   

def render_input_form():
  st.subheader("📝 Probanden-Eingabemaske")
  st.write(
      "Geben Sie hier die Frageinhalte ein und bewerten Sie die Items (1 bis"
      " 5):"
  )

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
    st.markdown("### Psychometrische Fragen")

    for field in personality_fields:
      # Zwei Spalten: Links das Freifeld für die Frage, rechts die Radiobuttons
      col_text, col_radio = st.columns([1.5, 2.5])

      with col_text:
        # Freifeld für den Inhalt der Frage
        st.text_input(
            f"Fragetext für {field}",
            value=f"Frage zu {field}...",
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
      input_data["gender"] = st.selectbox("Geschlecht", options=["male", "female"])

    with col_h:
      input_data["hand"] = st.selectbox("Händigkeit", options=["right", "left", "both"])

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
