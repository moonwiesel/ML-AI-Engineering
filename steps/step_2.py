import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from helpers.download_raw_data import download_project_raw_data
import io
import time

def step_2():
    df = pd.DataFrame(st.session_state.df_raw)
    df_clear = df.dropna() # NaN beseitigen
    df_clear = df_clear[df_clear["age"].between(1, 100)] # nur alter zwischen 1 bis 100
    df_clear = df_clear[df_clear["N3"].ne(0)] # zeile mit Nullen in der Umfrage raus. 
    df_clear.to_csv("data_clear.csv", index=False) # speichern ohne index

    st.write('## Die Datei wurde erstellt.')
    st.write('* Es wurden in sämtliche Zeilen NULL-Werten entfernt')
    st.write('* Es wurden sämtliche Zeilen mit einem Alter > 100 entfernt')
    st.write('* Es wurde eine Zeile mit ungültigen Umfragedaten entfernt')
    st.subheader("Die Datei liegt mit dem Namen: 'data_clear.csv' im Hauptverzeichnis deiner Anwendung.")
    return ''