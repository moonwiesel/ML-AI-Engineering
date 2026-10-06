# Datei: utils.py
import io
import pandas as pd
import requests


def download_project_raw_data():
  """Ein Generator, der den Download-Fortschritt schrittweise an das Hauptskript meldet."""
  response = requests.get("https://drive.google.com/uc?export=download&id=1pj6sCZKSmn0hptGgSvIWmlcIfCvkwR0G", stream=True)

  response.raise_for_status()

  total_size = int(response.headers.get("content-length", 0))
  downloaded_size = 0
  chunk_size = 8192
  buffer = io.BytesIO()

  for chunk in response.iter_content(chunk_size=chunk_size):
    if chunk:
      buffer.write(chunk)
      downloaded_size += len(chunk)

      # Berechne Fortschritt (0.0 bis 1.0)
      progress = (
          min(downloaded_size / total_size, 1.0) if total_size > 0 else 0.0
      )

      # "Yield" schickt den aktuellen Status an das Hauptskript zurück
      yield ("progress", progress, downloaded_size, total_size)

  # Wenn der Download fertig ist: DataFrame einlesen und zurückgeben
  buffer.seek(0)
  df = pd.read_csv(buffer)
  yield ("done", df)