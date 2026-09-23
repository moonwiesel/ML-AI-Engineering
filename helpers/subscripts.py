import datetime

# transformiert ein geburtsjahr in das aktuell alter 
# unter Berücksichtigung des laufenden Jahres
def adjust_age(age_value, median):
  #main_year = datetime.datetime.now().year
  if age_value > 100:
    return median
  return age_value