import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

import csv


def cargar_dator():
    Base_de_Datos = []
    with open("C:\Users\diego\Desktop\proyecto_progra\trafico_5_cruces_2024Q1.csv", mode='r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                try:
                    datos = {
                        'fecha': row['fecha'],
                        'hora': int(row['hora']),
                        'dia_semana': int(row['dia_semana']),
                        'cruce': row['cruce'],
                        'vehiculos': int(row['vehiculos']),
                        'velocidad_prom': float(row['velocidad_prom']),
                        'ocupacion': float(row['ocupacion']),
                        'accidentes': int(row['accidentes']),
                        'precipitacion_mm': float(row['precipitacion_mm']),
                        'visibilidad_km': float(row['visibilidad_km']),
                        'clima':str(row['clima']),
                    }
                    Base_de_Datos.append(datos)
                except (ValueError, KeyError):
                     print("No cargo bien")
    return Base_de_Datos
def clima_adverso(Base_de_Datos):
     promedio_precipitacion = 0
     promedio_visibilidad = 0
     for i in range(len(Base_de_Datos)):
          for j in range(len(Base_de_Datos[0])):
               if i  == 8:
                    promedio_precipitacion = Base_de_Datos[i][j]
               elif i == 9:
                    promedio_visibilidad = Base_de_Datos[i][j]
     promedio_visibilidad = promedio_visibilidad/len(Base_de_Datos[0])
     promedio_precipitacion = promedio_precipitacion/len(Base_de_Datos[0])
     return promedio_visibilidad,promedio_precipitacion


def cargar_modelo():

    datos = pd.read_csv("trafico_5_cruces_2024Q1.csv")
    d_hora = pd.get_dummies(datos["hora"], prefix="hora", dtype=int).drop(columns="hora_0")
    d_dia = pd.get_dummies(datos["dia_semana"], prefix="dia", dtype=int).drop(columns="dia_0")
    d_clima = pd.get_dummies(datos["clima"], prefix="clima", dtype=int).drop(columns="clima_Parcialmente nublado")
    tabla = pd.concat([datos[["precipitacion_mm", "vehiculos"]], d_hora, d_dia, d_clima], axis=1)
    entrenamiento, test = train_test_split(tabla, test_size=0.2, random_state=42)
    variables = [c for c in entrenamiento.columns if c != "vehiculos"]
    formula = "vehiculos ~ " + " + ".join('Q("' + c + '")' for c in variables)
    modelo = smf.ols(formula=formula, data=entrenamiento).fit()
    return modelo, modelo.resid.std()
    return modelo, modelo.resid.std()


def esperado(modelo, hora, dia, lluvia, clima):
    total = modelo.params["Intercept"]
    total = total + modelo.params['Q("precipitacion_mm")'] * lluvia
    if hora != 0:
        total = total + modelo.params['Q("hora_' + str(hora) + '")']
    if dia != 0:
        total = total + modelo.params['Q("dia_' + str(dia) + '")']
    if clima != "Parcialmente nublado":
        total = total + modelo.params['Q("clima_' + clima + '")']
    return total


def main():
    print("Este programa en base a una base de datos permite a un semaforo tomar decisiones")
    Base_de_Datos = cargar_dator()
    visibilidad,precipitacion = clima_adverso(Base_de_Datos)
    modelo, desviacion = cargar_modelo()



    exp = esperado(modelo, hora, dia, precipitacionCarretera, clima)
    if vehiculosActuales > exp + 2 * desviacion:
        print("Congestion anormal, se esperaban unos", round(exp))
    elif vehiculosActuales < exp - 2 * desviacion:
        print("Lectura sospechosa, posible falla del sensor")
    else:
        print("Trafico normal, se esperaban unos", round(exp))


main()




