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
    
     

def main():
    print("Este programa en base a una base de datos permite a un semaforo tomar decisiones")
    Base_de_Datos = cargar_dator()
    visibilidad,precipitacion = clima_adverso(Base_de_Datos)
