import csv

def cargar_dator():
    Base_de_Datos = []
    with open("trafico_5_cruces_2024Q1.csv", mode='r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                try:
                    datos = [
                        row['fecha'],
                        int(row['hora']),
                        int(row['dia_semana']),
                        row['cruce'],
                        int(row['vehiculos']),
                        float(row['velocidad_prom']),
                        float(row['ocupacion']),
                        int(row['accidentes']),
                        float(row['precipitacion_mm']),
                        float(row['visibilidad_km'])
                        ]
                    Base_de_Datos.append(datos)
                except (ValueError, KeyError):
                     print("No cargo bien")
    return Base_de_Datos

def clima_adverso(Base_de_Datos):
     promedio_precipitacion = 0
     promedio_visibilidad = 0
     for i in range(len(Base_de_Datos)):
        promedio_precipitacion += Base_de_Datos[i][8]
        promedio_visibilidad += Base_de_Datos[i][9]
     promedio_visibilidad = promedio_visibilidad/len(Base_de_Datos)
     promedio_precipitacion = promedio_precipitacion/len(Base_de_Datos)
     return promedio_visibilidad,promedio_precipitacion
    
     

def main():
    print("Este programa en base a una base de datos permite a un semaforo tomar decisiones")
    Base_de_Datos = cargar_dator()
    visibilidad,precipitacion = clima_adverso(Base_de_Datos)

    visibilidadCarretera = float(input("Cual es la visibilidad de la carretera: "))
    precipitacionCarretera = float(input("Cual es la precipitacion de la carretera: "))

    if visibilidadCarretera < visibilidad:
         print(f"Cuidado hay mas niebla de lo normal, el promedio es de {visibilidad}")
    if precipitacionCarretera > precipitacion:
         print(f"Cuidado esta lloviendo mas fuerte de lo normal, el promedio es de {precipitacion} ")
main()