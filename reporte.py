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

def congestionAutos(Base_de_Datos,hora,cruze):
    promedio_autos = 0
    promedio_velocidad = 0
    contador = 0
    for i in range(len(Base_de_Datos)):
        if hora == Base_de_Datos[i][1] and cruze == Base_de_Datos[i][3]:
            promedio_autos += Base_de_Datos[i][4]
            promedio_velocidad += Base_de_Datos[i][5]
            contador += 1
    if contador == 0:
         print("No se encontro ese cruze")
         return 0,0
    promedio_velocidad = promedio_velocidad/contador
    promedio_autos = promedio_autos / contador
    return promedio_autos, promedio_velocidad

def validar_Datos_Mayores_0(dato):
     while dato < 0:
          dato = float(input("El dato tiene que ser mayor a 0: "))
     return dato
def validar_Hora(dato):
     while dato < 0 or dato > 23:
          dato = int(input("El dato tiene que ser mayor a 0: "))
     return dato
     

def main():
    print("Este programa en base a una base de datos permite a un semaforo tomar decisiones")
    Base_de_Datos = cargar_dator()
    visibilidad,precipitacion = clima_adverso(Base_de_Datos)
    
    visibilidadCarretera = float(input("Cual es la visibilidad de la carretera: "))
    precipitacionCarretera = float(input("Cual es la precipitacion de la carretera: "))
    congestionCarretera = int(input("Cuanto coches hay en la carretera: "))
    velocidadCarretera = float(input("Cual es la velocidad promedio de los carros (km/h): "))
    horaCarretera = int(input("Que hora es: "))
    cruzeCarretera = input("En que cruze estas: ")

    velocidadCarretera = validar_Datos_Mayores_0(velocidadCarretera)
    visibilidadCarretera = validar_Datos_Mayores_0(visibilidadCarretera)
    precipitacionCarretera = validar_Datos_Mayores_0(precipitacionCarretera)
    congestionCarretera = validar_Datos_Mayores_0(congestionCarretera)
    horaCarretera = validar_Hora(horaCarretera)

    promedio_autos, promedio_velocidad = congestionAutos(Base_de_Datos, horaCarretera, cruzeCarretera)

    if visibilidadCarretera < visibilidad:
         print(f"Cuidado hay mas niebla de lo normal, el promedio de visivilidad es de {visibilidad:.2f}")
    if precipitacionCarretera > precipitacion:
         print(f"Cuidado esta lloviendo mas fuerte de lo normal, el promedio es de {precipitacion:.2f} ")

    if promedio_autos > 0:
        if congestionCarretera > promedio_autos and velocidadCarretera < promedio_velocidad:
            print(f"Alerta de congestion: hay mas carros de lo normal ({promedio_autos:.1f}) y van mas lento de lo usual ({promedio_velocidad:.1f} km/h).")
            print("Recomendacion: Extender el tiempo de luz verde (+15s) en el semaforo.")
        elif congestionCarretera > promedio_autos:
            print(f"Hay mas carros de lo normal ({promedio_autos:.1f}), pero el trafico fluye a buena velocidad.")
main()