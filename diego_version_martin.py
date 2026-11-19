import csv
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

CLIMAS = ["Parcialmente nublado", "Nublado", "Lluvia ligera", "Lluvia moderada", "Lluvia intensa", "Tormenta", "Niebla"]
CRUCES = ["Jl. Sudirman - Bundaran HI", "Jl. Thamrin - Monas", "Jl. Rasuna Said - Setiabudi", "Jl. Satrio - Mega Kuningan", "Jl. Jend. Gatot Subroto - Slipi"]

def cargar_datos():
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
                        float(row['visibilidad_km']),
                        str(row['clima']),
                        ]
                    Base_de_Datos.append(datos)
                except (ValueError, KeyError):
                     print("No cargo bien")
    return Base_de_Datos

def dividir_datos(Base_de_Datos):
    entrenamiento = []
    prueba = []
    for fila in Base_de_Datos:
        if fila[0][3:5] == "03":
            prueba.append(fila)
        else:
            entrenamiento.append(fila)
    return entrenamiento, prueba


def menu():
    opcion=input("Ingresa la opción que deseas:\n1. Consultar cruce,\n2. Simular marzo, \n3. Salir\n") 
    return opcion

def clima_adverso(Base_de_Datos):
     promedio_precipitacion = 0
     promedio_visibilidad = 0
     for i in range(len(Base_de_Datos)):
        promedio_precipitacion += Base_de_Datos[i][8]
        promedio_visibilidad += Base_de_Datos[i][9]
     promedio_visibilidad = promedio_visibilidad/len(Base_de_Datos)
     promedio_precipitacion = promedio_precipitacion/len(Base_de_Datos)
     return promedio_visibilidad,promedio_precipitacion

def crear_perfiles(entrenamiento):
    autos = [[0] * 24 for i in range(5)]
    vel = [[0] * 24 for i in range(5)]
    cuenta = [[0] * 24 for i in range(5)]
    for fila in entrenamiento:
        c = CRUCES.index(fila[3])
        h = fila[1]
        autos[c][h] += fila[4]
        vel[c][h] += fila[5]
        cuenta[c][h] += 1
    for c in range(5):
        for h in range(24):
            if cuenta[c][h] > 0:
                autos[c][h] = autos[c][h] / cuenta[c][h]
                vel[c][h] = vel[c][h] / cuenta[c][h]
    return autos, vel


def horas_de_riesgo(entrenamiento, minimo=3):
    conteo = [0] * 24
    for fila in entrenamiento:
        conteo[fila[1]] += fila[7]
    riesgo = []
    for h in range(24):
        if conteo[h] >= minimo:
            riesgo.append(h)
    return riesgo

def revisar_reglas(hora, cruce, vehiculos, velocidad, visibilidad, lluvia, perfil_autos, perfil_vel, riesgo):
    alertas = []
    c = CRUCES.index(cruce)
    if perfil_autos[c][hora] > 0 and vehiculos > 1.3 * perfil_autos[c][hora] and velocidad < 0.7 * perfil_vel[c][hora]:
        alertas.append("congestion")
    if hora in riesgo and (velocidad > 60 or visibilidad < 2 or lluvia >= 10):
        alertas.append("riesgo")
    return alertas

def simular(prueba, perfil_autos, perfil_vel, riesgo):
    filas_alerta = 0
    acc_total = 0
    acc_con_alerta = 0
    for fila in prueba:
        alertas = revisar_reglas(fila[1], fila[3], fila[4], fila[5], fila[9], fila[8], perfil_autos, perfil_vel, riesgo)
        acc_total += fila[7]
        if len(alertas) > 0:
            filas_alerta += 1
            acc_con_alerta += fila[7]
    return filas_alerta, acc_total, acc_con_alerta

def reporte(riesgo, filas_alerta, total_filas, acc_con_alerta, acc_total):
    print("Horas de riesgo (enero-febrero):", riesgo)
    print("Filas de marzo con alerta:", filas_alerta, "de", total_filas)
    print("Accidentes de marzo en horas con alerta:", acc_con_alerta, "de", acc_total)

def es_numero(texto):
    return texto.replace(".", "", 1).isdigit()

def pedir_numero(mensaje, minimo, maximo, entero=False):
    while True:
        texto = input(mensaje).strip()
        if not es_numero(texto):
            print("Escribe un numero (sin letras ni signo negativo).")
        elif entero and "." in texto:
            print("Debe ser un numero entero.")
        elif float(texto) < minimo or float(texto) > maximo:
            print("Debe estar entre", minimo, "y", maximo)
        else:
            if entero:
                return int(texto)
            return float(texto)

def pedir_opcion(mensaje, opciones):
    for i in range(len(opciones)):
        print(i + 1, "-", opciones[i])
    while True:
        texto = input(mensaje).strip()
        if texto.isdigit() and 1 <= int(texto) <= len(opciones):
            return opciones[int(texto) - 1]
        print("Elige un numero de la lista.")

######################################      Modelo  #########################
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

def solicitar_datos(perfil_autos, perfil_vel, riesgo, modelo, desviacion):
    visibilidadCarretera = pedir_numero("Cual es la visibilidad de la carretera (km): ", 0.1, 20)
    precipitacionCarretera = pedir_numero("Cual es la precipitacion (mm): ", 0, 100)
    congestionCarretera = pedir_numero("Cuantos coches hay en la carretera: ", 0, 500, True)
    velocidadCarretera = pedir_numero("Cual es la velocidad promedio (km/h): ", 0, 150)
    horaCarretera = pedir_numero("Que hora es (0-23): ", 0, 23, True)
    diaCarretera = pedir_numero("Que dia es (0=lunes ... 6=domingo): ", 0, 6, True)
    cruzeCarretera = pedir_opcion("En que cruce estas (numero): ", CRUCES)
    climaCarretera = pedir_opcion("Que clima hay (numero): ", CLIMAS)

    alertas = revisar_reglas(horaCarretera, cruzeCarretera, congestionCarretera, velocidadCarretera, visibilidadCarretera, precipitacionCarretera, perfil_autos, perfil_vel, riesgo)
    exp = esperado(modelo, horaCarretera, diaCarretera, precipitacionCarretera, climaCarretera)
    sospechosa = congestionCarretera < exp - 2 * desviacion

    if "congestion" in alertas:
        print("Alerta de congestion: hay mas de 130% de carros y la velocidad es menor al 70% de lo usual.")
        print("Recomendacion: Extender el tiempo de luz verde (+15s) en el semaforo.")
    if "riesgo" in alertas:
        print("Riesgo preventivo: hora con accidentes previos y condiciones peligrosas.")
    if sospechosa:
        print("Lectura sospechosa, posible falla del sensor (se esperaban unos", round(exp), "carros)")
    if len(alertas) == 0 and not sospechosa:
        print("Trafico normal, se esperaban unos", round(exp), "carros")

    """velocidadCarretera = validar_Datos_Mayores_0(velocidadCarretera)
    visibilidadCarretera = validar_Datos_Mayores_0(visibilidadCarretera)
    precipitacionCarretera = validar_Datos_Mayores_0(precipitacionCarretera)
    congestionCarretera = validar_Datos_Mayores_0(congestionCarretera)
    horaCarretera = validar_Hora(horaCarretera)
    promedio_autos, promedio_velocidad = congestionAutos(entrenamiento, horaCarretera, cruzeCarretera)

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

    exp = esperado(modelo, horaCarretera, diaCarretera, precipitacionCarretera, climaCarretera)
    if congestionCarretera > exp + 2 * desviacion:
        print("Congestion anormal, se esperaban unos", round(exp))
    elif congestionCarretera < exp - 2 * desviacion:
        print("Lectura sospechosa, posible falla del sensor")
    else:
        print("Trafico normal, se esperaban unos", round(exp))"""




def main():

    print("Este programa en base a una base de datos permite a un semaforo tomar decisiones")
    Base_de_Datos = cargar_datos()
    entrenamiento, prueba = dividir_datos(Base_de_Datos)
    perfil_autos, perfil_vel = crear_perfiles(entrenamiento)
    riesgo = horas_de_riesgo(entrenamiento)
    modelo, desviacion = cargar_modelo()
    

    while True:
        option=menu()

        while option not in ["1", "2", "3"]:
            print("Opcion invalida, ingresa 1, 2 o 3")
            option=menu()

        match option:
            case "1":
                solicitar_datos(perfil_autos, perfil_vel, riesgo, modelo, desviacion)

            case "2":
                filas_alerta, acc_total, acc_con_alerta = simular(prueba, perfil_autos, perfil_vel, riesgo)
                reporte(riesgo, filas_alerta, len(prueba), acc_con_alerta, acc_total)

            case "3":
                print("Saliendo del programa")
                break
                  
    

main()