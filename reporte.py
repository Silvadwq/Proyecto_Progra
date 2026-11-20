import csv
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

################################Menu#####################
def menu():
    print()
    print("=" * 52)
    print("|" + "SISTEMA DE ANALISIS DE TRAFICO".center(50) + "|")
    print("=" * 52)
    print("|" + " " * 50 + "|")
    print("|" + "  1.  Informe actual (alertas y graficas)".ljust(50) + "|")
    print("|" + "  2.  Prediccion de las proximas 2 horas".ljust(50) + "|")
    print("|" + "  3.  Registrar una observacion (ej. accidente)".ljust(50) + "|")
    print("|" + "  4.  Salir".ljust(50) + "|")
    print("|" + " " * 50 + "|")
    print("=" * 52)

    opcion = input("Elige una opcion (1-4): ").strip()
    while opcion not in ["1", "2", "3", "4"]:
        opcion = input("Opcion invalida, escribe un numero del 1 al 4: ").strip()
    return opcion


def pausa():
    input("\nPresiona Enter para volver al menu...")

def mostrar_opciones(lista):
    for i in range(len(lista)):
        print(f"  {i} = {lista[i]}")

#############################################################################

################################ Validaciones ################################
def validar_Datos_Mayores_0(dato):
     while dato < 0:
          dato = float(input("El dato tiene que ser mayor a 0: "))
     return dato
def validar_Hora(dato):
     while dato < 0 or dato > 23:
          dato = int(input("La hora debe estar entre 0 y 23:"))
     return dato
def validar_cruze(indice, cruces):
    while indice < 0 or indice >= len(cruces):
        indice = int(input(f"Opcion invalida, elige un numero entre 0 y {len(cruces) - 1}: "))
    return cruces[indice]

def validar_clima(indice, climas):
    while indice < 0 or indice >= len(climas):
        indice = int(input(f"Opcion invalida, elige un numero entre 0 y {len(climas) - 1}: "))
    return climas[indice]

def validar_Dia(dato):
    while dato < 0 or dato > 6:
        dato = int(input("El dia debe estar entre 0 (lunes) y 6 (domingo): "))
    return dato

def validar_Rango(dato, minimo, maximo):
    while dato < minimo or dato > maximo:
        dato = float(input(f"El dato debe estar entre {minimo} y {maximo}: "))
    return dato
################################ Fin de Validaciones ################################

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
###############################Fin modelo #######################

########################### Promedios y cosas ####################################
def clima_adverso(Base_de_Datos,hora,cruze):
     contador = 0
     promedio_precipitacion = 0
     promedio_visibilidad = 0
     for i in range(len(Base_de_Datos)):
        if hora == Base_de_Datos[i][1] and cruze == Base_de_Datos[i][3]:
            contador += 1
            promedio_precipitacion += Base_de_Datos[i][8]
            promedio_visibilidad += Base_de_Datos[i][9]
     promedio_visibilidad = promedio_visibilidad/contador
     promedio_precipitacion = promedio_precipitacion/contador

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

##########################################################################3

def informe_futuro(Base_de_Datos, modelo, dia, hora_inicial, hora_final, cruze, clima):
    print(f"Informe del dia {dia} de {hora_inicial}:00 a {hora_final}:00 en {cruze}:")
    total = 0
    for hora in range(hora_inicial, hora_final + 1):
        visibilidad, lluvia_normal = clima_adverso(Base_de_Datos, hora, cruze)
        promedio_autos, promedio_velocidad = congestionAutos(Base_de_Datos, hora, cruze)
        exp = esperado(modelo, hora, dia, lluvia_normal, clima)
        if promedio_autos > 0 and exp > promedio_autos:
            print(f"Hora {hora}: se esperan {round(exp)} carros, mas de lo usual ({promedio_autos:.0f}).")
        else:
            print(f"Hora {hora}: se esperan {round(exp)} carros, trafico normal (usual {promedio_autos:.0f}).")
        total += exp
    print(f"Total esperado de {hora_inicial}:00 a {hora_final}:00 = {round(total)} carros")

def alertas(Base_de_Datos, modelo, desviacion, visibilidadCarretera, precipitacionCarretera, congestionCarretera, velocidadCarretera, horaCarretera, diaCarretera, cruzeCarretera, climaCarretera):
    alerta = 0
    promedio_autos, promedio_velocidad = congestionAutos(Base_de_Datos, horaCarretera, cruzeCarretera)
    visibilidad, precipitacion = clima_adverso(Base_de_Datos, horaCarretera, cruzeCarretera)

    if visibilidadCarretera < visibilidad:
        alerta += 1
        print(f"Cuidado hay mas niebla de lo normal, el promedio de visivilidad es de {visibilidad:.2f}")

    if precipitacionCarretera > precipitacion:
        alerta += 1
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
        print("Trafico normal, se esperaban unos", round(exp))

    match alerta:
        case 0:
            print("Condiciones normales, mantener horario habitual del semaforo.")
        case 1:
            print("Cuidado")
        case 2:
            print("Que el semaforo dure menos en verde")
        case 3:
            print("Que el semaforo dure mas en rojo")
        case _:
            print("Alerta critica")
    return alerta

############################Grafica#############################################
def graficar_perfil_hora(Base_de_Datos, cruze, hora_actual, autos_actuales):
    horas = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    promedios = []
    for h in horas:
        prom_autos, _ = congestionAutos(Base_de_Datos, h, cruze)
        promedios.append(prom_autos)

    plt.figure(figsize=(9, 4))
    plt.plot(horas, promedios, marker='o', label='Promedio histórico')
    plt.scatter([hora_actual], [autos_actuales], color='red', s=100, zorder=5, label='Lectura actual')
    plt.title(f'Tráfico promedio por hora en {cruze}')
    plt.xlabel('Hora del día')
    plt.ylabel('Vehículos por hora')
    plt.xticks(horas)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def graficar_volumen_velocidad(Base_de_Datos, cruze):
    veh_norm, vel_norm = [], []
    veh_acc, vel_acc = [], []

    for fila in Base_de_Datos:
        if fila[3] == cruze:
            if fila[7] == 1:  # accidentes == 1
                veh_acc.append(fila[4])
                vel_acc.append(fila[5])
            else:
                veh_norm.append(fila[4])
                vel_norm.append(fila[5])

    plt.figure(figsize=(8, 5))
    plt.scatter(veh_norm, vel_norm, alpha=0.3, label='Sin accidente')
    plt.scatter(veh_acc, vel_acc, color='red', s=50, label='Con accidente')
    plt.title(f'Volumen vs. Velocidad en {cruze}')
    plt.xlabel('Vehículos por hora')
    plt.ylabel('Velocidad promedio (km/h)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
#################################################################################333


def main():
    CRUCES = ["Jl. Sudirman - Bundaran HI", "Jl. Thamrin - Monas", "Jl. Rasuna Said - Setiabudi", "Jl. Satrio - Mega Kuningan", "Jl. Jend. Gatot Subroto - Slipi"]
    CLIMAS = ["Parcialmente nublado", "Nublado", "Lluvia ligera", "Lluvia moderada", "Lluvia intensa", "Tormenta", "Niebla"]
    print("Este programa en base a una base de datos permite a un semaforo tomar decisiones")
    Base_de_Datos = cargar_datos()
    modelo, desviacion = cargar_modelo()
    observaciones=[]

    while True:
    
        #GUARDAR LA VARIABLE DEVUELTA POR LA FUNCIÓN MENU
        opcion=menu()

        while not opcion in ["1","2","3","4"]:
            print("Opcion incorrecta, ingrese de nuevo")
            opcion=menu()
        

        match opcion:

            case "1":
                
                visibilidadCarretera = float(input("Cual es la visibilidad de la carretera (0 a 15 km): "))
                visibilidadCarretera = validar_Rango(visibilidadCarretera, 0, 15)

                precipitacionCarretera = float(input("Cual es la precipitacion de la carretera (0 a 50 mm): "))
                precipitacionCarretera = validar_Rango(precipitacionCarretera, 0, 50)
                
                congestionCarretera = int(input("Cuanto coches hay en la carretera: "))
                congestionCarretera = validar_Datos_Mayores_0(congestionCarretera)

                velocidadCarretera = float(input("Cual es la velocidad promedio de los carros (km/h): "))
                velocidadCarretera = validar_Datos_Mayores_0(velocidadCarretera)

                horaCarretera = int(input("Que hora es: "))
                horaCarretera = validar_Hora(horaCarretera)
                
                diaCarretera = int(input("Que dia en numero es: "))
                diaCarretera = validar_Dia(diaCarretera)    
                
                print("Cruces disponibles:")
                mostrar_opciones(CRUCES)
                cruzeCarretera = int(input("En que cruze estas (0-4): "))
                cruzeCarretera = validar_cruze(cruzeCarretera, CRUCES)
                
                print("Climas disponibles:")
                mostrar_opciones(CLIMAS)
                climaCarretera = int(input("Que clima hay (0-6): "))
                climaCarretera = validar_clima(climaCarretera, CLIMAS)

                print()
                print("="*25)
                print("--- EVALUACION ACTUAL ---")
                nivel_alerta = alertas(
                        Base_de_Datos, modelo, desviacion,
                        visibilidadCarretera, precipitacionCarretera,
                        congestionCarretera, velocidadCarretera,
                        horaCarretera, diaCarretera, cruzeCarretera, climaCarretera
                        )
                if observaciones:
                    print("Observaciones registradas por el operador:", observaciones)
                graficar_perfil_hora(Base_de_Datos, cruzeCarretera, horaCarretera, congestionCarretera)
                graficar_volumen_velocidad(Base_de_Datos, cruzeCarretera)
                plt.show()

                pausa()

            case "2":
                horaCarretera = int(input("Que hora es: "))
                horaCarretera = validar_Hora(horaCarretera)

                diaCarretera = int(input("Que dia en numero es: "))
                diaCarretera = validar_Dia(diaCarretera)

                print("Cruces disponibles:")
                mostrar_opciones(CRUCES)
                cruzeCarretera = int(input("En que cruze estas (0-4): "))
                cruzeCarretera = validar_cruze(cruzeCarretera, CRUCES)
                
                print("Climas disponibles:")
                mostrar_opciones(CLIMAS)
                climaCarretera = int(input("Que clima hay (0-6): "))
                climaCarretera = validar_clima(climaCarretera, CLIMAS)
                

                print("\n--- INFORME FUTURO ---")
                hora_final = min(horaCarretera + 1, 23)
                informe_futuro(Base_de_Datos, modelo, diaCarretera, horaCarretera, hora_final, cruzeCarretera, climaCarretera)
                pausa()

            case "3":
                
                
                mensaje = input("Cual es el mensaje de alerta: ")

                while mensaje=="":
                    print("Error, no puede dejar el espacio en blanco")
                    mensaje = input("Cual es el mensaje de alerta: ")

                observaciones.append(mensaje)
                print(f"Alerta registrada: {mensaje}")
                print(f"Alertas reportadas por el usuario: {len(observaciones)}")

                pausa()


            case "4":
                print("PROGRAMA CERRADO")
                print("Maneje con cuidado")
                break

main()
