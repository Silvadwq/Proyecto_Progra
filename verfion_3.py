"""
Hecho por Diego Yael Silva(A01649786), Martín Alejandro Mosquera(A01985797), José Manuel Ramirez(A01649926)



"""
#Importar las librerias necesarias que no están integradas en Python
#Librerias para leer archivos csv, manipular datos y hacer regresion lineal
import csv
#Libreria para manipular datos, útil para crear el modelo de regresión  lineal
import pandas as pd
#Libreria para hacer regresion lineal
import statsmodels.formula.api as smf
#Libreria para dividir los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split



#Función para cargar los datos del archivo csv y guardarlos en una lista de listas.
#Cada lista representa una fila del archiv csv con sus respectivas columnas.
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


#Función encargada de calcular el promedio del precipitación y visibilidad de cada hora.
#La función recibe la base de datos, la hroa y el cruce que son los parámetros que permiten hallar el dato y que no se mezcle con otros, o que el programa no se confunda.
#El for simplemente recorre cada fila de la base de datos y toma en cuanto los valores cuya hora y cruce coincidad con los que el usuario haya ingresado.
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


#Función encargada de calcular el promedio de autos y de velocidad_promedio por cada hora.
#Al igual que la función anterior, recibe la base de datos, la hora y el cruce para poder calcular los promedios de cada hora y cruce.
def congestionAutos(Base_de_Datos,hora,cruze):
    promedio_autos = 0
    promedio_velocidad = 0
    #El contador permite obtener el número de datos últiles(n) para así obtener (#datos), que divide a la suma total.
    contador = 0
    for i in range(len(Base_de_Datos)):
        if hora == Base_de_Datos[i][1] and cruze == Base_de_Datos[i][3]:
            promedio_autos += Base_de_Datos[i][4]
            promedio_velocidad += Base_de_Datos[i][5]
            contador += 1
    #Si el contador se mantiene 0 significa que no se hallo esa combinación y por lo tnato ese cruce no existe.
    if contador == 0:
         print("No se encontro ese cruze")
         return 0,0
    promedio_velocidad = promedio_velocidad/contador
    promedio_autos = promedio_autos / contador
    return promedio_autos, promedio_velocidad


#Función encarga de validar si los los datos ingresados por el usuario son mayores a 0, si no lo son, se le pedirá al usuario que ingrese un nuevo dato.
def validar_Datos_Mayores_0(dato):
     while dato < 0:
          dato = float(input("El dato tiene que ser mayor a 0: "))
     return dato

#Verificar si la hora ingresa es mayor a 0  y menor que 23, sino crear un bucle que obligue al usuario ingresar un dato en el rango aceptado
def validar_Hora(dato):
     while dato < 0 or dato > 23:
          dato = int(input("La hora debe estar entre 0 y 23:"))
     return dato

#Verficar si el cruce ingresado es uno de los 5 cruces y además que la opción ingresada sea un número.
def validar_cruze(dato):
    cruces = ["Jl. Sudirman - Bundaran HI", "Jl. Thamrin - Monas", "Jl. Rasuna Said - Setiabudi", "Jl. Satrio - Mega Kuningan", "Jl. Jend. Gatot Subroto - Slipi"]
    while dato not in cruces or not dato.isdigit():
        dato = input("El cruce es incorrecto, ingrésalo de nuevo: ")
    return dato

#Verficar si el clima que se ingresa está en las opciones y que la opción ingresada sea un número.
def validar_clima(dato):
    climas = ["Parcialmente nublado", "Nublado", "Lluvia ligera", "Lluvia moderada", "Lluvia intensa", "Tormenta", "Niebla"]
    while dato not in climas or not dato.isdigit():
        dato = input("El clima es incorrecto, ingrésalo de nuevo: ")
    return dato

#Verifficar si el día que se ingresa es una opción válida.
def validar_Dia(dato):
    while dato < 0 or dato > 6:
        dato = int(input("El dia debe estar entre 0 (lunes) y 6 (domingo): "))
    return dato


#Función encarga de cargar el modelo de regresión lineal

def cargar_modelo():

    #Importar la base de datos
    datos = pd.read_csv("trafico_5_cruces_2024Q1.csv")
    #Horas, día, clima es una variable categórica por lo tanto se lo debe convertir en una dicotómica para que se pueda ingresar en el modelo.
    d_hora = pd.get_dummies(datos["hora"], prefix="hora", dtype=int).drop(columns="hora_0")
    d_dia = pd.get_dummies(datos["dia_semana"], prefix="dia", dtype=int).drop(columns="dia_0")
    d_clima = pd.get_dummies(datos["clima"], prefix="clima", dtype=int).drop(columns="clima_Parcialmente nublado")
    #Concatenar las columnas de las variables numéricas con las dummies(dicotómicas) para crear la tabla final que se usará en el modelo.
    tabla = pd.concat([datos[["precipitacion_mm", "vehiculos"]], d_hora, d_dia, d_clima], axis=1)
    #Dividor la base de datos en entrenamiento y prueba, para poder evaluar el modelo.
    entrenamiento, test = train_test_split(tabla, test_size=0.2, random_state=42)
    variables = [c for c in entrenamiento.columns if c != "vehiculos"]
    #Unificar la formula para poder ingresarla en el modelo de regresión lineal.
    formula = "vehiculos ~ " + " + ".join('Q("' + c + '")' for c in variables)
    #Implementar el modelo de regresión lineal con la base de datos de entrenamientno.
    modelo = smf.ols(formula=formula, data=entrenamiento).fit()
    #Devolver el modelo y las desviacioens estándares del modelo para poder evaluarlo.
    return modelo, modelo.resid.std()


#Función encargada de responder la siguiente prefunta: ¿cuántos carros deberían pasar con esta hora, este día, esta lluvia y este clima?
def esperado(modelo, hora, dia, lluvia, clima):
    #Para poder calcular el número de carros totales empieza desde el número base(en este caso el intercepto) y luego se le van sumando los coeficiente multilpicado por su respectivo argumento (hora, dia, lluvia, clima).
   
#Guardar el modelo con sus coeficientes en un modelo.params para poder utilizar sus coeficientes y el intercepto.
    total = modelo.params["Intercept"]
    
    #Sumar el coeficiente de la lluvia por la cantidad de lluvia que se ingresa.
    total = total + modelo.params['Q("precipitacion_mm")'] * lluvia

    #Si la hora es diferente de 0, se multiplica el valor ingresado por su respecitvo cofientes y se le añade el total.
    if hora != 0:
        total = total + modelo.params['Q("hora_' + str(hora) + '")']

    #Si el día es diferente de 0, se multiplica el valor ingresado por su respecitvo cofientes y se le añade el total.
    if dia != 0:
        total = total + modelo.params['Q("dia_' + str(dia) + '")']

    #Si el clima es diferente de 0, se multiplica el valor ingresado por su respecitvo cofientes y se le añade el total
    if clima != "Parcialmente nublado":
        total = total + modelo.params['Q("clima_' + clima + '")']

    #Se devuelve el valor total calculado.
    return total


###############################Fin modelo #######################
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

def main():
    print("Este programa en base a una base de datos permite a un semaforo tomar decisiones")
    Base_de_Datos = cargar_datos()
    modelo, desviacion = cargar_modelo()

    visibilidadCarretera = float(input("Cual es la visibilidad de la carretera: "))
    precipitacionCarretera = float(input("Cual es la precipitacion de la carretera: "))
    congestionCarretera = int(input("Cuanto coches hay en la carretera: "))
    velocidadCarretera = float(input("Cual es la velocidad promedio de los carros (km/h): "))
    horaCarretera = int(input("Que hora es: "))
    diaCarretera = int(input("Que dia en numero es: "))
    cruzeCarretera = input("En que cruze estas: ")
    climaCarretera = input("Que clima hay: ")

    velocidadCarretera = validar_Datos_Mayores_0(velocidadCarretera)
    visibilidadCarretera = validar_Datos_Mayores_0(visibilidadCarretera)
    precipitacionCarretera = validar_Datos_Mayores_0(precipitacionCarretera)
    congestionCarretera = validar_Datos_Mayores_0(congestionCarretera)
    horaCarretera = validar_Hora(horaCarretera)
    cruzeCarretera = validar_cruze(cruzeCarretera)
    climaCarretera = validar_clima(climaCarretera)
    diaCarretera = validar_Dia(diaCarretera)

    print("\n--- EVALUACION ACTUAL ---")
    nivel_alerta = alertas(
        Base_de_Datos, modelo, desviacion,
        visibilidadCarretera, precipitacionCarretera,
        congestionCarretera, velocidadCarretera,
        horaCarretera, diaCarretera, cruzeCarretera, climaCarretera
    )

    print("\n--- INFORME FUTURO ---")
    hora_final = min(horaCarretera + 1, 23)
    informe_futuro(Base_de_Datos, modelo, diaCarretera, horaCarretera, hora_final, cruzeCarretera, climaCarretera)

main()