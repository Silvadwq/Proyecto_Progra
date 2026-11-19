"""Panel de tráfico — versión comentada para revisión de Diego.

Ejecutar: python -m pip install streamlit pandas numpy statsmodels plotly
          python -m streamlit run interfaz_trafico.py
El CSV original debe estar junto a este archivo. No importa los scripts anteriores
porque ejecutan main() y solicitan input al importarse.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DIAS = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
CLIMAS = ['Parcialmente nublado', 'Nublado', 'Lluvia ligera', 'Lluvia moderada',
          'Lluvia intensa', 'Tormenta', 'Niebla']
BASE = Path(__file__).resolve().parent
# Tratamientos explícitos equivalentes a las variables dummy del repositorio.
HORA = 'C(hora, Treatment(reference=0))'
DIA = 'C(dia_semana, Treatment(reference=0))'
CLIMA = 'C(clima, Treatment(reference="Parcialmente nublado"))'
COMUN = f'precipitacion_mm + {HORA} + {DIA} + {CLIMA}'
FORMULAS = {
    'OLS reporte': 'vehiculos ~ ' + COMUN,
    # En regresion.py el ajuste final omite hora_5 y dia_3: se conserva exactamente
    # esa codificación; esas categorías comparten la referencia del intercepto.
    'OLS regresión (ocupación observada)': 'vehiculos ~ ocupacion + ' + COMUN,
    'OLS por cruce': 'vehiculos ~ ' + COMUN + ' + C(cruce)',
}


def validar_datos(raw):
    """Excluir lecturas inválidas y duplicados sin corregirlas silenciosamente."""
    d = raw.copy()
    requeridas = ['fecha', 'hora', 'dia_semana', 'cruce', 'sensor', 'vehiculos',
                  'velocidad_prom', 'ocupacion', 'accidentes', 'precipitacion_mm',
                  'visibilidad_km', 'clima', 'es_no_laborable']
    faltan = set(requeridas) - set(d.columns)
    if faltan:
        raise ValueError('Faltan columnas: ' + ', '.join(sorted(faltan)))
    d['fecha'] = pd.to_datetime(d.fecha, format='%d/%m/%Y', errors='coerce')
    rangos = {'hora': (0, 23), 'dia_semana': (0, 6), 'vehiculos': (0, np.inf),
              'velocidad_prom': (0, 150), 'ocupacion': (0, 1), 'accidentes': (0, np.inf),
              'precipitacion_mm': (0, np.inf), 'visibilidad_km': (0, np.inf)}
    valida = d.fecha.between('2024-01-01', '2024-03-31')
    for c, (menor, mayor) in rangos.items():
        d[c] = pd.to_numeric(d[c], errors='coerce')
        valida &= np.isfinite(d[c]) & d[c].between(menor, mayor)
    for c in ['hora', 'dia_semana', 'vehiculos', 'accidentes']:
        valida &= d[c].mod(1).eq(0)
    d['es_no_laborable'] = d.es_no_laborable.astype(str).str.lower().map({'true': True, 'false': False})
    valida &= d.es_no_laborable.notna() & d.clima.isin(CLIMAS)
    valida &= d.cruce.notna() & d.sensor.notna()
    valida &= d.dia_semana.eq(d.fecha.dt.dayofweek)
    invalidas = int((~valida).sum())
    d = d.loc[valida].copy()
    duplicadas = int(d.duplicated(['fecha', 'hora', 'sensor']).sum())
    d = d.drop_duplicates(['fecha', 'hora', 'sensor']).sort_values(['fecha', 'hora', 'cruce'])
    d['instante'] = d.fecha + pd.to_timedelta(d.hora, unit='h')
    # Comprobar integridad del calendario sin inventar lecturas faltantes.
    esperadas = 91 * 24 * d.sensor.nunique()
    return d.reset_index(drop=True), {'inválidas': invalidas, 'duplicadas': duplicadas,
                                    'faltantes': max(0, esperadas - len(d))}


def dividir(d):
    """La evaluación es temporal: marzo nunca participa en los ajustes."""
    train = d[d.fecha < '2024-03-01'].copy()
    test = d[d.fecha >= '2024-03-01'].copy()
    if train.empty or test.empty:
        raise ValueError('Se necesitan enero-febrero y marzo para la evaluación temporal.')
    return train, test


def preparar_regresion(d):
    """Reproducir la selección final de regresion.py sin ejecutar ese script."""
    r = d.copy()
    r['hora'] = r.hora.where(r.hora.ne(5), 0)
    r['dia_semana'] = r.dia_semana.where(r.dia_semana.ne(3), 0)
    return r


def entrenar(train):
    modelos = {}
    for nombre, formula in FORMULAS.items():
        entrada = preparar_regresion(train) if 'ocupación' in nombre else train
        modelos[nombre] = smf.ols(formula, data=entrada).fit()
    return modelos


def predecir(modelo, datos, nombre):
    entrada = preparar_regresion(datos) if 'ocupación' in nombre else datos
    # obs_ci es un intervalo para una lectura, no el intervalo de la media.
    p = modelo.get_prediction(entrada).summary_frame(alpha=0.05)
    return pd.DataFrame({'estimado': p['mean'].to_numpy().clip(0),
                         'inferior': p.obs_ci_lower.to_numpy().clip(0),
                         'superior': p.obs_ci_upper.to_numpy().clip(0)}, index=datos.index)


def referencia(train, datos):
    """Perfil por cruce/día/hora con retroceso al perfil cruce/hora y media."""
    perfil = train.groupby(['cruce', 'dia_semana', 'hora']).vehiculos.mean()
    respaldo = train.groupby(['cruce', 'hora']).vehiculos.mean()
    return np.array([perfil.get((r.cruce, r.dia_semana, r.hora),
                     respaldo.get((r.cruce, r.hora), train.vehiculos.mean()))
                     for r in datos.itertuples()])


def comparar(train, test, modelos):
    pred = {'Perfil histórico': referencia(train, test)}
    pred.update({n: predecir(m, test, n).estimado.to_numpy() for n, m in modelos.items()})
    filas = []
    y = test.vehiculos.to_numpy()
    for nombre, p in pred.items():
        error = y - p
        denom = ((y - y.mean()) ** 2).sum()
        filas.append({'Modelo': nombre, 'MAE': np.abs(error).mean(),
                      'RMSE': np.sqrt((error ** 2).mean()),
                      'R²': 1 - (error ** 2).sum() / denom if denom else np.nan})
    return pd.DataFrame(filas), pd.DataFrame(pred, index=test.index)


def tasa(d, grupos):
    r = d.groupby(grupos, observed=True).agg(accidentes=('accidentes', 'sum'),
                                           vehiculos=('vehiculos', 'sum')).reset_index()
    r['Tasa por mil vehículos'] = r.accidentes.div(r.vehiculos.replace(0, np.nan)) * 1000
    return r


def simular(train, test):
    """Reglas de la rama Martin_Cambio; coincidencia en la misma hora, no anticipación."""
    perfil = train.groupby(['cruce', 'dia_semana', 'hora'])[['vehiculos', 'velocidad_prom']].mean()
    s = test.join(perfil, on=['cruce', 'dia_semana', 'hora'], rsuffix='_tipico').copy()
    riesgo = train.groupby('hora').accidentes.sum()
    s['congestion'] = ((s.vehiculos > 1.3 * s.vehiculos_tipico) &
                       (s.velocidad_prom < 0.7 * s.velocidad_prom_tipico))
    s['riesgo'] = s.hora.isin(riesgo[riesgo >= 3].index) & (
        (s.velocidad_prom > 60) | (s.visibilidad_km < 2) | (s.precipitacion_mm >= 10))
    s['alerta'] = s.congestion | s.riesgo
    s['motivo'] = np.select([s.congestion & s.riesgo, s.congestion, s.riesgo],
                           ['Congestión y riesgo', 'Congestión', 'Riesgo preventivo'], default='Sin alerta')
    s['recomendacion'] = np.select([s.riesgo, s.congestion],
        ['Revisión del operador por condiciones de riesgo',
         'Evaluar +15 s de verde con el operador'], default='Mantener plan habitual')
    return s


def main():
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go

    st.set_page_config(page_title='NEXO · Tráfico', page_icon='◉', layout='wide')
    st.markdown('''<style>
    .stApp{background:#0b101b;color:#e9edf5} [data-testid="stSidebar"]{background:#111927}
    .block-container{padding-top:5rem;max-width:1500px}
    h1,h2,h3{letter-spacing:-.035em} h1{font-weight:750!important}
    [data-testid="stMetric"]{background:#151f30;border:1px solid #26344a;
      border-radius:16px;padding:18px} [data-testid="stMetricValue"]{color:#72e5c0}
    .eyebrow{color:#72e5c0;font-size:12px;letter-spacing:.2em;font-weight:700}
    </style>''', unsafe_allow_html=True)
    px.defaults.template = 'plotly_dark'
    px.defaults.color_discrete_sequence = ['#72e5c0', '#809bff', '#ffbe76', '#ff7d91', '#c1a4ff']

    @st.cache_data
    def cargar():
        return validar_datos(pd.read_csv(BASE / 'trafico_5_cruces_2024Q1.csv'))

    @st.cache_resource
    def ajustar(_train, firma):
        return entrenar(_train)

    def chart(fig, key=None):
        fig.update_layout(paper_bgcolor='#0b101b', plot_bgcolor='#111927',
                          font_color='#dce5f4', margin=dict(t=45, b=35, l=30, r=20),
                          legend_title_text='', hovermode='closest')
        st.plotly_chart(fig, use_container_width=True, key=key)

    def descargar(d, nombre, etiqueta='Descargar CSV'):
        st.download_button(etiqueta, d.to_csv(index=False).encode('utf-8-sig'),
                           nombre, 'text/csv', key=nombre)

    try:
        datos, calidad = cargar()
        train, test = dividir(datos)
        modelos = ajustar(train, pd.util.hash_pandas_object(train, index=True).sum())
    except (ValueError, FileNotFoundError) as e:
        st.error(str(e))
        st.stop()
    with st.sidebar:
        st.markdown('## ◉ NEXO')
        st.caption('LABORATORIO DE MOVILIDAD')
        pagina = st.radio('Explorar', ['Panorama', 'Comparar modelos', 'Predicción', 'Alertas y revisión'])
        st.divider()
        cruce = st.selectbox('Cruce', ['Todos'] + sorted(datos.cruce.unique()))
        periodo = st.selectbox('Periodo de análisis', ['Todo el trimestre', 'Enero–febrero', 'Marzo'])
        st.caption('5 cruces · Yakarta · 2024 Q1\n\nDatos sintéticos • Sin sensores en vivo')
    d = datos if periodo == 'Todo el trimestre' else train if periodo == 'Enero–febrero' else test
    if cruce != 'Todos':
        d = d[d.cruce.eq(cruce)]
    st.markdown('<div class="eyebrow">MOVILIDAD / INTELIGENCIA DE TRÁFICO</div>', unsafe_allow_html=True)
    st.title({'Panorama': 'Cada cruce cuenta.', 'Comparar modelos': 'Medir antes de predecir.',
              'Predicción': 'Anticipa el próximo escenario.', 'Alertas y revisión': 'Decisiones con contexto.'}[pagina])
    st.caption('Enero–marzo 2024 · Análisis histórico, evaluación temporal y escenarios de tráfico')

    if pagina == 'Panorama':
        c = st.columns(4)
        for col, label, value in zip(c, ['Vehículos registrados', 'Velocidad media', 'Accidentes', 'Lecturas'],
            [f'{d.vehiculos.sum():,}', f'{d.velocidad_prom.mean():.1f} km/h',
             str(d.accidentes.sum()), f'{len(d):,}']):
            col.metric(label, value)
        st.subheader('El pulso de los cruces')
        volumen, seguridad, condiciones = st.tabs(['Volumen y velocidad', 'Accidentes', 'Clima y calendario'])
        with volumen:
            h = d.groupby(['hora', 'cruce'])[['vehiculos', 'velocidad_prom']].mean().reset_index()
            chart(px.line(h, x='hora', y='vehiculos', color='cruce', markers=True,
                          labels={'vehiculos': 'Vehículos por hora (media)', 'hora': 'Hora'}, title='Volumen por cruce'))
            chart(px.line(h, x='hora', y='velocidad_prom', color='cruce',
                          labels={'velocidad_prom': 'Velocidad media (km/h)', 'hora': 'Hora'}, title='Velocidad por cruce'))
            puntos = d.assign(Estado=np.where(d.accidentes > 0, 'Con accidente', 'Sin accidente'))
            chart(px.scatter(puntos, x='vehiculos', y='velocidad_prom', color='Estado', opacity=.45,
                             hover_data=['fecha', 'hora', 'cruce'], labels={'vehiculos': 'Vehículos/h',
                             'velocidad_prom': 'Velocidad (km/h)'}, title='Volumen frente a velocidad'))
        with seguridad:
            mapa = d.pivot_table(index='dia_semana', columns='hora', values='accidentes', aggfunc='sum').reindex(index=range(7), columns=range(24), fill_value=0).fillna(0)
            chart(px.imshow(mapa.to_numpy(), x=list(range(24)), y=DIAS, color_continuous_scale='Mint',
                            labels={'x': 'Hora', 'y': 'Día', 'color': 'Accidentes'}, title='Accidentes · día y hora'))
            franjas = d.assign(Franja=pd.cut(d.hora, [-1, 5, 11, 17, 23], labels=['00–05', '06–11', '12–17', '18–23']))
            chart(px.bar(tasa(franjas, ['Franja']), x='Franja', y='Tasa por mil vehículos',
                         hover_data=['accidentes', 'vehiculos'], title='Tasa de accidentes por franja horaria'))
        with condiciones:
            v = d.assign(Visibilidad=pd.cut(d.visibilidad_km, [-np.inf, 2, 5, np.inf],
                                           labels=['Baja (≤2 km)', 'Media (2–5 km)', 'Alta (>5 km)']))
            chart(px.bar(tasa(v, ['clima', 'Visibilidad']), x='clima', y='Tasa por mil vehículos',
                         color='Visibilidad', barmode='group', hover_data=['accidentes', 'vehiculos'],
                         title='Clima, visibilidad y tasa de accidentes'))
            labor = d.assign(Jornada=np.where(d.es_no_laborable, 'No laborable', 'Laborable'))
            perfil = labor.groupby(['hora', 'Jornada']).vehiculos.mean().reset_index()
            chart(px.line(perfil, x='hora', y='vehiculos', color='Jornada', markers=True,
                          title='Días laborables frente a no laborables', labels={'vehiculos': 'Vehículos/h (media)'}))
        descargar(d, 'datos_filtrados.csv')

    elif pagina == 'Comparar modelos':
        st.info('Todos los modelos se ajustan con enero–febrero y se evalúan en marzo. Los filtros de periodo no alteran esta separación.')
        t = test if cruce == 'Todos' else test[test.cruce.eq(cruce)]
        metricas, predicciones = comparar(train, t, modelos)
        chart(px.bar(metricas, x='Modelo', y='MAE', color='Modelo', title='Error absoluto medio · menor es mejor'))
        st.dataframe(metricas.round(3), use_container_width=True, hide_index=True)
        st.caption('MAE y RMSE en vehículos por lectura horaria. R² puede ser negativo. La variante con ocupación usa un dato de la misma hora: mide estimación contemporánea, no un pronóstico autónomo.')
        nombre = st.selectbox('Modelo para inspeccionar', list(predicciones.columns))
        detalle = t[['fecha', 'instante', 'cruce', 'vehiculos']].copy()
        detalle['Estimado'] = predicciones[nombre]
        diario = detalle.groupby('fecha')[['vehiculos', 'Estimado']].mean().reset_index()
        chart(px.line(diario, x='fecha', y=['vehiculos', 'Estimado'], markers=True,
                      title='Marzo · observado y estimado (media por lectura)', labels={'value': 'Vehículos/h'}))
        chart(px.scatter(detalle, x='vehiculos', y='Estimado', opacity=.35,
                         title='Observado frente a estimado', hover_data=['instante', 'cruce']))
        descargar(metricas, 'comparacion_modelos.csv')
        descargar(detalle, 'predicciones_marzo.csv')

    elif pagina == 'Predicción':
        st.caption('Escenario de 24 horas con lluvia y clima definidos por ti. Los modelos permanecen ajustados con enero–febrero.')
        a, b, c = st.columns(3)
        with a:
            elegido = st.selectbox('Cruce del escenario', sorted(datos.cruce.unique()),
                                   index=sorted(datos.cruce.unique()).index(cruce) if cruce != 'Todos' else 0)
            fecha = st.date_input('Fecha del escenario', pd.Timestamp('2024-03-01').date(),
                                  min_value=pd.Timestamp('2024-03-01').date())
        with b:
            clima = st.selectbox('Clima previsto', CLIMAS)
            lluvia = st.number_input('Precipitación prevista (mm)', min_value=0.0,
                                     max_value=float(train.precipitacion_mm.max()), value=0.0, step=.1)
        with c:
            nombre = st.selectbox('Modelo del escenario', list(modelos), index=2)
            ocupacion = st.slider('Ocupación supuesta', 0.0, 1.0, .25, .01) if 'ocupación' in nombre else 0.0
        if pd.Timestamp(fecha) > datos.fecha.max():
            st.warning('Fecha fuera del trimestre disponible: no hay validación de estabilidad del modelo para este periodo.')
        if 'ocupación' in nombre:
            st.info('Esta ocupación se supone constante durante el día. Cambiarla altera el escenario; no es una lectura futura conocida.')
        escenario = pd.DataFrame({'hora': range(24), 'dia_semana': pd.Timestamp(fecha).dayofweek,
                                  'cruce': elegido, 'clima': clima, 'precipitacion_mm': lluvia, 'ocupacion': ocupacion})
        p = predecir(modelos[nombre], escenario, nombre)
        resultado = pd.concat([escenario, p], axis=1)
        resultado['Perfil histórico'] = referencia(train, escenario)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=resultado.hora, y=resultado.superior, line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=resultado.hora, y=resultado.inferior, fill='tonexty',
                      fillcolor='rgba(114,229,192,.15)', line=dict(width=0), name='Intervalo de predicción 95%'))
        fig.add_trace(go.Scatter(x=resultado.hora, y=resultado.estimado, name='Predicción', line=dict(color='#72e5c0', width=3)))
        fig.add_trace(go.Scatter(x=resultado.hora, y=resultado['Perfil histórico'], name='Perfil histórico',
                                line=dict(color='#809bff', dash='dot')))
        fig.update_layout(title='El día, hora por hora', xaxis_title='Hora', yaxis_title='Vehículos/h')
        chart(fig)
        c = st.columns(3)
        c[0].metric('Volumen diario estimado', f'{resultado.estimado.sum():,.0f}')
        c[1].metric('Hora de mayor volumen', f'{int(resultado.estimado.idxmax()):02d}:00')
        c[2].metric('Pico estimado', f'{resultado.estimado.max():.0f} vehículos/h')
        st.caption('Intervalos nominales OLS para una lectura bajo supuestos de errores independientes, homocedásticos y aproximadamente normales. Se recortan valores negativos a cero; la cobertura real no está garantizada. El modelo OLS reporte no distingue cruces.')
        st.dataframe(resultado.round(2), use_container_width=True, hide_index=True)
        descargar(resultado, 'escenario_prediccion.csv')

    else:
        s = simular(train, test)
        if cruce != 'Todos':
            s = s[s.cruce.eq(cruce)]
        total = int(s.accidentes.sum())
        coinciden = int(s.loc[s.alerta, 'accidentes'].sum())
        c = st.columns(3)
        c[0].metric('Lecturas con alerta · marzo', int(s.alerta.sum()))
        c[1].metric('Accidentes en horas con alerta', f'{coinciden} / {total}')
        c[2].metric('Coincidencia horaria', f'{coinciden / total:.1%}' if total else 'Sin accidentes')
        st.caption('Las alertas usan lecturas de la misma hora. Esta coincidencia no demuestra que se anticiparon o evitaron accidentes. Son recomendaciones para un operador humano.')
        conteo = s.groupby('cruce')[['congestion', 'riesgo']].sum().reset_index()
        chart(px.bar(conteo, x='cruce', y=['congestion', 'riesgo'], barmode='group', title='Alertas por tipo y cruce'))
        st.dataframe(s[['instante', 'cruce', 'vehiculos', 'velocidad_prom', 'motivo', 'recomendacion', 'accidentes']],
                     use_container_width=True, hide_index=True)
        descargar(s, 'bitacora_marzo.csv')
        with st.expander('Calidad de los datos y revisión de Diego', expanded=True):
            st.write({'lecturas_válidas': len(datos), **calidad})
            st.markdown('''**Para revisar:** selección de variables, intervalos OLS, umbrales de congestión
            (volumen >130% y velocidad <70% del perfil), horas con ≥3 accidentes en entrenamiento
            y las recomendaciones de semáforo. Los perfiles se calculan por cruce/día/hora.
            Los datos son sintéticos; el panel reproduce históricos y no opera infraestructura real.''')
            st.write('Entrenamiento:', len(train), 'lecturas. Evaluación:', len(test), 'lecturas.')
        nombre = st.selectbox('Resumen estadístico OLS', list(modelos))
        st.code(modelos[nombre].summary().as_text(), language='text')


if __name__ == '__main__':
    main()
