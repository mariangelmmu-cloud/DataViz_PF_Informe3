# app_dash_entrega2.py
import dash
from dash import dcc, html, Input, Output, dash_table, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import io, base64
import joblib 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np


# -------------------------------------
# Configuración general (igual estilo a codigo_juan.txt)
# -------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE], suppress_callback_exceptions=True)
app.title = "Dashboard de Riesgo de Readmisión"
server = app.server

# -------------------------------------
# Cargar datos (viene de codigo_entrega2.txt)
# -------------------------------------
# Asegúrate de tener data_nombres.csv en la misma carpeta
df = pd.read_csv("data_nombres.csv", sep=",")

# Intentos de conversión para columnas numéricas si aplica
for col in ["time_in_hospital", "num_medications", "number_diagnoses", "number_outpatient", "number_emergency", "number_inpatient"]:
    if col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except:
            pass

# Detectar columnas numéricas y categóricas
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Excluir variables no deseadas
excluir = ["encounter_id", "patient_nbr"]  # cambia por las que quieras
num_cols = [col for col in num_cols if col not in excluir]
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# ====================================================
# Cargar modelos con tolerancia a errores
# ====================================================
import joblib, sys, types, pickle

def safe_joblib_load(path):
    """Carga archivos .pkl ignorando errores de import (ej. 'No module named modelo')."""
    try:
        if "modelo" not in sys.modules:
            sys.modules["modelo"] = types.ModuleType("modelo")
        return joblib.load(path)
    except Exception as e:
        print(f"⚠️ No se pudo cargar completamente {path}: {e}")
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e2:
            print(f"❌ Error total al abrir {path}: {e2}")
            return None


modelos_pack = {
    "Decision Tree": safe_joblib_load("best_dt_pack.pkl"),
    "Logistic Regression": safe_joblib_load("best_lg_pack.pkl"),
    "SVM": safe_joblib_load("best_svm_pack.pkl"),
    "XGBoost": safe_joblib_load("best_xgb_pack.pkl")
}

def generar_pestaña_indicadores():
    tabs = dcc.Tab(label="Indicadores del Modelo", children=[
        html.Br(),
        html.H3("Comparativa de Métricas y Errores de Modelos", className="text-center mb-4"),

        # --- SECCIÓN 1: Métricas principales (tabla + barras)
        html.H4("1️. Métricas principales por modelo"),
        dbc.Row([
            dbc.Col(html.Div(id="tabla-metrics"), md=6),
            dbc.Col(dcc.Graph(id="grafico-metrics"), md=6),
        ]),
        html.Div([
            html.P("Al revisar los resultados de los modelos en el contexto del riesgo de reingreso hospitalario en pacientes con diabetes, se puede ver que XGBoost es el que mejor logra identificar qué pacientes podrían volver a ser hospitalizados, ya que obtiene las métricas más altas en casi todos los indicadores. Esto es importante porque un buen recall y un buen AUC significan que el modelo es capaz de detectar a tiempo a los pacientes con riesgo real, sin dejar pasar tantos casos que podrían convertirse en nuevas hospitalizaciones. En comparación, modelos como Decision Tree, Logistic Regression y SVM muestran un rendimiento más moderado, lo que implica que podrían fallar más al momento de anticipar un reingreso. Para la gerencia hospitalaria, contar con un modelo más preciso como XGBoost puede marcar una gran diferencia: permite planear mejor los recursos, priorizar pacientes vulnerables, reducir costos por reingresos y mejorar la continuidad del cuidado. En otras palabras, un modelo más acertado no solo ayuda a predecir riesgos, sino que también contribuye a tomar decisiones más informadas que benefician tanto al hospital como a los pacientes."),
        ], className="mb-4"),

        # --- SECCIÓN 2: Recall comparativo
        html.H4("2️. Métrica principal: Recall"),
        dcc.Graph(id="recall-plot"),
        html.Div([
            html.P("""
En un problema como la predicción del riesgo de reingreso en pacientes con diabetes, la métrica más importante es el recall, porque lo que realmente necesita el hospital es no dejar pasar a los pacientes que sí tienen alto riesgo. Un modelo con buen recall ayuda a la gerencia hospitalaria a identificar oportunamente a quienes podrían volver a ser hospitalizados, lo que permite planear mejor los recursos, hacer intervenciones tempranas y reducir costos y complicaciones. En otras palabras, aquí es más grave “no detectar” a un paciente riesgoso que dar una falsa alarma.

Con esto en mente, los resultados muestran que XGBoost es el modelo que mejor logra identificar a los pacientes que realmente presentan riesgo de reingreso, seguido por Logistic Regression y SVM, mientras que Decision Tree queda rezagado. Esto significa que, en un contexto hospitalario, XGBoost sería el más útil para apoyar decisiones de seguimiento y monitoreo, ya que es el que menos pacientes de riesgo se le escapan. Así, los hallazgos no solo comparan modelos, sino que ayudan a entender cuál aporta más valor para mejorar la gestión del cuidado en pacientes con diabetes.
                   """),
        ], className="mb-4"),

        # --- SECCIÓN 3: Indicadores de error probabilístico
        html.H4("3️. Indicadores de error probabilístico"),
        dcc.Graph(id="prob-error-plot"),
        html.Div([
            html.P("""
                   En este tipo de problema, donde buscamos predecir el riesgo de que un paciente con diabetes vuelva a ser hospitalizado, también es clave analizar el error probabilístico, porque no solo importa si el modelo acierta o falla, sino qué tan bien calibra las probabilidades. En la práctica hospitalaria, estas probabilidades pueden usarse para priorizar pacientes, asignar recursos y planear intervenciones, por lo que un modelo mal calibrado podría llevar a decisiones poco efectivas. Métricas como el Log Loss y el Brier Score muestran qué tan confiable es la probabilidad que el modelo asigna a cada paciente, mientras que el ROC-AUC indica qué tan bien separa a los pacientes con riesgo real de los que no.

Al observar los resultados, se nota que XGBoost es el modelo con menor Log Loss y menor Brier Score, lo que significa que es el que mejor calibra y menos se equivoca al estimar las probabilidades de reingreso. Además, obtiene el ROC-AUC más alto, reforzando que distingue mejor entre pacientes de alto y bajo riesgo. Por el contrario, modelos como Logistic Regression y SVM presentan errores probabilísticos más altos, lo que indica que aunque puedan acertar ciertos casos, sus probabilidades no son tan confiables. En el contexto de la gerencia hospitalaria, esto refuerza la idea de que XGBoost es la opción más útil, ya que permite tomar decisiones basadas en estimaciones más precisas y enfocadas en optimizar la gestión del riesgo en pacientes con diabetes.
                   """),
        ], className="mb-4"),
    ])
    return tabs

interpretaciones_num = {
    "time_in_hospital": "Los pacientes permanecen hospitalizados en promedio 4,4 días, con una mediana de 4 días y un rango que va de 1 a 14. El 75% no supera los 6 días, lo que indica una estancia moderada en la mayoría de los casos. El diagrama de caja confirma una distribución ligeramente sesgada a la derecha, con la mayor concentración entre 2 y 6 días, y la presencia de valores atípicos en 13 y 14 días, asociados a hospitalizaciones prolongadas.",
    "num_lab_procedures": "El promedio es de 43 procedimientos, con una mediana de 44 y una amplia variabilidad (DE = 19,67). Los valores oscilan entre 1 y 132, lo que refleja diferencias significativas entre pacientes. En el diagrama de caja, la mayoría se ubica entre 30 y 58 procedimientos, mientras que los outliers superiores (más de 97) indican pacientes con un seguimiento clínico más exhaustivo o patologías complejas.",
    "num_procedures": "Los pacientes presentan en promedio 1,3 procedimientos, con una mediana de 1 y valores entre 0 y 6. La mayoría recibe 0 a 2 procedimientos, evidenciando baja intervención médica en términos quirúrgicos o especializados. El diagrama de caja muestra un sesgo positivo pronunciado, con pocos casos que superan los 4 procedimientos, considerados outliers que reflejan pacientes con alta complejidad clínica.",
    "num_medications": "El promedio de medicamentos prescritos es 16, con una mediana de 15 y una desviación estándar de 8,1. El rango intercuartílico (10–20) sugiere un nivel moderado de tratamiento farmacológico. En el diagrama de caja, la distribución está fuertemente sesgada a la derecha, con valores atípicos que superan los 35 e incluso alcanzan 81 medicamentos, lo que indica casos de polifarmacia asociada a condiciones complejas o múltiples comorbilidades.",
    "number_outpatient": "El promedio es 0,37, con una mediana de 0, lo que significa que la gran mayoría de los pacientes no tuvo consultas externas recientes. El diagrama de caja muestra una concentración en cero y una larga cola hacia la derecha, con valores atípicos hasta 40 visitas, reflejando pocos pacientes con seguimiento médico ambulatorio intensivo.",
    "number_emergency": "La media es 0,20, con una mediana de 0, indicando que casi todos los pacientes no acudieron a urgencias antes de su hospitalización. El rango llega hasta 76 visitas, aunque estos valores son excepcionales. En el diagrama de caja, la distribución se concentra en cero con outliers que superan las 20 o 60 visitas, representando casos atípicos de pacientes con alta dependencia del servicio de urgencias o condiciones crónicas descontroladas.",
    "number_inpatient": "El promedio es 0,64, con una mediana de 0, y valores entre 0 y 21. La mayoría de los pacientes no presenta hospitalizaciones previas recientes. El diagrama de caja confirma un sesgo positivo pronunciado, con valores atípicos que superan las 5 o 15 hospitalizaciones, lo que identifica a pacientes con reingresos frecuentes, posiblemente por enfermedades crónicas o tratamientos prolongados.",
    "number_diagnoses": "Los pacientes tienen en promedio 7,4 diagnósticos, con una mediana de 8 y valores entre 1 y 16. La dispersión es baja (DE = 1,93), lo que indica cierta homogeneidad en la cantidad de diagnósticos por paciente. El diagrama de caja muestra una distribución casi simétrica, con la mayoría concentrada entre 6 y 9 diagnósticos. Los outliers —uno inferior (1) y algunos superiores (≥14)— reflejan casos atípicos con menor o mayor carga diagnóstica de lo habitual.",
}

cat_meds_1 = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide',
    'citoglipton'
]

cat_meds_2 = [
    'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone'
]

interpretaciones_cat = {
    "race": "Se observa que la mayoría de los pacientes pertenecen a la categoría 2, correspondiente a Caucasian (caucásicos). Las demás categorías, como AfricanAmerican, Hispanic, Asian y Other, muestran una frecuencia considerablemente menor. Esto indica que la población del estudio está compuesta mayoritariamente por pacientes de origen caucásico, lo que puede influir en la representatividad del análisis.",
    
    "gender": "La variable muestra una ligera mayoría de la categoría 0 (Female) frente a la 1 (Male), reflejando una leve predominancia de mujeres en la muestra. La categoría 2 (Unknown/Invalid) tiene una presencia mínima, lo cual sugiere una buena calidad del registro en este campo.",
    
    "age": "Se identifica una mayor concentración de pacientes en las categorías 5 a 7, que representan los grupos de edad entre 50 y 80 años. Este patrón es coherente con el perfil clínico de enfermedades como la diabetes tipo 2, que afecta principalmente a adultos mayores. La frecuencia disminuye en los extremos inferiores y superiores de la edad.",
    
    "admission_type": "Predomina la categoría 1 (Emergency), lo que indica que la mayoría de los pacientes ingresaron al hospital por situaciones urgentes o no programadas. En contraste, los ingresos de tipo Elective (0) y Urgent (4) son menos frecuentes, lo que refleja la naturaleza crítica de la atención hospitalaria en este conjunto de datos.",
    
    "discharge_disposition": "Aunque existen numerosas categorías, la más común es la 1 (Discharged to home), lo que implica que la mayoría de los pacientes fueron dados de alta a su domicilio tras el tratamiento. Otras categorías como 3 (Transfer to another facility) o 11 (Expired) aparecen con menor frecuencia, pero son relevantes para el análisis del desenlace clínico.",
    
    "admission_source": "La categoría más representativa es la 2 (Emergency Room), confirmando que la mayoría de los ingresos provienen de la sala de urgencias. Esto es consistente con el predominio de ingresos de tipo Emergency observado en la variable admission_type_id, lo que refuerza la coherencia interna del dataset.",
    
    "diag_1": "La categoría más frecuente son enfermedades del sistema circulatorio. Le sigue la categoría 14, que representa los diagnósticos de diabetes (por ejemplo, códigos 250.xx). Esta combinación muestra que muchos pacientes ingresan al hospital con complicaciones cardíacas o directamente por complicaciones derivadas de la diabetes.",
    
    "diag_2": "Las estadísticas muestran que muchas personas presentan enfermedades cardiovasculares como condición secundaria. Le siguen las categorías 3 (enfermedades del sistema respiratorio) y 14 (diabetes), con frecuencias similares. Esto refleja que, en pacientes con múltiples condiciones, es muy común ver esta combinación de enfermedades circulatorias, respiratorias y metabólicas.",
    
    "diag_3": "En este caso, la categoría 2 continúa siendo la más común, y la 3 le sigue con una frecuencia notable, un poco más de la mitad de la que tiene la categoría 2. Esta tendencia refuerza la idea de que las enfermedades del corazón y pulmón son condiciones crónicas recurrentes en pacientes hospitalizados, muchas veces en conjunto con la diabetes.",
    
    "max_glu_serum": "La mayoría de los registros se encuentran en la categoría 3, que corresponde a No se realizó el test. Esto indica que para la gran mayoría de pacientes no se midió el valor máximo de glucosa durante su estancia hospitalaria. Las demás categorías (0, 1, 2) tienen frecuencias muy bajas, lo que sugiere que cuando sí se hace la prueba, es en muy pocos casos.",
    
    "A1Cresult": "Al igual que en la variable anterior, la categoría más frecuente es la 3, lo que también significa que no se realizó el test de hemoglobina glicosilada. Las otras tres categorías (0: normal, 1: mayor que 7, 2: mayor que 8) están presentes pero con frecuencias mucho menores. Esto puede implicar que el seguimiento a largo plazo del control glucémico no se hace de manera sistemática en los pacientes hospitalizados.",
    
    "insulin": "La insulina muestra frecuencias distribuidas entre todas las categorías. Esto indica que hubo ajustes importantes en su administración, probablemente en respuesta a las necesidades clínicas inmediatas de los pacientes hospitalizados.",
    
    "change": "Sobre la variable change se observa que la mayoría de los pacientes presentan un valor de 1, lo que indica que hubo un cambio en la medicación durante la hospitalización. Este comportamiento sugiere que, en muchos casos, el tratamiento fue ajustado, probablemente en respuesta a evaluaciones médicas o complicaciones agudas. No obstante, una cantidad considerable de pacientes también se mantuvo con su tratamiento sin cambios (0), lo que podría reflejar condiciones estables o seguimiento de un protocolo ya establecido.",
    
    "diabetesMed": "La variable diabetesMed muestra que la mayoría de los pacientes tienen un valor de 1, lo que significa que recibieron medicación para la diabetes durante su estancia hospitalaria. En contraste, una proporción menor de pacientes (0) no recibió medicación, lo cual puede deberse a múltiples factores, como estadías cortas, control dietético o decisiones clínicas específicas. La predominancia del uso de medicamentos refleja la importancia del tratamiento farmacológico en el manejo hospitalario de la diabetes.",
    
    "readmitted": "Las estadísticas revelan que la mayoría de los pacientes no fueron readmitidos (2), con una frecuencia superior a 50.000 casos. En segundo lugar se encuentran los pacientes que sí fueron readmitidos, pero después de 30 días (1), y finalmente, con una frecuencia mucho menor, los pacientes que fueron readmitidos antes de 30 días (0)."
}


nav_buttons = dbc.ButtonGroup(
    [
        dbc.Button("1. Introducción", id="btn-1", outline=True, color="primary"),
        dbc.Button("2. Contexto", id="btn-2", outline=True, color="primary"),
        dbc.Button("3. Planteamiento del Problema", id="btn-3", outline=True, color="primary"),
        dbc.Button("4. Objetivos y Justificación", id="btn-4", outline=True, color="primary"),
        dbc.Button("5. Marco Teórico", id="btn-5", outline=True, color="primary"),
        dbc.Button("6. Metodología", id="btn-6", outline=True, color="primary"),
        dbc.Button("7. Resultados/Análisis Final", id="btn-7", outline=True, color="primary"),
        dbc.Button("8. Conclusiones", id="btn-8", outline=True, color="primary")
    ],
    className="d-flex justify-content-around mb-4 flex-wrap gap-2",
)

app.layout = dbc.Container(
    [
        html.H1("Dashboard de Riesgo de Readmisión", className="text-center mt-3 mb-4"),
        html.Hr(),
        nav_buttons,
        html.Div(id="content-area"),
    ],
    fluid=True,
)

# -------------------------------------
# Callback principal 
# -------------------------------------
@app.callback(
    Output("content-area", "children"),
    [Input(f"btn-{i}", "n_clicks") for i in range(1, 9)]
)
def mostrar_contenido(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        # Página por defecto (Introducción)
        boton_id = "btn-1"
    else:
        boton_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # --- Btn 1: Introducción ---
    if boton_id == "btn-1":
        return dbc.Card([
            dbc.CardBody([
                html.H4("Introducción", className="card-title"),
                html.P("""
                    Este dashboard presenta un análisis interactivo del riesgo de readmisión basado en el dataset
                    original (Diabetes 130-US hospitals for years 1999–2008). 
                """),
                html.P("Objetivo: ")
            ])
        ])

    # --- Btn 2: Contexto ---
    elif boton_id == "btn-2":
        return dbc.Card([dbc.CardBody([
            html.H4("Contexto", className="card-title"),
            html.P("""
                Datos tomados del dataset 'diabetic_data' (UCI). Contiene variables demográficas, de diagnósticos,
                procedimientos y medicación, con el objetivo de explorar factores asociados a readmisión.
            """)
        ])])

    # --- Btn 3: Planteamiento del Problema ---
    elif boton_id == "btn-3":
        return dbc.Card([dbc.CardBody([
            html.H4("Planteamiento del Problema", className="card-title"),
            html.P("""
                Identificar patrones y factores asociados al riesgo de readmisión en el dataset, mediante
                análisis exploratorio e indicadores resumen.
            """)
        ])])

    # --- Btn 4: Objetivos y Justificación ---
    elif boton_id == "btn-4":
        return dbc.Card([dbc.CardBody([
            html.H4("Objetivos y Justificación", className="card-title"),
            html.Ul([
                html.Li("Explorar la distribución de variables clave (edad, género, tiempo en hospital, medicamentos)."),
                html.Li("Construir KPIs y visualizaciones que faciliten la identificación de grupos de riesgo."),
                html.Li("Generar un dashboard interactivo replicable en Dash.")
            ])
        ])])

    # --- Btn 5: Marco Teórico ---
    elif boton_id == "btn-5":
        return dbc.Card([dbc.CardBody([
            html.H4("Marco Teórico", className="card-title"),
            html.P("Análisis descriptivo y visualizaciones son herramientas clave para la identificación de patrones en salud pública.")
        ])])

    # --- Btn 6: Metodología ---
    elif boton_id == "btn-6":
        return dbc.Card([dbc.CardBody([
            html.H4("Metodología", className="card-title"),
            html.P("Se utiliza Pandas para manipulación, Plotly para visualizaciones interactivas y Dash/Bootstrap para la interfaz.")
        ])])

    # --- Btn 7: Resultados/Analisis Final ---
    elif boton_id == "btn-7":
        tabs = dcc.Tabs([
            # ==========================
            # EDA: UNIVARIADO
            # ==========================
                dcc.Tab(label="EDA - Univariado", children=[
                html.Br(),
                html.H3("Análisis Univariado", className="text-center mb-4"),
                dbc.Row([
                    # --- NUMÉRICO ---
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H5("Variable Numérica")),
                            dbc.CardBody([
                                html.Label("Selecciona una variable numérica:"),
                                dcc.Dropdown(
                                    id="uni-num-dropdown",
                                    options=[{"label": c, "value": c} for c in num_cols],
                                    value=None,
                                    placeholder="Selecciona una variable numérica...",
                                    clearable=True
                                ),
                                dcc.Graph(id="uni-num-box", style={"height": "500px"}),
                                html.Div(id="uni-num-table", className="mt-3"),
                                html.Div(id="uni-num-interpretacion", className="mt-3")  
                            ])
                        ])
                    ], md=6),

                    # --- CATEGÓRICA ---
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H5("Variable Categórica")),
                            dbc.CardBody([
                                html.Label("Selecciona una variable categórica:"),
                                dcc.Dropdown(
                                    id="uni-cat-dropdown",
                                    options=[{"label": c, "value": c} for c in cat_cols],
                                    value=None,
                                    placeholder="Selecciona una variable categórica...",
                                    clearable=True
                                ),
                                dcc.Graph(id="uni-cat-bar", style={"height": "500px"}),
                                html.Div(id="uni-cat-table", className="mt-3"),
                                html.Div(id="uni-cat-interpretacion", className="mt-3")
                            ])
                        ])
                    ], md=6),
                ])
            ]),
            # ==========================
            # EDA: BIVARIADO + MULTIVARIADO
            # ==========================
            dcc.Tab(label="EDA - Bivariado/Multivariado", children=[
                html.Br(),
                html.H3("Análisis Bivariado", className="text-center mb-4"),

                # --- CATEGÓRICA vs CATEGÓRICA ---
                dbc.Card([
                    dbc.CardHeader(html.H5("Categórica vs Categórica")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Variable categórica (X):"),
                                dcc.Dropdown(
                                    id="biv-cat-x",
                                    options=[{"label": c, "value": c} for c in cat_cols],
                                    value=None,
                                    placeholder="Selecciona variable X...",
                                    clearable=True
                                )
                            ], md=6),
                            dbc.Col([
                                html.Label("Variable categórica (Y):"),
                                dcc.Dropdown(
                                    id="biv-cat-y",
                                    options=[{"label": c, "value": c} for c in cat_cols],
                                    value=None,
                                    placeholder="Selecciona variable Y...",
                                    clearable=True
                                )
                            ], md=6),
                        ]),
                        html.Br(),
                        dcc.Graph(id="biv-catcat", style={"height": "550px", "width": "95%"})
                    ])
                ]), 
                html.Div([
                    html.P(""" 
                    De forma general, en estas gráficas se observa que, para la mayoría de las variables categóricas, una gran proporción de los pacientes corresponde a aquellos que no han sido readmitidos en el hospital. Este patrón se repite de manera consistente, independientemente de la variable analizada, lo que sugiere que la clase de no readmisión es predominante en el conjunto de datos. Si bien algunas variables como race, gender o age muestran una distribución más equilibrada entre sus categorías internas, la tendencia general sigue favoreciendo a los pacientes sin readmisión.
                    En conjunto, no se identifican patrones categóricos evidentes que permitan diferenciar de forma clara a los pacientes según su estado de readmisión únicamente a partir de estas variables. Sin embargo, se detecta que ciertas variables con un mayor número de categorías —como medical_specialty o payer_code— podrían contener información útil si se agrupan o transforman para reducir la dispersión y aumentar su representatividad. Asimismo, variables relacionadas con tratamientos o medicación, aunque concentradas en pocas categorías, podrían aportar valor predictivo al combinarse con otras variables en un modelo multivariado, ya que podrían reflejar prácticas clínicas o perfiles de pacientes asociados con un mayor riesgo de readmisión.
                    """),
                ], className="mb-4"),

                # --- NUMÉRICA vs NUMÉRICA ---
                dbc.Card([
                    dbc.CardHeader(html.H5("Numérica vs Numérica")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Variable numérica (X):"),
                                dcc.Dropdown(
                                    id="biv-num-x",
                                    options=[{"label": c, "value": c} for c in num_cols],
                                    value=None,
                                    placeholder="Selecciona variable X...",
                                    clearable=True
                                )
                            ], md=6),
                            dbc.Col([
                                html.Label("Variable numérica (Y):"),
                                dcc.Dropdown(
                                    id="biv-num-y",
                                    options=[{"label": c, "value": c} for c in num_cols],
                                    value=None,
                                    placeholder="Selecciona variable Y...",
                                    clearable=True
                                )
                            ], md=6),
                        ]),
                        html.Br(),
                        dcc.Graph(id="biv-scatter", style={"height": "400px"})
                    ])
                ]),
                html.Div([
                    html.P("Como análisis general, no se aprecian relaciones lineales claras o fuertes entre la mayoría de las variables numéricas, lo que coincide con lo observado en la matriz de correlación, donde no se identificaron valores de 𝑟^2 elevados. En su lugar, predominan correlaciones débiles o negativas, lo que sugiere que estas variables, de forma individual, podrían tener una capacidad limitada para explicar la variabilidad de otras dentro del conjunto de datos."),
                ], className="mb-4"),

                # --- NUMÉRICA vs CATEGÓRICA ---
                dbc.Card([
                    dbc.CardHeader(html.H5("Numérica vs Categórica")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Variable numérica (Y):"),
                                dcc.Dropdown(
                                    id="biv-num-y2",
                                    options=[{"label": c, "value": c} for c in num_cols],
                                    value=None,
                                    placeholder="Selecciona variable numérica...",
                                    clearable=True
                                )
                            ], md=6),
                            dbc.Col([
                                html.Label("Variable categórica (X):"),
                                dcc.Dropdown(
                                    id="biv-cat-x2",
                                    options=[{"label": c, "value": c} for c in cat_cols],
                                    value=None,
                                    placeholder="Selecciona variable categórica...",
                                    clearable=True
                                )
                            ], md=6),
                        ]),
                        html.Br(),
                        dcc.Graph(id="biv-boxplot", style={"height": "400px"})
                    ]),
                ],
                className="mb-4"),

                # --- MULTIVARIADO ---
                dbc.Card([
                    dbc.CardHeader(html.H5("Multivariado")),
                    dbc.CardBody([
                        dcc.Graph(id="corr-heatmap", style={"height": "600px"})
                    ])
                ]),
                html.Div([
                    html.P("La matriz de correlación muestra que no existen relaciones lineales fuertes entre las variables numéricas del conjunto de datos. La mayoría de los coeficientes están cerca de cero, lo que indica una asociación débil o inexistente entre las variables. La correlación más alta es 0.47 entre time_in_hospital y num_medications, lo que representa una relación moderada. También se observa una correlación leve de 0.39 entre num_procedures y num_medications. En general, los valores indican que las variables numéricas analizadas tienden a comportarse de manera independiente unas de otras."),
                ], className="mb-4"),
            ]),


            dcc.Tab(label="Dashboard", children=[
                html.Br(),
                html.H3("Dashboard Interactivo - KPIs y Visualizaciones"),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Grupo de Edad"),
                            dcc.Dropdown(id="filter-age", options=[{"label": "Todos", "value": "Todos"}] + [{"label": v, "value": v} for v in sorted(df['age'].dropna().unique())], value="Todos")
                        ], md=2),
                        dbc.Col([
                            html.Label("Tipo de admisión"),
                            dcc.Dropdown(id="filter-admission", options=[{"label": "Todos", "value": "Todos"}] + [{"label": v, "value": v} for v in sorted(df['admission_type'].dropna().unique())], value="Todos")
                        ], md=2),
                        dbc.Col([
                            html.Label("Tipo de insulina"),
                            dcc.Dropdown(id="filter-insulin", options=[{"label": "Todos", "value": "Todos"}] + [{"label": v, "value": v} for v in sorted(df['insulin'].dropna().unique())], value="Todos")
                        ], md=2),
                        dbc.Col([
                            html.Label("Género"),
                            dcc.Dropdown(id="filter-gender", options=[{"label": "Todos", "value": "Todos"}] + [{"label": v, "value": v} for v in sorted(df['gender'].dropna().unique())], value="Todos")
                        ], md=2),
                        dbc.Col([
                            html.Label("Estado de readmisión"),
                            dcc.Dropdown(id="filter-readmit", options=[{"label": "Todos", "value": "Todos"}] + [{"label": v, "value": v} for v in sorted(df['readmitted'].dropna().unique())], value="Todos")
                        ], md=2),
                        dbc.Col([
                            html.Label("Rango días en hospital"),
                            dcc.RangeSlider(
                                id="filter-days",
                                min=int(df['time_in_hospital'].min()) if 'time_in_hospital' in df.columns else 0,
                                max=int(df['time_in_hospital'].max()) if 'time_in_hospital' in df.columns else 30,
                                value=[int(df['time_in_hospital'].min()) if 'time_in_hospital' in df.columns else 0, int(df['time_in_hospital'].max()) if 'time_in_hospital' in df.columns else 30],
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": False}
                            )
                        ], md=12)
                    ], className="g-2")
                ], style={"marginBottom": "20px"}),
                # KPIs
                html.Div(id="kpi-cards"),
                html.Hr(),
                # Gráficos
                dbc.Row([
                    dbc.Col(dcc.Graph(id="gauge-readmit"), md=4),
                    dbc.Col(dcc.Graph(id="scatter-meds-days"), md=8)
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="bar-admission-readmit"), md=6),
                    dbc.Col(dcc.Graph(id="heatmap-age-diagnoses"), md=6)
                ]),
                html.Hr(),
                html.H5("Tabla de Pacientes Filtrados"),
                html.Div(id="filtered-table")
            ]),

            # ==========================
            # Visualización DE MODELOS
            # ==========================
            dcc.Tab(label="Visualización de Modelos", children=[
                html.Br(),
                html.H3("Visualización de Resultados del Modelo", className="text-center mb-4"),

                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H5("Selecciona un modelo entrenado")),
                            dbc.CardBody([
                                dcc.Dropdown(
                                    id="modelo-dropdown",
                                    options=[
                                        {"label": "Regresión Logística", "value": "best_lg_pack.pkl"},
                                        {"label": "Árbol de Decisión", "value": "best_dt_pack.pkl"},
                                        {"label": "SVM", "value": "best_svm_pack.pkl"},
                                        {"label": "XGBoost", "value": "best_xgb_pack.pkl"},
                                    ],
                                    placeholder="Selecciona el archivo .pkl del modelo...",
                                    clearable=True
                                )
                            ])
                        ])
                    ], md=4),
                ], justify="center"),

                html.Br(),

                # --- Tabla de métricas ---
                html.Div(id="metrics-table", className="mb-4"),
                

                # --- Gráficas: matriz de confusión + ROC ---
                dbc.Row([
                    dbc.Col(dcc.Graph(id="ytrue-vs-ypred", style={"height": "550px", "width": "95%"}), md=6),
                    dbc.Col(dcc.Graph(id="residuals-plot", style={"height": "550px", "width": "95%"}), md=6),
                ], justify="center"),

                # --- Interpretación centrada debajo de las gráficas ---
                html.Div(id="modelo-interpretacion", className="mt-5 text-center")
            ]),

            # ==========================
            # INDICADORES DE EVALUACIÓN
            # ==========================

            dcc.Tab(label="Indicadores del Modelo", children=[generar_pestaña_indicadores()])

        ])
        return tabs

    # --- Btn 8: Conclusiones / Créditos ---
    elif boton_id == "btn-8":
        return dbc.Card([dbc.CardBody([
            html.H4("Conclusiones", className="card-title"),
            html.P("Conclusiones breves del análisis "),
            html.Ul([
                html.Li(""),
                html.Li("Visualizaciones: histogramas, boxplots, scatter, heatmaps y KPIs.")
            ]),
            html.P("Fuente: UCI Machine Learning Repository - Diabetes 130-US hospitals for years 1999–2008.")
        ])])

    # Fallback
    return html.P("Selecciona una sección del informe para comenzar.", className="text-muted text-center")

@app.callback(
    Output("modelo-interpretacion", "children"),
    Input("modelo-dropdown", "value")
)
def interpretar_modelo(modelo):
    if not modelo:
        return ""

    interpretaciones_modelos = {
        "best_rl_pack.pkl": """
La regresión logística muestra un rendimiento moderado en este problema, con métricas alrededor del 0.60. Aunque no es el modelo más fuerte, sí logra capturar una parte importante de los pacientes que realmente regresan al hospital (Recall ≈ 0.594). Esto es relevante porque en la gestión hospitalaria es más grave no identificar a un paciente que sí terminará reingresando, ya que se perdería la oportunidad de intervenir a tiempo. La matriz de confusión confirma esto: aunque el modelo se equivoca en ambos sentidos, logra identificar correctamente a más de la mitad de los pacientes que sí reingresarán (5572 casos), lo cual es útil para priorizar seguimientos o monitoreo adicional.

Por otro lado, el AUC de 0.602 y la curva ROC muestran que el modelo distingue los casos positivos un poco mejor que el azar, pero todavía queda margen de mejora. Esto implica que, aunque el modelo ofrece señales útiles, su capacidad para separar pacientes de alto y bajo riesgo no es completamente sólida, por lo que decisiones estrictamente basadas en él deberían complementarse con criterio clínico y otras herramientas. En general, este modelo sirve como una base estable y explicativa, pero en la práctica hospitalaria lo ideal sería utilizarlo como apoyo, no como única guía para gestionar el riesgo de reingreso.
        """,
        "best_dt_pack.pkl": "El modelo de Árbol de Decisión ofrece un desempeño moderado para predecir el reingreso de pacientes con diabetes. Aunque clasifica bien a quienes no regresan al hospital, tiene dificultades para identificar a quienes sí lo harán (recall ≈ 0.487), lo que es crítico porque muchos reingresos diabéticos están asociados a complicaciones que podrían prevenirse con seguimiento oportuno. La matriz de confusión muestra que el modelo deja pasar a casi la mitad de los pacientes que realmente reingresan, lo que limita su utilidad clínica. Aun así, el AUC de 0.674 indica que logra distinguir mejor que el azar entre pacientes de alto y bajo riesgo, capturando algunos patrones relevantes, pero no con la fuerza necesaria para usarlo como herramienta principal; funciona mejor como apoyo y no como única guía de decisión.",
        "best_svm_pack.pkl": "El modelo SVM con SMOTE muestra un desempeño moderado para identificar el riesgo de reingreso en pacientes con diabetes. Aunque su accuracy es de 0.623, lo más relevante es que logra un recall de 0.594, es decir, identifica correctamente a más de la mitad de los pacientes que realmente volverán al hospital, algo clave porque muchos reingresos diabéticos pueden prevenirse con seguimiento oportuno. La matriz de confusión confirma esto: el modelo detecta 5572 pacientes que sí reingresan, aunque aún deja pasar a una cantidad importante (3809). Esto indica que el SVM, incluso con el balanceo de clases, todavía tiene dificultades para separar completamente a quienes están en alto riesgo. El AUC de 0.602 muestra que el modelo distingue los casos mejor que el azar, pero sin llegar a ser un predictor fuerte. En general, el SVM aporta señales útiles para priorizar pacientes diabéticos que podrían necesitar más acompañamiento, pero no es suficientemente preciso como para basar decisiones clínicas únicamente en él.",
        "best_xgb_pack.pkl": "El modelo XGBoost muestra un mejor desempeño que varios modelos anteriores en la predicción del reingreso de pacientes con diabetes, alcanzando un recall de 0.629, lo que significa que identifica correctamente a más de la mitad de los pacientes que realmente volverán al hospital. Esto es valioso porque muchos reingresos en personas con diabetes se relacionan con descompensaciones y problemas que podrían prevenirse con seguimiento. La matriz de confusión muestra que el modelo detecta 5902 casos positivos y reduce la cantidad de pacientes de alto riesgo que pasan desapercibidos (3479), lo cual representa una mejora respecto a otros algoritmos. Además, el AUC de 0.70 indica que XGBoost distingue mejor entre pacientes de alto y bajo riesgo, capturando patrones complejos de la enfermedad. Aunque no es perfecto, ofrece una capacidad predictiva más sólida, lo que lo convierte en una herramienta útil para apoyar decisiones de priorización y seguimiento clínico en pacientes diabéticos."
    }

    texto = interpretaciones_modelos.get(modelo, "Modelo seleccionado: interpretación genérica no definida.")
    
    return html.P(texto, style={"font-weight": "bold", "font-size": "18px"})


# ====================================================
# Callbacks EDA - Univariado
# ====================================================
@app.callback(
    Output("uni-num-box", "figure"),
    Output("uni-num-table", "children"),
    Input("uni-num-dropdown", "value")
)
def actualizar_univariado_num(var_num):
    if not var_num or var_num not in df.columns:
        return go.Figure(), html.Div()

    fig = px.box(df, y=var_num, points="all", title=f"Boxplot — {var_num}")
    fig.update_layout(template="plotly_white", height=320)

    stats_df = df[var_num].describe().to_frame().T.round(4)
    table = dbc.Table.from_dataframe(stats_df, striped=True, bordered=True, hover=True)

    return fig, html.Div([html.H6("Estadísticas descriptivas:"), table])

@app.callback(
    Output("uni-cat-bar", "figure"),
    Output("uni-cat-table", "children"),
    Input("uni-cat-dropdown", "value")
)
def actualizar_univariado_cat(var_cat):
    if not var_cat or var_cat not in df.columns:
        return go.Figure(), html.Div()

    # --- Gráfico de barras ---
    vc = df[var_cat].value_counts().nlargest(30)
    fig = px.bar(
        x=vc.index.astype(str),
        y=vc.values,
        labels={"x": var_cat, "y": "Frecuencia"},
        title=f"Distribución de {var_cat}"
    )
    fig.update_layout(template="plotly_white", xaxis_tickangle=25, height=320)

    # --- Tabla tipo describe(include='object') ---
    desc_df = df[[var_cat]].describe(include="object").T  # describe para esa sola variable
    desc_df = desc_df.rename_axis("Variable").reset_index()
    desc_df = desc_df.round(2)

    table = dbc.Table.from_dataframe(desc_df, striped=True, bordered=True, hover=True)

    return fig, html.Div([html.H6("Estadísticas descriptivas:"), table])


# ====================================================
# Callbacks EDA - Bivariado
# ====================================================
@app.callback(
    Output("biv-catcat", "figure"),
    Output("biv-scatter", "figure"),
    Output("biv-boxplot", "figure"),
    Input("biv-cat-x", "value"),
    Input("biv-cat-y", "value"),
    Input("biv-num-x", "value"),
    Input("biv-num-y", "value"),
    Input("biv-num-y2", "value"),
    Input("biv-cat-x2", "value")
)
def actualizar_bivariado(cat_x, cat_y, num_x, num_y, num_y2, cat_x2):
    # --- Categórica vs Categórica ---
    fig_catcat = go.Figure()
    if cat_x and cat_y and cat_x != cat_y:
        ct = pd.crosstab(df[cat_x], df[cat_y])
        fig_catcat = px.imshow(ct, title=f"Contingencia entre {cat_x} y {cat_y}")
        fig_catcat.update_layout(template="plotly_white", height=550)

    # --- Numérica vs Numérica ---
    fig_scatter = go.Figure()
    if num_x and num_y and num_x != num_y:
        fig_scatter = px.scatter(
            df, x=num_x, y=num_y,
            color="readmitted" if "readmitted" in df.columns else None,
            title=f"Relación entre {num_x} y {num_y}"
        )
        fig_scatter.update_layout(template="plotly_white", height=350)

    # --- Numérica vs Categórica ---
    fig_box = go.Figure()
    if num_y2 and cat_x2:
        fig_box = px.box(
            df, x=cat_x2, y=num_y2, color=cat_x2,
            title=f"{num_y2} según {cat_x2}"
        )
        fig_box.update_layout(template="plotly_white", height=350)

    return fig_catcat, fig_scatter, fig_box



# ====================================================
# Callback EDA - Correlaciones (Multivariado)
# ====================================================
@app.callback(
    Output("corr-heatmap", "figure"),
    Input("biv-cat-x", "value")  # trigger when selection changes (or any other)
)
def actualizar_corr_heatmap(_):
    if len(num_cols) < 2:
        return go.Figure()
    corr = df[num_cols].corr(method='spearman')
    fig = px.imshow(corr, text_auto=True, title="Matriz de correlaciones (Spearman)")
    fig.update_layout(template="plotly_white")
    return fig

import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ====================================================
# Callbacks - Visualización de Modelos (Clasificación)
# ====================================================
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, auc, confusion_matrix, log_loss, brier_score_loss
)

@app.callback(
    Output("metrics-table", "children"),
    Output("ytrue-vs-ypred", "figure"),
    Output("residuals-plot", "figure"),
    Input("modelo-dropdown", "value")
)
def visualizar_resultados_modelo(pkl_file):
    """
    Visualiza resultados de modelos de CLASIFICACIÓN
    (métricas, matriz de confusión, curva ROC, parámetros óptimos).
    """
    if not pkl_file:
        return html.Div(), go.Figure(), go.Figure()

    try:
        data = safe_joblib_load_full(pkl_file)
        if not data:
            raise ValueError("Archivo .pkl ilegible o vacío")

        # --- Extraer componentes del pack ---
        nombre_modelo = data.get("nombre", "Modelo sin nombre")
        y_test = pd.Series(data.get("y_test", []))
        y_pred = pd.Series(data.get("y_pred", []))
        y_pred_proba = data.get("y_pred_proba", None)
        metricas = data.get("metricas", {})
        cm = data.get("confusion_matrix", None)
        best_params = data.get("best_params", {})
        best_score_cv = data.get("best_score_cv", None)

        # =====================
        # 1️⃣ TABLA DE MÉTRICAS
        # =====================
        # Si el diccionario de métricas viene vacío, recalculamos
        if not metricas:
            metricas = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            }
            if y_pred_proba is not None:
                metricas["ROC-AUC"] = roc_auc_score(y_test, y_pred_proba)

        metricas_df = pd.DataFrame(list(metricas.items()), columns=["Métrica", "Valor"]).round(3)
        metricas_table = dbc.Table.from_dataframe(metricas_df, striped=True, bordered=True, hover=True)

        # ===========================
        # 2️⃣ MATRIZ DE CONFUSIÓN
        # ===========================
        fig_cm = go.Figure()
        if cm is not None:
            fig_cm = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues",
                title=f"Matriz de Confusión — {nombre_modelo}",
                labels=dict(x="Predicho", y="Real")
            )
            fig_cm.update_layout(template="plotly_white", height=350)
        else:
            # Si no está en el pkl, la calculamos
            cm_calc = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(
                cm_calc, text_auto=True, color_continuous_scale="Blues",
                title=f"Matriz de Confusión — {nombre_modelo}",
                labels=dict(x="Predicho", y="Real")
            )
            fig_cm.update_layout(template="plotly_white", height=500)

        # ===========================
        # 3️⃣ CURVA ROC
        # ===========================
        fig_roc = go.Figure()
        if y_pred_proba is not None and len(y_pred_proba) == len(y_test):
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.3f}"))
            fig_roc.add_shape(
                type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="red", dash="dash")
            )
            fig_roc.update_layout(
                title=f"Curva ROC — {nombre_modelo}",
                xaxis_title="Tasa de Falsos Positivos (FPR)",
                yaxis_title="Tasa de Verdaderos Positivos (TPR)",
                template="plotly_white", height=500
            )

        # ===========================
        # 4️⃣ PARÁMETROS ÓPTIMOS
        # ===========================
        params_df = pd.DataFrame(best_params.items(), columns=["Parámetro", "Valor"])
        if best_score_cv is not None:
            params_df.loc[len(params_df)] = ["Mejor Puntuación CV", round(best_score_cv, 3)]
        params_table = dbc.Table.from_dataframe(params_df, striped=True, bordered=True, hover=True)

        return (
            dbc.Row([
                # Columna 1: Tabla de métricas
                dbc.Col([
                    html.H5("Métricas de Evaluación:"),
                    metricas_table
                ], md=6),

                # Columna 2: Tabla de parámetros óptimos
                dbc.Col([
                    html.H5("Parámetros Óptimos del Modelo:"),
                    params_table
                ], md=6)
            ]),
            fig_cm,
            fig_roc
        )

    except Exception as e:
        print("❌ Error al cargar o procesar el modelo:", e)
        return html.Div([html.P(f"Error al procesar {pkl_file}: {e}")]), go.Figure(), go.Figure()

from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, f1_score, accuracy_score
import pickle

@app.callback(
    Output("uni-num-interpretacion", "children"),
    Input("uni-num-dropdown", "value")
)
def interpretar_num_variable(var):
    if not var:
        return ""
    return interpretaciones_num.get(var, "No hay interpretación disponible para esta variable.")

interpretacion_meds_1 = """  
En general, la mayoría de los medicamentos individuales como metformin, glimepiride, glyburide, pioglitazone, entre otros, tienen como categoría más común la uno, lo que indica que la medicación no cambió durante la estancia, es decir, los pacientes que ya estaban tomando estos medicamentos continuaron con ellos. Las categorías dos, correspondiente a medicación añadida, y tres, correspondiente a medicación discontinuada, aparecen con menor frecuencia. En la mayoría de los casos, la categoría cero, que representa medicación nunca utilizada, tiene una baja frecuencia, excepto en medicamentos poco comunes como acetohexamide, tolazamide o en combinaciones poco frecuentes.
 """

interpretacion_meds_2 = "En los medicamentos combinados, como glyburide-metformin, glipizide-metformin y metformin-pioglitazone, se observa que la mayoría de los pacientes no recibieron estas combinaciones durante la hospitalización. Esto puede deberse a que estas terapias son más comunes en tratamientos ambulatorios que en contextos agudos."

@app.callback(
    Output("uni-cat-interpretacion", "children"),
    Input("uni-cat-dropdown", "value")
)
def interpretar_cat_variable(var):
    if not var:
        return ""
    elif var in cat_meds_1:
        return interpretacion_meds_1
    elif var in cat_meds_2:
        return interpretacion_meds_2
    else:
        return interpretaciones_cat.get(var, "No hay interpretación disponible para esta variable.")

@app.callback(
    Output("biv-boxplot-interpretacion", "children"),
    Input("biv-num-y2", "value"),
    Input("biv-cat-x2", "value")
)
def interpretar_num_vs_cat(num_y, cat_x):
    if not num_y or not cat_x:
        return ""

    # Diccionario de interpretaciones personalizadas
    interpretaciones_biv = {
        ("age", "readmitted"): "Los pacientes mayores de 65 años tienen mayor probabilidad de readmisión.",
        ("num_lab_procedures", "num_medications"): "A mayor número de procedimientos, más medicamentos suelen prescribirse."
    }

    # Buscar interpretación personalizada
    texto = interpretaciones_biv.get((num_y, cat_x))
    if texto:
        return texto

    # ------------------------------
    # Fallback genérico si no está en el diccionario
    # ------------------------------
    return f"No hay interpretación diponible."

@app.callback(
    Output("biv-scatter-interpretacion", "children"),
    Input("biv-num-x", "value"),
    Input("biv-num-y", "value")
)
def interpretar_num_vs_num(num_x, num_y):
    if not num_x or not num_y:
        return ""

    interpretaciones_biv = {
        ("age", "num_medications"): "A mayor edad, los pacientes tienden a recibir más medicamentos.",
        ("num_lab_procedures", "num_procedures"): "Se observa correlación positiva entre procedimientos y laboratorios realizados."
    }

    texto = interpretaciones_biv.get((num_x, num_y))
    if texto:
        return texto

    corr = df[[num_x, num_y]].corr().iloc[0,1]
    return f"No hay interpretación diponible."

@app.callback(
    Output("biv-catcat-interpretacion", "children"),
    Input("biv-cat-x", "value"),
    Input("biv-cat-y", "value")
)
def interpretar_cat_vs_cat(cat_x, cat_y):
    if not cat_x or not cat_y:
        return ""

    interpretaciones_biv = {
        ("gender", "readmitted"): "Se observa que la readmisión es ligeramente más frecuente en pacientes de género femenino.",
        ("diabetesMed", "readmitted"): "Pacientes con medicación para diabetes muestran mayor probabilidad de readmisión."
    }

    # Fallback: tabla de frecuencias cruzadas
    ct = pd.crosstab(df[cat_x], df[cat_y])
    return f"No hay interpretación diponible."



# --- CALLBACK para generar métricas y gráficos comparativos ---
@app.callback(
    Output("tabla-metrics", "children"),
    Output("grafico-metrics", "figure"),
    Output("recall-plot", "figure"),
    Output("prob-error-plot", "figure"),
    Input("btn-7", "n_clicks")  # usamos un trigger cualquiera
)
def actualizar_comparativa_modelos(_):
    modelos = modelos_pack  # tu diccionario de modelos cargados

    # Preparar DataFrames
    metrics_list = []
    recall_list = []
    prob_list = []

    for nombre, pack in modelos.items():
        y_test = pd.Series(pack.get("y_test", []))
        y_pred = pd.Series(pack.get("y_pred", []))
        y_pred_proba = pack.get("y_pred_proba", None)
        metricas = pack.get("metricas", {})

        # Si metricas está vacío, calcular
        if not metricas:
            metricas = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1": f1_score(y_test, y_pred, zero_division=0),
            }
            if y_pred_proba is not None:
                metricas["ROC-AUC"] = roc_auc_score(y_test, y_pred_proba)

        metrics_list.append({"Modelo": nombre, **metricas})
        recall_list.append({"Modelo": nombre, "Recall": metricas.get("Recall", 0)})

        if y_pred_proba is not None:
            prob_list.append({
                "Modelo": nombre,
                "Log Loss": log_loss(y_test, y_pred_proba),
                "Brier Score": brier_score_loss(y_test, y_pred_proba),
                "ROC-AUC": roc_auc_score(y_test, y_pred_proba)
            })

    # --- Tabla métricas principales ---
    df_metrics = pd.DataFrame(metrics_list).round(3)
    table_metrics = dbc.Table.from_dataframe(df_metrics, striped=True, bordered=True, hover=True)

    # --- Gráfico de barras agrupadas métricas principales ---
    fig_metrics = px.bar(df_metrics.melt(id_vars="Modelo"), x="Modelo", y="value", color="variable",
                         barmode="group", title="Comparativa de métricas principales")
    fig_metrics.update_layout(template="plotly_white", height=350)

    # --- Gráfico Recall ---
    df_recall = pd.DataFrame(recall_list)
    fig_recall = px.bar(df_recall, x="Modelo", y="Recall", text="Recall",
                        title="Comparativa de Recall por modelo", color="Modelo")
    fig_recall.update_layout(template="plotly_white", height=350)

    # --- Gráfico de errores probabilísticos ---
    if prob_list:
        df_prob = pd.DataFrame(prob_list).melt(id_vars="Modelo", var_name="Indicador", value_name="Valor")
        fig_prob = px.bar(df_prob, x="Modelo", y="Valor", color="Indicador", barmode="group",
                          title="Indicadores de error probabilístico")
        fig_prob.update_layout(template="plotly_white", height=350)
    else:
        fig_prob = go.Figure()

    return table_metrics, fig_metrics, fig_recall, fig_prob


def safe_joblib_load_full(path):
    import joblib, sys, types, pickle, numpy as np
    from sklearn.pipeline import Pipeline

    try:
        if "modelo" not in sys.modules:
            fake_modelo = types.ModuleType("modelo")
            fake_modelo.dtype = np.dtype
            fake_modelo.Pipeline = Pipeline
            sys.modules["modelo"] = fake_modelo

        return joblib.load(path)

    except Exception as e:
        print(f"⚠️ Error parcial con {path}: {e}")
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e2:
            print(f"❌ Error total al abrir {path}: {e2}")
            return None


# ====================================================
# Callbacks Dashboard (KPIs, filtros y gráficos)
# ====================================================
@app.callback(
    Output("kpi-cards", "children"),
    Output("gauge-readmit", "figure"),
    Output("scatter-meds-days", "figure"),
    Output("bar-admission-readmit", "figure"),
    Output("heatmap-age-diagnoses", "figure"),
    Output("filtered-table", "children"),
    Input("filter-age", "value"),
    Input("filter-admission", "value"),
    Input("filter-insulin", "value"),
    Input("filter-gender", "value"),
    Input("filter-readmit", "value"),
    Input("filter-days", "value")
)
def actualizar_dashboard(age, admission_type, insulin, gender, readmit_filter, hospital_range):
    dff = df.copy()

    # Aplicar filtros (igual que en codigo_entrega2.txt)
    try:
        if age and age != "Todos":
            dff = dff[dff['age'] == age]
        if admission_type and admission_type != "Todos":
            dff = dff[dff['admission_type'] == admission_type]
        if insulin and insulin != "Todos":
            dff = dff[dff['insulin'] == insulin]
        if gender and gender != "Todos":
            dff = dff[dff['gender'] == gender]
        if readmit_filter and readmit_filter != "Todos":
            dff = dff[dff['readmitted'] == readmit_filter]
        if hospital_range and 'time_in_hospital' in dff.columns:
            dff = dff[(dff['time_in_hospital'] >= hospital_range[0]) & (dff['time_in_hospital'] <= hospital_range[1])]
    except Exception as e:
        # En caso de error en los filtros, retornar vacíos
        pass

    if dff.empty:
        # Mensaje de advertencia y figuras vacías
        warning = dbc.Alert(" No hay datos que coincidan con los filtros seleccionados.", color="warning")
        return [warning], go.Figure(), go.Figure(), go.Figure(), go.Figure(), html.Div()

    # KPIs
    cant_personas = len(dff)
    tasa_readmit = round((dff['readmitted'].astype(str).str.upper() != 'NO').mean() * 100, 2) if 'readmitted' in dff.columns else 0
    prom_estancia = round(dff['time_in_hospital'].mean(), 2) if 'time_in_hospital' in dff.columns else None
    prom_meds = round(dff['num_medications'].mean(), 2) if 'num_medications' in dff.columns else None

    cards = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Cantidad de Pacientes"), html.H3(f"{cant_personas:,}")]), className="mb-2"), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Tasa de Readmisión (%)"), html.H3(f"{tasa_readmit}%")]), className="mb-2"), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Promedio Estancia (días)"), html.H3(f"{prom_estancia if prom_estancia is not None else 'N/A'}")]), className="mb-2"), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Promedio de Medicamentos"), html.H3(f"{prom_meds if prom_meds is not None else 'N/A'}")]), className="mb-3"), md=3),
    ], className="mb-4")

    # Gauge (Indicator) - Tasa readmisión
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=tasa_readmit,
        title={'text': "Tasa Readmisión (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "royalblue"},
            'steps': [
                {'range': [0, 20], 'color': "#b8e994"},
                {'range': [20, 40], 'color': "#f6e58d"},
                {'range': [40, 100], 'color': "#ff7979"}
            ]
        }
    ))
    fig_gauge.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))

    # Scatter meds vs time_in_hospital
    if 'num_medications' in dff.columns and 'time_in_hospital' in dff.columns:
        fig_scatter = px.scatter(
            dff, x='num_medications', y='time_in_hospital',
            color='readmitted' if 'readmitted' in dff.columns else None,
            size='number_diagnoses' if 'number_diagnoses' in dff.columns else None,
            hover_data=['encounter_id', 'age', 'gender'] if 'encounter_id' in dff.columns else None,
            title="Medicaciones vs Tiempo en Hospital"
        )
        fig_scatter.update_layout(height=400, template="plotly_white")
    else:
        fig_scatter = go.Figure()

    # Bar: tasa de readmisión por tipo de admisión
    if 'admission_type' in dff.columns and 'readmitted' in dff.columns:
        df_bar = dff.groupby('admission_type')['readmitted'].apply(lambda x: (x.astype(str).str.upper() != 'NO').mean() * 100).reset_index()
        fig_bar = px.bar(df_bar, x='admission_type', y='readmitted', color='admission_type',
                         title="Tasa de Readmisión por Tipo de Admisión")
        fig_bar.update_layout(height=400, template="plotly_white")
    else:
        fig_bar = go.Figure()

    # Heatmap: edad vs number_diagnoses -> riesgo (proporción readmitted)
    if 'age' in dff.columns and 'number_diagnoses' in dff.columns and 'readmitted' in dff.columns:
        df_heat = dff.groupby(['age', 'number_diagnoses'])['readmitted'].apply(lambda x: (x.astype(str).str.upper() != 'NO').mean()).reset_index()
        # pivot para heatmap; manejar tipos mixtos en age
        try:
            heat_pivot = df_heat.pivot(index='age', columns='number_diagnoses', values='readmitted')
            fig_heat = px.imshow(heat_pivot, labels=dict(x="Número Diagnósticos", y="Edad", color="Prob. Readmisión"),
                                 title="Riesgo de Readmisión: Edad vs Diagnósticos")
            fig_heat.update_layout(height=400, template="plotly_white")
        except:
            fig_heat = go.Figure()
    else:
        fig_heat = go.Figure()

    # Tabla resumen: columnas importantes
    cols_to_show = [c for c in ["encounter_id", "age", "gender", "time_in_hospital", "num_medications", "readmitted"] if c in dff.columns]
    table = dash_table.DataTable(
        data=dff[cols_to_show].to_dict('records'),
        columns=[{"name": c, "id": c} for c in cols_to_show],
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "5px"},
    )

    return cards, fig_gauge, fig_scatter, fig_bar, fig_heat, table

# -------------------------------------
# Ejecutar app
# -------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
