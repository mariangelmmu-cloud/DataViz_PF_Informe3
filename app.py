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
import plotly.graph_objects as go

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE], suppress_callback_exceptions=True)
app.title = "Dashboard de Riesgo de Readmisión"
server = app.server

df = pd.read_csv("data_nombres.csv", sep=",")

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

# -------------------------------------------------
# CONTENIDO DE LOS SUB-PANELES
# -------------------------------------------------

TAB_1_FALTANTES = html.Div([
    html.H4("Tratamiento de Valores Faltantes", className="text-primary fw-bold"),
    html.Hr(),
    dbc.Row([
        dbc.Col(dbc.Alert([html.H3("<10%"), " Imputación"], color="success"), width=4),
        dbc.Col(dbc.Alert([html.H3("30%–90%"), " Recodificación como 'Desconocido'"], color="warning"), width=4),
        dbc.Col(dbc.Alert([html.H3(">90%"), " Eliminación"], color="danger"), width=4),
    ], className="text-center mb-3"),
    html.P("La estrategia se definió según el porcentaje de pérdida por columna, "
           "buscando minimizar sesgos y preservar la información clínica relevante.")
])


TAB_2_INCONSISTENCIAS = html.Div([
    html.H4("Corrección de Inconsistencias", className="text-primary fw-bold"),
    html.Hr(),

    # CONTENEDOR PRINCIPAL (ICONO + TEXTO + IMAGEN)
    html.Div([
        
        # Ícono grande
        html.I(className="bi bi-shield-check display-4 text-info me-4"),

        # TEXTO EXPLICATIVO
        html.Div([
            html.H5("Validación y Depuración del Dato", className="fw-bold mb-2"),
            html.P(
                "Se aplicó un proceso de corrección y estandarización destinado a garantizar la coherencia "
                "de todas las variables del dataset. Esto incluyó la depuración de valores no válidos, "
                "la asignación correcta de tipos (numéricos o categóricos) y la normalización de categorías "
                "inconsistentes o mal codificadas.",
                className="mb-2"
            ),
            html.Ul([
                html.Li("Conversión de registros numéricos mal almacenados como texto."),
                html.Li("Unificación de categorías inconsistentes en variables clínicas y demográficas."),
                html.Li("Estandarización de códigos no válidos como 'Unknown'.")
            ], className="mb-0")
        ], className="flex-grow-1"),

        # AYUDA VISUAL (IMAGEN)
        html.Div([
            html.Img(
                src="/assets/data_cleaning.png",  # <- Reemplaza este archivo por el tuyo
                style={
                    "width": "120px",
                    "borderRadius": "10px",
                    "boxShadow": "0px 3px 8px rgba(0,0,0,0.15)"
                }
            )
        ], className="ms-4")
    ], 
    className="d-flex align-items-center p-3 bg-light rounded shadow-sm")

])



TAB_3_TRANSFORMACION = html.Div([
    html.H4("Transformación de Variables", className="text-primary fw-bold"),
    html.Hr(),

    html.Div([
        # Ícono representativo
        html.I(className="bi bi-gear-fill display-4 text-warning me-4"),

        # TEXTO PRINCIPAL
        html.Div([
            html.H5("Feature Engineering aplicado al dataset", className="fw-bold mb-2"),

            html.P(
                "Se realizaron transformaciones orientadas a mejorar la representación de las "
                "variables clínicas y a facilitar el desempeño de los modelos predictivos. "
                "Estas modificaciones permiten reducir dimensionalidad, corregir ruido y "
                "proveer una estructura más interpretativa para los algoritmos.",
                className="mb-3"
            ),

            html.Ul([
                html.Li([
                    html.B("Agrupación Clínica de Diagnósticos: "),
                    "Los códigos ICD-9 presentes en diag_1, diag_2 y diag_3 fueron reclasificados en "
                    "categorías clínicas amplias (cardiovasculares, respiratorias, metabólicas, etc.), "
                    "reduciendo la alta cardinalidad y permitiendo un análisis más interpretable."
                ]),
                html.Li([
                    html.B("Binarización del Target: "),
                    "La variable readmitted fue convertida a un formato binario para el modelado: ",
                    html.Span("0 = no readmisión, "),
                    html.Span("1 = readmisión (<30 días o >30 días).")
                ])
            ])
        ], className="flex-grow-1"),

        # IMAGEN ACOMPAÑANTE (AYUDA VISUAL)
        html.Div([
            html.Img(
                src="/assets/feature_engineering.png",  # Cambia la imagen según lo que tengas en assets
                style={
                    "width": "130px",
                    "borderRadius": "12px",
                    "boxShadow": "0px 3px 10px rgba(0,0,0,0.15)"
                }
            )
        ], className="ms-4"),

    ], className="d-flex align-items-center p-3 bg-light rounded shadow-sm")
])



TAB_4_ENCODING = html.Div([
    html.H4("Encoding y Escalamiento", className="text-primary fw-bold"),
    html.Hr(),

    html.Div([
        # Texto de explicación
        html.Div([
            html.P(
                "Se aplicaron dos técnicas fundamentales: One-Hot Encoding para "
                "variables categóricas y StandardScaler para variables numéricas.",
                className="mb-3"
            ),

            html.H6("Variables Categóricas", className="fw-bold"),
            html.Div([
                html.I(className="bi bi-grid-3x3-gap-fill text-primary me-2"),
                html.Span("One-Hot Encoding: conversión de categorías en variables binarias.")
            ], className="d-flex mb-3"),

            html.H6("Variables Numéricas", className="fw-bold"),
            html.Div([
                html.I(className="bi bi-graph-down text-info me-2"),
                html.Span("StandardScaler: normalización Z-score (media = 0, desviación estándar = 1).")
            ], className="d-flex")
        ], className="flex-grow-1"),

        # Imagen visual
        html.Div([
            html.Img(
                src="/assets/encoding_scaling.png",
                style={
                    "width": "160px",
                    "borderRadius": "10px",
                    "boxShadow": "0px 3px 10px rgba(0,0,0,0.15)",
                }
            )
        ], className="ms-4")

    ], className="d-flex align-items-center p-3 bg-light rounded shadow-sm")
])



# --- 1. Gráfico Visual del Split Train/Test (Barra Horizontal) --- el gráfico de Split (Train/Test) por si no lo definiste antes
fig_split_mini = go.Figure()

fig_split_mini.add_trace(go.Bar(
    x=["Train", "Test"],
    y=[80, 20],
    text=["80%", "20%"],
    textposition="auto"
))

fig_split_mini.update_layout(
    title="Split del Dataset (Train 80% / Test 20%)",
    xaxis_title="Subset",
    yaxis_title="Porcentaje",
    template="plotly_white",
    margin=dict(l=40, r=40, t=50, b=40)
)
# IMPORTANTE: asegúrate de tener definida la figura fig_split_mini ANTES de esta sección
TAB_5_SPLIT = html.Div([
    html.H4("División del Dataset", className="text-primary fw-bold"),
    html.Hr(),
    dcc.Graph(figure=fig_split_mini, config={'displayModeBar': False}),
    html.P("La estrategia 80/20 garantiza un balance adecuado entre entrenamiento y evaluación.",
           className="mt-3 text-muted")
])
# -------------------------------------------------
import plotly.graph_objects as go
from dash import dcc

# Definición del Diagrama de Flujo (Sankey) - Paleta AZUL
fig_sankey = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15, thickness = 20,
      line = dict(color = "white", width = 0.5), # Línea blanca queda mejor con azules
      label = ["NO Readmitido", ">30 Días", "<30 Días", "Clase 0 (NO)", "Clase 1 (SI)"],
      # Paleta de Azules:
      color = [
          "#4FC3F7",  # Nodo 0: NO (Azul Cielo/Claro)
          "#1E88E5",  # Nodo 1: >30 (Azul Medio)
          "#0D47A1",  # Nodo 2: <30 (Azul Oscuro)
          "#4FC3F7",  # Nodo 3: Clase 0 (Coincide con el origen)
          "#1565C0"   # Nodo 4: Clase 1 (Azul Intenso - la suma)
      ]
    ),
    link = dict(
      source = [0, 1, 2], 
      target = [3, 4, 4], 
      value =  [50, 25, 25],
      # Opcional: Hacer los flujos semitransparentes para que se vea elegante
      color = ["rgba(79, 195, 247, 0.4)", "rgba(30, 136, 229, 0.4)", "rgba(13, 71, 161, 0.4)"]
  ))])

fig_sankey.update_layout(
    margin=dict(l=10, r=10, t=30, b=10),
    height=250,
    font_size=12,
    title_text="<b>Flujo de Transformación de Variable</b>",
    title_font_size=20
)


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
        dbc.Button("1. Introducción", id="btn-1", color="light", className="main-nav-btn"),
        dbc.Button("2. Contexto", id="btn-2", color="light", className="main-nav-btn"),
        dbc.Button("3. Planteamiento del Problema", id="btn-3", color="light", className="main-nav-btn"),
        dbc.Button("4. Objetivos y Justificación", id="btn-4", color="light", className="main-nav-btn"),
        dbc.Button("5. Marco Teórico", id="btn-5", color="light", className="main-nav-btn"),
        dbc.Button("6. Metodología", id="btn-6", color="light", className="main-nav-btn"),
        dbc.Button("7. Resultados/Análisis Final", id="btn-7", color="light", className="main-nav-btn"),
        dbc.Button("8. Conclusiones", id="btn-8", color="light", className="main-nav-btn"),
    ],
    className="d-flex justify-content-around mb-4 flex-wrap gap-2",
)

app.layout = dbc.Container(
    [
        html.H1("Dashboard de Riesgo de Readmisión", className="text-center fw-bold text-primary mb-4"),
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
    Output("vertical-content", "children"),
    [
        Input("vtab-1", "n_clicks"),
        Input("vtab-2", "n_clicks"),
        Input("vtab-3", "n_clicks"),
        Input("vtab-4", "n_clicks"),
        Input("vtab-5", "n_clicks"),
    ]
)
def update_vertical_tab(*args):
    ctx = dash.callback_context

    if not ctx.triggered:
        tab = "vtab-1"
    else:
        tab = ctx.triggered[0]["prop_id"].split(".")[0]

    if tab == "vtab-1":
        return TAB_1_FALTANTES
    elif tab == "vtab-2":
        return TAB_2_INCONSISTENCIAS
    elif tab == "vtab-3":
        return TAB_3_TRANSFORMACION
    elif tab == "vtab-4":
        return TAB_4_ENCODING
    elif tab == "vtab-5":
        return TAB_5_SPLIT
    
@app.callback(
    Output("contenido-modelo", "children"),
    Input("tabs-modelos", "active_tab")
)
def actualizar_modelo(tab):

    if tab == "tab-lr":
        return dbc.Row([
            dbc.Col([
                html.H4("Logistic Regression", className="text-primary mt-2"),
                dbc.Badge("Interpretabilidad: Alta", color="success", className="me-2"),
                dbc.Badge("Base Line", color="secondary"),
                html.H6("Justificación:", className="fw-bold mt-3"),
                html.P("Modelo base por excelencia. Permite interpretar la contribución de cada variable al riesgo de readmisión. Es un estándar en salud pública para problemas de clasificación binaria."),
            ], md=5),
            dbc.Col([
                html.H6("Formulación Matemática:", className="fw-bold text-center"),
                dcc.Markdown(r'''
                    La probabilidad de que un paciente sea readmitido se modela como:

                    $$P(Y=1|X) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

                    donde el puntaje lineal es:

                    $$z = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n$$

                    **Regla de decisión:**

                    $$
                    \hat{y} = 
                    \begin{cases}
                    1, & \text{si } \sigma(z) \geq 0.5 \\
                    0, & \text{si } \sigma(z) < 0.5
                    \end{cases}
                    $$
                    ''',
                    mathjax=True,
                    style={"fontSize": "16px"}
                    )], md=7)
        ])

    if tab == "tab-dt":
        return dbc.Row([
            dbc.Col([
                html.H4("Decision Tree", className="text-primary mt-2"),
                dbc.Badge("No Lineal", color="warning", text_color="dark", className="me-2"),
                dbc.Badge("Reglas Claras", color="info"),
                html.H6("Justificación:", className="fw-bold mt-3"),
                html.P("Permite capturar relaciones no lineales entre variables, interpretar la estructura de decisión y manejar datos mixtos (numéricos y categóricos) sin necesidad de escalamiento intenso.")
            ], md=5),
            dbc.Col([
                html.H6("Formulación Matemática:", className="fw-bold text-center"),
                dcc.Markdown(r'''
                El modelo construye reglas de decisión basadas en divisiones sucesivas del espacio de características:

                $$\text{Si } x_j \leq t \Rightarrow \text{rama izquierda}; \quad \text{si } x_j > t \Rightarrow \text{rama derecha}$$

                **Criterio de impureza (Gini):**

                $$G(t) = 1 - \sum_{k=1}^{K} p_k^2$$

                donde $p_k$ es la proporción de clase en el nodo.

                **Parámetros:**
                Profundidad máxima, número mínimo de muestras por hoja, criterio (Gini/Entropía).
                ''', mathjax=True)
            ], md=7)
        ])

    if tab == "tab-svm":
        return dbc.Row([
            dbc.Col([
                html.H4("Support Vector Machine", className="text-primary mt-2"),
                dbc.Badge("Alta Dimensionalidad", color="dark", className="me-2"),
                dbc.Badge("Robusto", color="primary"),
                html.H6("Justificación:", className="fw-bold mt-3"),
                html.P("Maneja problemas de alta dimensionalidad, es la opción cuando se busca maximizar la separación entre clases. Su capacidad para trabajar con márgenes amplios lo hace robusto ante datos con límites complejos.")
            ], md=5),
            dbc.Col([
                html.H6("Formulación Matemática:", className="fw-bold text-center"),
               dcc.Markdown(r'''
                El modelo SVM busca encontrar un hiperplano que maximice el margen entre clases:

                $$\min_{\mathbf{w},b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i$$

                **Sujeto a:**

                $$y_i(\mathbf{w}^\top x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

                **Kernel radial (RBF):**

                $$K(x_i, x_j) = e^{-\gamma \|x_i - x_j\|^2}$$

                **Parámetros:**
                Penalidad $C$, parámetro del kernel $\gamma$.
                ''', mathjax=True)
            ], md=7)
        ])

    if tab == "tab-xgb":
        return dbc.Row([
            dbc.Col([
                html.H4("Extreme Gradient Boosting", className="text-primary mt-2"),
                dbc.Badge("SOTA", color="danger", className="me-2"),
                dbc.Badge("Alto Rendimiento", color="primary"),
                html.H6("Justificación:", className="fw-bold mt-3"),
                html.P("Excelente rendimiento en tareas de clasificación tabular. Este algoritmo optimiza de manera eficiente árboles con boosting secuencial, maneja desbalance en las clases y permite controlar el overfitting mediante regularización.")
            ], md=5),
            dbc.Col([
                html.H6("Formulación Matemática:", className="fw-bold text-center"),
                dcc.Markdown(r'''
                XGBoost optimiza la siguiente función objetivo:

                $$\text{Obj} = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$

                con la regularización:

                $$\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2$$

                donde:
                * $T$ = número de hojas del árbol
                * $w_j$ = peso de cada hoja

                **Predicción final (ensamble):**

                $$\hat{y}_i = \sum_{k=1}^{K} f_k(x_i)$$

                **Parámetros:**
                Learning rate $\eta$, número de árboles, profundidad máxima, $\gamma, \lambda$.
                ''', mathjax=True)
            ], md=7)
        ])

    return html.P("Seleccione un modelo.")


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
    # --- Btn 1: Introducción ---
    if boton_id == "btn-1":
        return dbc.Card(
            dbc.CardBody([
                html.H4("Introducción", className="text-center fw-bold text-primary mb-4"),

                # Caja gris
                dbc.Card(
                    dbc.CardBody([
                        html.P("""
                    La diabetes mellitus es una de las enfermedades crónicas más prevalentes a nivel global y representa un desafío significativo para los sistemas de salud debido a sus complicaciones y a la alta demanda de servicios que genera. Solo en Estados Unidos, 38.4 millones de personas —equivalentes al 11.6% de la población— viven con diabetes, y aproximadamente el 22.8% de los adultos con esta condición no han sido diagnosticados (CDC, 2024). Esta situación, caracterizada por una alta prevalencia y un importante subdiagnóstico, se relaciona directamente con un mayor riesgo de complicaciones y readmisiones hospitalarias. De hecho, la literatura señala que los pacientes con diabetes tienen una probabilidad 40% mayor de ser readmitidos al hospital en comparación con personas sin esta condición, lo que revela la gravedad del problema y la necesidad de comprender mejor sus determinantes (Endocrine Society, 2019).

                    Ante este panorama, este proyecto se enfoca en analizar los factores asociados a la readmisión hospitalaria en pacientes con diabetes, considerando tanto su impacto en la salud de los pacientes como en la gestión de los servicios médicos. Para ello, se emplea un conjunto de datos con más de cien mil registros de pacientes diabéticos, abordando un proceso de revisión de calidad, exploración de variables clínicas y demográficas, e identificación de patrones relacionados con la readmisión. A partir de este análisis, se desarrollan modelos de machine learning con el propósito de predecir el riesgo de readmisión y construir una herramienta que permita identificar oportunamente a pacientes de alto riesgo, contribuyendo a la toma de decisiones clínicas y a la optimización de los recursos hospitalarios.
                """)
                    ]),
                    color="light",      # gris suave
                    className="mb-3 fs-5",    # margen inferior
                ),

                # Imagen debajo del texto
                html.Img(
                    src="/assets/intro_img.jpg",  # tu imagen en la carpeta /assets
                    style={"width": "60%", "borderRadius": "10px","centered": True},
                )
            ])
        )


    elif boton_id == "btn-2":
        return dbc.Card(
            dbc.CardBody([

                # ---------------------------------------------------------
                # TÍTULO PRINCIPAL
                # ---------------------------------------------------------
                html.H3(
                    "Contexto General del Problema",
                    className="text-center fw-bold text-primary mb-4"
                ),

                # ---------------------------------------------------------
                # SECCIÓN 1: Imagen + Overlay + Texto
                # ---------------------------------------------------------
                html.Div([
                    html.Div([
                        html.Img(
                            src="/assets/contexto.png",
                            className="img-fluid rounded shadow",
                            style={
                                "width": "100%",
                                "maxHeight": "320px",
                                "objectFit": "cover",
                                "filter": "brightness(85%)"
                            }
                        ),
                    ], className="position-relative"),

                    # Overlay de texto sobre la imagen
                    html.Div(
                        html.Div([
                            html.I(className="bi bi-hospital fs-2 me-2 text-white"),
                            html.Span(
                                "Panorama Hospitalario y la Carga de la Diabetes",
                                className="fw-bold text-white fs-4"
                            )
                        ], className="d-flex align-items-center"),
                        style={
                            "position": "absolute",
                            "bottom": "15px",
                            "left": "20px",
                            "background": "rgba(0,0,0,0.45)",
                            "padding": "8px 15px",
                            "borderRadius": "8px"
                        }
                    )
                ], className="mb-4"),

                # ---------------------------------------------------------
                # SECCIÓN 2: TARJETA DE NARRATIVA PRINCIPAL
                # ---------------------------------------------------------
                dbc.Card(
                    dbc.CardBody([
                        html.P(
                            """
                            Los reingresos hospitalarios en pacientes con diabetes representan uno de los 
                            principales retos para los sistemas de salud modernos. Su presencia suele reflejar 
                            problemas en el seguimiento clínico, en el control metabólico o en la continuidad 
                            de la atención entre los distintos niveles asistenciales. Analizar estos patrones 
                            permite no solo comprender mejor el fenómeno, sino también diseñar intervenciones 
                            más oportunas y optimizar recursos hospitalarios.
                            """,
                            className="text-muted fs-6 mb-0",
                            style={"textAlign": "justify"}
                        )
                    ]),
                    className="shadow-sm border-0 mb-4",
                    style={"backgroundColor": "#f8f9fa", "borderRadius": "12px"}
                ),

                html.Hr(className="my-4"),

                # ---------------------------------------------------------
                # SECCIÓN 3: MINI TARJETAS DE INFORMACIÓN (KPIs Estáticos)
                # ---------------------------------------------------------
                html.H4("Datos Relevantes del Dataset", className="fw-bold text-primary mb-3"),

                dbc.Row([
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.I(className="bi bi-clipboard2-pulse text-primary fs-1"),
                                html.H5("100,000+", className="fw-bold mt-2"),
                                html.P("Registros hospitalarios", className="text-muted")
                            ], className="text-center"),
                        className="shadow-sm h-100"
                        ), md=4
                    ),

                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.I(className="bi bi-calendar-range text-success fs-1"),
                                html.H5("1999 - 2008", className="fw-bold mt-2"),
                                html.P("Periodo de observación", className="text-muted")
                            ], className="text-center"),
                        className="shadow-sm h-100"
                        ), md=4
                    ),

                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.I(className="bi bi-diagram-3 text-warning fs-1"),
                                html.H5("130 hospitales", className="fw-bold mt-2"),
                                html.P("Diversidad en la procedencia", className="text-muted")
                            ], className="text-center"),
                        className="shadow-sm h-100"
                        ), md=4
                    ),
                ], className="mb-4"),

                # ---------------------------------------------------------
                # SECCIÓN 4: TARJETA DE FUENTE DE DATOS
                # ---------------------------------------------------------
                dbc.Card([
                    dbc.CardHeader(
                        html.H5("Fuente de los Datos", className="fw-bold text-center"),
                        className="bg-white border-bottom-0"
                    ),
                    dbc.CardBody([
                        html.P(
                            "El dataset proviene de los Centers for Medicare & Medicaid Services (CMS), "
                            "y fue publicado a través del UCI Machine Learning Repository.",
                            className="text-muted"
                        ),
                        html.P([
                            "Acceso oficial: ",
                            html.A(
                                "UCI Repository Link",
                                href="https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008",
                                target="_blank",
                                className="fw-bold text-decoration-none"
                            )
                        ]),
                    ])
                ], className="shadow-sm border-0 bg-light bg-opacity-50"),

            ])
        , className="border-0 shadow-sm")

# --- Btn 3: Planteamiento del Problema ---
    elif boton_id == "btn-3":
        return html.Div([
            
            # ---------------------------------------------------------
            # SECCIÓN 1: EL DESAFÍO (Imagen + Texto lado a lado)
            # ---------------------------------------------------------
            dbc.Card([
                dbc.CardBody([
                    html.H4("Planteamiento del Problema", className="card-title text-primary fw-bold mb-4"),
                    
                    dbc.Row([
                        # Columna Izquierda: Imagen
                        dbc.Col([
                            html.Img(
                                src="assets/planteamiento.png",   # <-- tu imagen
                                className="img-fluid rounded shadow-sm", # img-fluid la hace responsiva automáticamente
                                style={"width": "100%", "objectFit": "cover"}
                            )
                        ], width=12, md=5, className="mb-3 mb-md-0 d-flex align-items-center"),

                        # Columna Derecha: Texto descriptivo
                        dbc.Col([
                            html.P('''
                                El reingreso hospitalario en pacientes con diabetes es un desafío crítico para los sistemas de salud. 
                                Esta situación combina una alta prevalencia de la enfermedad, un riesgo elevado de readmisión 
                                y un impacto importante en la eficiencia de los servicios hospitalarios.
                            ''', className="text-justify text-muted fs-4"),
                            
                            html.P('''
                                A pesar de contar con bases de datos amplias, aún es necesario identificar los factores que más 
                                contribuyen al reingreso y desarrollar herramientas predictivas que permitan anticipar estos eventos.
                                Comprender estas variables es clave para optimizar el cuidado, reducir costos y mejorar los resultados clínicos.
                            ''', className="text-justify text-muted fs-4")
                        ], width=12, md=7)
                    ])
                ])
            ], className="shadow-sm border-0 mb-4 bg-white"),

            # ---------------------------------------------------------
            # SECCIÓN 2: INDICADORES (3 Columnas)
            # ---------------------------------------------------------
            html.H5("Indicadores Clave", className="text-secondary fw-bold mb-3"),
            
            dbc.Row([
                # Tarjeta 1: Prevalencia (Rojo/Danger)
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6([html.I(className="bi bi-people-fill me-2 text-danger"), "Alta Prevalencia"], className="fw-bold"),
                        html.Ul([
                            html.Li("38.4 millones de personas viven con diabetes en Estados Unidos."),
                            html.Li("La alta prevalencia incrementa la demanda hospitalaria y el riesgo de complicaciones."),
                        ], className="small text-muted mb-2 ps-3 fs-5"),
                        html.Div("Fuente: CDC, 2024", className="text-end fst-italic text-danger small", style={"fontSize": "0.75rem"})
                    ])
                ], className="h-100 shadow-sm border-top border-4 border-danger"), width=12, md=4, className="mb-3"),

                # Tarjeta 2: Probabilidad (Naranja/Warning)
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6([html.I(className="bi bi-exclamation-triangle-fill me-2 text-warning"), "Probabilidad de Readmisión"], className="fw-bold"),
                        html.Ul([
                            html.Li("Los pacientes diabéticos presentan un 40% más riesgo de reingreso hospitalario."),
                            html.Li("Este comportamiento refleja fallas en el control, adherencia y continuidad del cuidado."),
                        ], className="small text-muted mb-2 ps-3 fs-5"),
                        html.Div("Fuente: Endocrine Society, 2019", className="text-end fst-italic text-warning small", style={"fontSize": "0.75rem"})
                    ])
                ], className="h-100 shadow-sm border-top border-4 border-warning"), width=12, md=4, className="mb-3"),

                # Tarjeta 3: Gestión (Azul/Info)
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6([html.I(className="bi bi-hospital-fill me-2 text-info"), "Impacto en Gestión"], className="fw-bold"),
                        html.Ul([
                            html.Li("Los reingresos generan mayores costos y saturación de servicios."),
                            html.Li("Identificar factores de riesgo permite mejorar decisiones clínicas y operativas."),
                        ], className="small text-muted mb-2 ps-3 fs-5"),
                        html.Div("Fuente: Krishnan et al., 2022", className="text-end fst-italic text-info small", style={"fontSize": "0.75rem"})
                    ])
                ], className="h-100 shadow-sm border-top border-4 border-info"), width=12, md=4, className="mb-3"),
            ], className="mb-4"),

            # ---------------------------------------------------------
            # SECCIÓN 3: PREGUNTA DE INVESTIGACIÓN (Destacado)
            # ---------------------------------------------------------
            dbc.Card([
                dbc.CardBody([
                    html.H5([html.I(className="bi bi-search me-2"), "Pregunta de Investigación"], className="card-title text-primary fw-bold text-center mb-3"),
                    html.Blockquote('''
                        ¿Qué factores clínicos, demográficos y relacionados con el uso del sistema de salud influyen 
                        en el reingreso hospitalario de pacientes con diabetes? 
                        ¿Y en qué medida los modelos de machine learning pueden predecir este riesgo con precisión 
                        para apoyar la toma de decisiones clínicas?
                    ''', className="blockquote text-center fs-5 text-dark mb-0")
                ])
            ], className="shadow-sm border-start border-5 border-primary bg-light")

        ], className="px-2")

    # --- Btn 4: Objetivos y Justificación ---
    elif boton_id == "btn-4":
        return html.Div([
            
            # ---------------------------------------------------------
            # SECCIÓN 1: OBJETIVOS
            # ---------------------------------------------------------
            html.H4("Objetivos del Proyecto", className="text-primary fw-bold text-center mb-4"),

            # A. OBJETIVO GENERAL (Tarjeta Destacada)
            dbc.Card([
                dbc.CardHeader(html.H5("Objetivo General", className="m-0 fw-bold text-white"), className="bg-primary"),
                dbc.CardBody([
                    html.P(
                        "Desarrollar un modelo predictivo basado en técnicas de machine learning para identificar el riesgo de readmisión hospitalaria en pacientes con diabetes, utilizando variables clínicas, demográficas y del uso de servicios de salud provenientes de un conjunto de más de cien mil registros.",
                        className="lead fs-6 mb-0 text-dark"
                    )
                ])
            ], className="shadow-sm border-0 mb-4"),

            # B. OBJETIVOS ESPECÍFICOS (Grid 2x2)
            html.H5("Objetivos Específicos", className="text-primary fw-bold mb-3"),
            dbc.Row([
                # Objetivo 1
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.Div(html.I(className="bi bi-1-circle-fill text-info fs-4 me-2"), className="mb-2"),
                        html.P("Realizar un Análisis Exploratorio de Datos (EDA) para examinar la calidad, distribución y comportamiento de las variables asociadas a la readmisión.", className="small text-muted mb-0")
                    ])
                ], className="h-100 shadow-sm border-start border-4 border-info fs-4 text-center"), width=12, md=6, className="mb-3"),

                # Objetivo 2
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.Div(html.I(className="bi bi-2-circle-fill text-warning fs-4 me-2"), className="mb-2"),
                        html.P("Identificar los principales factores que influyen en el reingreso hospitalario, considerando variables clínicas, del paciente y hospitalarias.", className="small text-muted mb-0")
                    ])
                ], className="h-100 shadow-sm border-start border-4 border-warning fs-4 text-center"), width=12, md=6, className="mb-3"),

                # Objetivo 3
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.Div(html.I(className="bi bi-3-circle-fill text-danger fs-4 me-2"), className="mb-2"),
                        html.P("Entrenar y comparar modelos de machine learning como Logistic Regression, SVM, Decision Tree, XGBoost, Random Forest y CatBoost, optimizando hiperparámetros y eligiendo el mejor desempeño.", className="small text-muted mb-0")
                    ])
                ], className="h-100 shadow-sm border-start border-4 border-danger fs-4 text-center"), width=12, md=6, className="mb-3"),

                # Objetivo 4
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.Div(html.I(className="bi bi-4-circle-fill text-success fs-4 me-2"), className="mb-2"),
                        html.P("Desarrollar una herramienta predictiva integrada en un dashboard interactivo que permita estimar el riesgo de readmisión para apoyar decisiones clínicas.", className="small text-muted mb-0")
                    ])
                ], className="h-100 shadow-sm border-start border-4 border-success fs-4 text-center"), width=12, md=6, className="mb-3"),
            ], className="mb-4"),

            html.Hr(),

            # ---------------------------------------------------------
            # SECCIÓN 2: JUSTIFICACIÓN
            # ---------------------------------------------------------
            html.H4("Justificación", className="text-primary fw-bold text-center mb-4"),

            dbc.Row([
                # Párrafo 1: El Problema
                dbc.Col([
                    html.Div(className="p-3 border rounded h-100 bg-white shadow-sm", children=[
                        html.H6([html.I(className="bi bi-exclamation-circle me-2 text-primary fs-3"), "Contexto"], className="fw-bold mb-3 fs-3"),
                        html.P(
                            "La readmisión hospitalaria en pacientes con diabetes representa un desafío importante para los sistemas de salud debido a su impacto en la calidad del cuidado, los costos hospitalarios y la eficiencia operativa. Los estudios muestran que los pacientes diabéticos tienen un 40% más probabilidad de ser readmitidos que la población no diabética, lo que evidencia la necesidad de estrategias predictivas y preventivas más efectivas.",
                            className="text-muted small text-justify fs-4"
                        ),
                    ])
                ], width=12, lg=4, className="mb-3"),

                # Párrafo 2: El Análisis
                dbc.Col([
                    html.Div(className="p-3 border rounded h-100 bg-white shadow-sm", children=[
                        html.H6([html.I(className="bi bi-search me-2 text-primary "), "Análisis"], className="fw-bold mb-3 fs-3"),
                        html.P(
                            "Un análisis profundo de un conjunto de datos hospitalarios de gran escala permite identificar patrones clínicos y administrativos que influyen en el reingreso, generando información valiosa para mejorar la continuidad del cuidado. Además, los modelos de machine learning permiten anticipar eventos de readmisión con mayor precisión, facilitando intervenciones tempranas y una asignación más eficiente de recursos.",
                            className="text-muted small text-justify fs-4"
                        ),
                    ])
                ], width=12, lg=4, className="mb-3"),

                # Párrafo 3: El Aporte
                dbc.Col([
                    html.Div(className="p-3 border rounded h-100 bg-white shadow-sm", children=[
                        html.H6([html.I(className="bi bi-check-circle me-2 text-primary fs-3"), "Aporte"], className="fw-bold mb-4 fs-3"),
                        html.P(
                            "Por estas razones, este proyecto aporta tanto al ámbito clínico como al administrativo, al proponer una herramienta útil para reducir complicaciones, mejorar la atención del paciente y optimizar la gestión hospitalaria.",
                            className="text-muted small text-justify fs-4"
                        ),
                    ])
                ], width=12, lg=4, className="mb-3"),
            ])

        ], className="px-2")


    # --- Btn 5: Marco Teórico ---
    elif boton_id == "btn-5":
        return dbc.Card(
            dbc.CardBody([
                html.H4("Marco Teórico", className="card-title mb-4 text-center fw-bold"),

                # 1. Diabetes Mellitus
                dbc.Card(
                    dbc.CardBody([
                        html.H5("1. Diabetes Mellitus", className="text-primary fw-bold mb-4"),
                        html.P(
                            "La diabetes mellitus es una enfermedad metabólica crónica caracterizada "
                            "por niveles elevados de glucosa debido a alteraciones en la secreción o acción "
                            "de la insulina. Esta condición incrementa la probabilidad de complicaciones "
                            "y hospitalizaciones, especialmente cuando no existe un control adecuado."
                        ),
                    ]),
                    color="light",
                    className="mb-3 shadow-sm"
                ),

                # 2. Readmisión Hospitalaria
                dbc.Card(
                    dbc.CardBody([
                        html.H5("2. Readmisión Hospitalaria", className="text-primary fw-bold mb-4"),
                        html.P(
                            "La readmisión hospitalaria corresponde al reingreso no planificado de un paciente "
                            "dentro de los 30 días posteriores al alta. Este indicador se usa para evaluar la "
                            "calidad del cuidado, la continuidad del tratamiento y la eficacia en la transición "
                            "del paciente hacia el manejo ambulatorio."
                        ),
                    ]),
                    color="light",
                    className="mb-3 shadow-sm"
                ),

                # 3. Factores Asociados al Reingreso
                dbc.Card(
                    dbc.CardBody([
                        html.H5("3. Factores Asociados al Reingreso", className="text-primary fw-bold mb-4"),
                        html.P(
                            "El riesgo de readmisión depende de variables clínicas (control glucémico, "
                            "alteraciones de laboratorio, número de diagnósticos), sociodemográficas "
                            "(edad, desigualdad en acceso), y administrativas (tipo de admisión, "
                            "disposición al egreso, historial hospitalario). Los pacientes con "
                            "diabetes presentan un riesgo particularmente elevado."
                        ),
                    ]),
                    color="light",
                    className="mb-3 shadow-sm"
                ),

                # 4. Modelos Predictivos en Salud
                dbc.Card(
                    dbc.CardBody([
                        html.H5("4. Modelos Predictivos en Salud", className="text-primary fw-bold mb-4"),
                        html.P(
                            "Los modelos de machine learning permiten identificar patrones y predecir "
                            "eventos clínicos como readmisiones. Entre los modelos usados se encuentran "
                            "Regresión Logística, Árboles de Decisión, Random Forest, SVM, XGBoost y CatBoost, "
                            "que manejan relaciones complejas propias de los datos clínicos."
                        ),
                    ]),
                    color="light",
                    className="mb-3 shadow-sm"
                ),

                # 5. Importancia de Predecir Readmisiones
                dbc.Card(
                    dbc.CardBody([
                        html.H5("5. Importancia de la Predicción", className="text-primary fw-bold mb-4"),
                        html.P(
                            "Predecir el riesgo de readmisión permite identificar pacientes vulnerables, "
                            "optimizar recursos, reducir costos hospitalarios y mejorar la calidad del cuidado. "
                            "Es una herramienta clave para la toma de decisiones en salud pública."
                        ),
                    ]),
                    color="light",
                    className="mb-3 shadow-sm"
                ),
            ])
        )


    # --- Btn 6: Metodología ---
    elif boton_id == "btn-6":
        return dbc.Card([
            dbc.CardBody([
                
                # Título principal de la sección
                html.H3("Metodología", className="card-title mb-4 text-center fw-bold"),

                # --- AQUI COMIENZAN LAS PESTAÑAS ---
                dbc.Tabs([
                    
                    # -------------------------------------------------
                    # PESTAÑA A: Definición del Problema
                    # -------------------------------------------------
                    dbc.Tab(
                        label="a. Definición del Problema",
                        children=[
                            dbc.Container([
                                dbc.Card([
                                    dbc.CardHeader(
                                        html.H4("Definición del Problema a Resolver",
                                                className="text-center text-primary fw-bold mb-4",
                                                style={"fontWeight": "600"}),
                                        style={"backgroundColor": "#f0f2f5"}
                                    ),
                                    dbc.CardBody([

                                        # --- Sección 1: Tipo de problema (Igual que antes) ---
                                        html.Div([
                                            html.H5("Tipo de Problema", className="fw-bold text-primary"),
                                            html.Div([
                                                html.I(className="bi bi-diagram-3 me-2 text-primary"),
                                                html.Span(
                                                    "Clasificación supervisada. El objetivo es predecir si un paciente diabético será "
                                                    "readmitido después del alta.",
                                                    style={"fontSize": "16px"}
                                                )
                                            ], className="d-flex align-items-start")
                                        ], className="mb-4"),

                                        html.Hr(), # Separador visual

                                        # --- Sección 2 y 3 Fusionadas: Texto + Gráfico ---
                                        html.H5("Variable Objetivo y Transformación", className="fw-bold text-primary mb-3"),
                                        
                                        dbc.Row([
                                            # Columna Izquierda: Explicación de texto
                                            dbc.Col([
                                                html.P("Variable original (readmitted):", className="fw-bold text-primary mb-3"),
                                                html.Ul([
                                                    html.Li([html.B("NO"), " → No readmitido"], className="text-success"),
                                                    html.Li([html.B(">30"), " → Readmisión > 30 días"], className="text-secondary"),
                                                    html.Li([html.B("<30"), " → Readmisión < 30 días"], className="text-danger"),
                                                ], style={"fontSize": "14px"}, className="mb-3"),
                                                
                                                html.P("Redefinición binaria para el modelo:", className="fw-bold text-primary mb-3"),
                                                dbc.Alert([
                                                    html.I(className="bi bi-arrow-right-circle-fill me-2"),
                                                    "Objetivo: Agrupar cualquier readmisión como '1'."
                                                ], color="light", className="small py-2")
                                            ], width=12, md=5), # Ocupa 5 columnas en pantallas medianas

                                            # Columna Derecha: El Gráfico Dinámico
                                            dbc.Col([
                                                dcc.Graph(
                                                    figure=fig_sankey, 
                                                    config={'displayModeBar': False} # Oculta barra de herramientas para limpieza
                                                )
                                            ], width=12, md=7) # Ocupa 7 columnas
                                        ], align="center") # Alineación vertical centrada

                                    ])
                                ], className="shadow-sm border-0", style={
                                    "backgroundColor": "white",
                                    "borderRadius": "12px"
                                }),

                            ], fluid=True, className="py-4 px-3")
                        ]
                    ),

                   # -------------------------------------------------
                # PESTAÑA B: Preparación de Datos
                # -------------------------------------------------
                dbc.Tab(
                    label="b. Preparación de Datos",
                    children=[
                        dbc.Container([
                            html.H4("Pipeline de Ingeniería de Datos",
                                    className="text-center text-primary fw-bold mb-4"),

                            dbc.Card([
                                dbc.CardBody([
                                    dbc.Row([

                                        # ---------------------------------------------
                                        # COLUMNA IZQUIERDA: MENÚ VERTICAL (PILLS)
                                        # ---------------------------------------------
                                        dbc.Col([
                                            dbc.Nav(
                                                [
                                                    dbc.NavLink("1. Valores Faltantes", id="vtab-1", active=True, n_clicks=0),
                                                    dbc.NavLink("2. Inconsistencias", id="vtab-2", n_clicks=0),
                                                    dbc.NavLink("3. Transformación", id="vtab-3", n_clicks=0),
                                                    dbc.NavLink("4. Encoding & Scaling", id="vtab-4", n_clicks=0),
                                                    dbc.NavLink("5. Split Dataset", id="vtab-5", n_clicks=0),
                                                ],
                                                vertical=True,
                                                pills=True,
                                                className="h-100"
                                            )
                                        ], width=3),

                                        # ---------------------------------------------
                                        # COLUMNA DERECHA: CONTENIDO DINÁMICO
                                        # ---------------------------------------------
                                        dbc.Col([
                                            html.Div(id="vertical-content", className="p-3")
                                        ], width=9)

                                    ])
                                ])
                            ], className="shadow-sm border-0", style={"borderRadius": "12px"}),

                        ], fluid=True, className="py-4 px-3")
                    ]),

                    # -------------------------------------------------
                    # PESTAÑA C: Selección del Modelo (Diseño Ficha Técnica)
                    # -------------------------------------------------
                    dbc.Tab(label="c. Selección del Modelo", children=[
                        dbc.CardBody([
                            dbc.Container([

                                # --- Encabezado ---
                                html.H5("Selección de Modelos Predictivos",
                                        className="text-center fw-bold text-primary mb-4"),

                                dbc.Alert([
                                    html.I(className="bi bi-robot me-2"),
                                    "Se seleccionaron 4 algoritmos de clasificación para evaluar el riesgo de readmisión, equilibrando interpretabilidad (médica) y potencia predictiva."
                                ], color="info", className="mb-4"),

                                # --- Tarjeta Principal con Pestañas ---
                                dbc.Card([
                                    dbc.CardHeader(
                                        dbc.Tabs(
                                            id="tabs-modelos",
                                            active_tab="tab-lr",
                                            className="nav-fill",
                                            children=[
                                                dbc.Tab(label="Regresión Logística", tab_id="tab-lr"),
                                                dbc.Tab(label="Árbol de Decisión", tab_id="tab-dt"),
                                                dbc.Tab(label="SVM", tab_id="tab-svm"),
                                                dbc.Tab(label="XGBoost", tab_id="tab-xgb"),
                                            ]
                                        )
                                    ),

                                    dbc.CardBody(id="contenido-modelo")  # Aquí cambia el contenido dinámicamente
                                ], className="shadow-sm border-0")

                            ], fluid=True)
                        ], style={"backgroundColor": "#f8f9fa"})
                    ]),

                    # -------------------------------------------------
# PESTAÑA D: Evaluación del Modelo (Diseño Dashboard)
# -------------------------------------------------
dbc.Tab(label="D. Evaluación del Modelo", children=[
    dbc.CardBody([
        dbc.Container([

            # --- SECCIÓN 1: WORKFLOW DE ENTRENAMIENTO ---
            html.H5("Workflow de Entrenamiento", className="text-primary fw-bold mb-3"),
            
            dbc.Row([
                # Paso 1
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.Div(html.I(className="bi bi-cpu display-6 text-info"), className="mb-2"),
                        html.H6("Preprocesamiento", className="fw-bold"),
                        html.Ul([
                            html.Li("One-Hot Encoding"),
                            html.Li("StandardScaler"),
                            html.Li("Split 80/20")
                        ], className="small text-muted ps-3 mb-0")
                    ])
                ], className="h-100 shadow-sm border-light text-center"), width=12, md=4),

                # Paso 2
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.Div(html.I(className="bi bi-gear-wide-connected display-6 text-primary"), className="mb-2"),
                        html.H6("Entrenamiento", className="fw-bold"),
                        html.P("Ajuste de hiperparámetros (GridSearch) para reducir overfitting y optimizar la generalización.", className="small text-muted mb-0")
                    ])
                ], className="h-100 shadow-sm border-light text-center"), width=12, md=4),

                # Paso 3
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.Div(html.I(className="bi bi-check-circle display-6 text-success"), className="mb-2"),
                        html.H6("Predicción", className="fw-bold"),
                        html.P("Clasificación binaria aplicando un umbral de decisión estándar de 0.5.", className="small text-muted mb-0")
                    ])
                ], className="h-100 shadow-sm border-light text-center"), width=12, md=4),
            ], className="mb-5"),

            # --- SECCIÓN 2: MÉTRICAS DE EVALUACIÓN ---
            html.H5( "Métricas de Desempeño", className="text-primary fw-bold mb-3"),

            dbc.Row([
                # Columna Izquierda: Tarjetas Pequeñas para Accuracy, Precision, F1
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Accuracy (Exactitud)", className="fw-bold text-dark"),
                            html.P("Proporción total de predicciones correctas.", className="small text-muted mb-0")
                        ])
                    ], className="mb-2 shadow-sm border-start border-4 border-info"),
                    
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Precision", className="fw-bold text-dark"),
                            html.P("Calidad de los positivos (reducción de falsas alarmas).", className="small text-muted mb-0")
                        ])
                    ], className="mb-2 shadow-sm border-start border-4 border-warning"),

                    dbc.Card([
                        dbc.CardBody([
                            html.H6("AUC-ROC", className="fw-bold text-dark"),
                            html.P("Capacidad del modelo para distinguir entre clases.", className="small text-muted mb-0")
                        ])
                    ], className="shadow-sm border-start border-4 border-secondary"),
                ], width=12, md=6),

                # Columna Derecha: TARJETA DESTACADA PARA RECALL (La más importante en salud)
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Métrica Crítica: Recall (Sensibilidad)", className="bg-primary text-white fw-bold"),
                        dbc.CardBody([
                            html.P("Capacidad para identificar correctamente todos los casos de riesgo real (Minimizar Falsos Negativos).", className="text-center"),
                            html.Hr(),
                            # Renderizado LaTeX de la fórmula
                            dcc.Markdown(r'''
                            $$Recall = \frac{TP}{TP + FN}$$
                            ''', mathjax=True, className="text-center fs-4"),
                            
                            dbc.Alert([
                                html.I(className="bi bi-exclamation-triangle-fill me-2"),
                                "Prioridad en contextos médicos."
                            ], color="light", className="small text-center mt-3 mb-0")
                        ])
                    ], className="h-100 shadow border-primary")
                ], width=12, md=6),
            ])

        ], fluid=True)
    ], style={"backgroundColor": "#f8f9fa"})
]),

                ]),
            ])
    ])

            



    # --- Btn 7: Resultados/Analisis Final ---
# --- Btn 7: Resultados/Analisis Final ---
    elif boton_id == "btn-7":
        return dbc.Card([
            dbc.CardBody([
                
                # 1. Título Principal de la Sección (Igual que en btn-6)
                html.H3("Resultados y Análisis Final", className="card-title mb-4 text-center fw-bold"),

                # 2. Pestañas estilo Bootstrap (dbc.Tabs)
                dbc.Tabs([
                    
                    # ==========================
                    # PESTAÑA 1: EDA UNIVARIADO
                    # ==========================
                    dbc.Tab(label="EDA - Univariado", children=[
                        dbc.Container([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("Análisis Univariado", className="text-center text-primary fw-bold mb-4"),
                                    
                                    dbc.Row([
                                        # --- NUMÉRICO ---
                                        dbc.Col([
                                            dbc.Card([
                                                dbc.CardHeader(html.H5("Variable Numérica", className="m-0")),
                                                dbc.CardBody([
                                                    html.Label("Selecciona una variable numérica:"),
                                                    dcc.Dropdown(
                                                        id="uni-num-dropdown",
                                                        options=[{"label": c, "value": c} for c in num_cols],
                                                        value=None,
                                                        placeholder="Selecciona...",
                                                        clearable=True
                                                    ),
                                                    dcc.Graph(id="uni-num-box", style={"height": "400px"}),
                                                    html.Div(id="uni-num-table", className="mt-3"),
                                                    html.Div(id="uni-num-interpretacion", className="mt-3 small text-muted")  
                                                ])
                                            ], className="h-100 border-light shadow-sm")
                                        ], md=6),

                                        # --- CATEGÓRICO ---
                                        dbc.Col([
                                            dbc.Card([
                                                dbc.CardHeader(html.H5("Variable Categórica", className="m-0")),
                                                dbc.CardBody([
                                                    html.Label("Selecciona una variable categórica:"),
                                                    dcc.Dropdown(
                                                        id="uni-cat-dropdown",
                                                        options=[{"label": c, "value": c} for c in cat_cols],
                                                        value=None,
                                                        placeholder="Selecciona...",
                                                        clearable=True
                                                    ),
                                                    dcc.Graph(id="uni-cat-bar", style={"height": "400px"}),
                                                    html.Div(id="uni-cat-table", className="mt-3"),
                                                    html.Div(id="uni-cat-interpretacion", className="mt-3 small text-muted")
                                                ])
                                            ], className="h-100 border-light shadow-sm")
                                        ], md=6),
                                    ])
                                ])
                            ], className="shadow-sm border-0", style={"borderRadius": "12px"})
                        ], fluid=True, className="py-4")
                    ]),

                    # ==========================
                    # PESTAÑA 2: EDA BIVARIADO / MULTIVARIADO
                    # ==========================
                    dbc.Tab(label="EDA - Bivariado/Multi", children=[
                        dbc.Container([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("Análisis Bivariado y Multivariado", className="text-center text-primary fw-bold mb-4"),

                                    # --- CATEGÓRICA vs CATEGÓRICA ---
                                    dbc.Card([
                                        dbc.CardHeader("1. Relación Categórica vs Categórica", className="fw-bold bg-light"),
                                        dbc.CardBody([
                                            dbc.Row([
                                                dbc.Col([html.Label("Variable X:"), dcc.Dropdown(id="biv-cat-x", options=[{"label": c, "value": c} for c in cat_cols], value=None, placeholder="Selecciona X...")], md=6),
                                                dbc.Col([html.Label("Variable Y:"), dcc.Dropdown(id="biv-cat-y", options=[{"label": c, "value": c} for c in cat_cols], value=None, placeholder="Selecciona Y...")], md=6),
                                            ]),
                                            dcc.Graph(id="biv-catcat", style={"height": "500px"}),
                                            dbc.Alert("Nota: La mayoría de pacientes no son readmitidos, lo que domina la visualización.", color="info", className="mt-2 small py-2")
                                        ])
                                    ], className="mb-4 border-light shadow-sm"),

                                    # --- NUMÉRICA vs NUMÉRICA ---
                                    dbc.Card([
                                        dbc.CardHeader("2. Relación Numérica vs Numérica", className="fw-bold bg-light"),
                                        dbc.CardBody([
                                            dbc.Row([
                                                dbc.Col([html.Label("Variable X:"), dcc.Dropdown(id="biv-num-x", options=[{"label": c, "value": c} for c in num_cols], value=None, placeholder="Selecciona X...")], md=6),
                                                dbc.Col([html.Label("Variable Y:"), dcc.Dropdown(id="biv-num-y", options=[{"label": c, "value": c} for c in num_cols], value=None, placeholder="Selecciona Y...")], md=6),
                                            ]),
                                            dcc.Graph(id="biv-scatter", style={"height": "400px"}),
                                            dbc.Alert("Se observan correlaciones débiles entre variables numéricas individuales.", color="secondary", className="mt-2 small py-2")
                                        ])
                                    ], className="mb-4 border-light shadow-sm"),

                                    # --- NUMÉRICA vs CATEGÓRICA ---
                                    dbc.Card([
                                        dbc.CardHeader("3. Relación Numérica vs Categórica", className="fw-bold bg-light"),
                                        dbc.CardBody([
                                            dbc.Row([
                                                dbc.Col([html.Label("Variable Numérica (Y):"), dcc.Dropdown(id="biv-num-y2", options=[{"label": c, "value": c} for c in num_cols], value=None)], md=6),
                                                dbc.Col([html.Label("Variable Categórica (X):"), dcc.Dropdown(id="biv-cat-x2", options=[{"label": c, "value": c} for c in cat_cols], value=None)], md=6),
                                            ]),
                                            dcc.Graph(id="biv-boxplot", style={"height": "400px"})
                                        ])
                                    ], className="mb-4 border-light shadow-sm"),

                                    # --- MULTIVARIADO (HEATMAP) ---
                                    dbc.Card([
                                        dbc.CardHeader("4. Matriz de Correlación (Multivariado)", className="fw-bold bg-light"),
                                        dbc.CardBody([
                                            dcc.Graph(id="corr-heatmap", style={"height": "600px"}),
                                            html.P("Correlación más alta: 0.47 (Time in Hospital vs Num Medications).", className="text-muted mt-2")
                                        ])
                                    ], className="border-light shadow-sm"),

                                ])
                            ], className="shadow-sm border-0", style={"borderRadius": "12px"})
                        ], fluid=True, className="py-4")
                    ]),

                    # ==========================
                    # PESTAÑA 3: DASHBOARD
                    # ==========================
                    dbc.Tab(label="Dashboard Interactivo", children=[
                        dbc.Container([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("Dashboard de Gestión de Pacientes", className="text-center text-primary fw-bold mb-4"),
                                    
                                    # Filtros en una "Caja de Control"
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6("Filtros Globales", className="card-subtitle text-muted mb-3"),
                                            dbc.Row([
                                                dbc.Col([html.Label("Grupo de Edad"), dcc.Dropdown(id="filter-age", options=[{"label": "Todos", "value": "Todos"}] + [{"label": v, "value": v} for v in sorted(df['age'].dropna().unique())], value="Todos")], md=2),
                                                dbc.Col([html.Label("Tipo Admisión"), dcc.Dropdown(id="filter-admission", options=[{"label": "Todos", "value": "Todos"}] + [{"label": v, "value": v} for v in sorted(df['admission_type'].dropna().unique())], value="Todos")], md=2),
                                                dbc.Col([html.Label("Insulina"), dcc.Dropdown(id="filter-insulin", options=[{"label": "Todos", "value": "Todos"}] + [{"label": v, "value": v} for v in sorted(df['insulin'].dropna().unique())], value="Todos")], md=2),
                                                dbc.Col([html.Label("Género"), dcc.Dropdown(id="filter-gender", options=[{"label": "Todos", "value": "Todos"}] + [{"label": v, "value": v} for v in sorted(df['gender'].dropna().unique())], value="Todos")], md=2),
                                                dbc.Col([html.Label("Readmisión"), dcc.Dropdown(id="filter-readmit", options=[{"label": "Todos", "value": "Todos"}] + [{"label": v, "value": v} for v in sorted(df['readmitted'].dropna().unique())], value="Todos")], md=2),
                                                dbc.Col([
                                                    html.Label("Días en Hospital"),
                                                    dcc.RangeSlider(
                                                        id="filter-days",
                                                        min=int(df['time_in_hospital'].min()) if 'time_in_hospital' in df.columns else 0,
                                                        max=int(df['time_in_hospital'].max()) if 'time_in_hospital' in df.columns else 30,
                                                        value=[int(df['time_in_hospital'].min()) if 'time_in_hospital' in df.columns else 0, int(df['time_in_hospital'].max()) if 'time_in_hospital' in df.columns else 30],
                                                        tooltip={"placement": "bottom", "always_visible": False}
                                                    )
                                                ], md=2)
                                            ], className="g-2")
                                        ], className="bg-light")
                                    ], className="mb-4 border-0"),

                                    # KPIs
                                    html.Div(id="kpi-cards", className="mb-4"),
                                    
                                    # Gráficos Superiores
                                    dbc.Row([
                                        dbc.Col(dbc.Card(dcc.Graph(id="gauge-readmit"), body=True, className="h-100 shadow-sm"), md=4),
                                        dbc.Col(dbc.Card(dcc.Graph(id="scatter-meds-days"), body=True, className="h-100 shadow-sm"), md=8)
                                    ], className="mb-4"),

                                    # Gráficos Inferiores
                                    dbc.Row([
                                        dbc.Col(dbc.Card(dcc.Graph(id="bar-admission-readmit"), body=True, className="h-100 shadow-sm"), md=6),
                                        dbc.Col(dbc.Card(dcc.Graph(id="heatmap-age-diagnoses"), body=True, className="h-100 shadow-sm"), md=6)
                                    ], className="mb-4"),

                                    html.H5("Tabla Detallada de Pacientes", className="mt-4"),
                                    html.Div(id="filtered-table")
                                ])
                            ], className="shadow-sm border-0", style={"borderRadius": "12px"})
                        ], fluid=True, className="py-4")
                    ]),

                    # ==========================
                    # PESTAÑA 4: VISUALIZACIÓN MODELOS
                    # ==========================
                    dbc.Tab(label="Visualización de Modelos", children=[
                        dbc.Container([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("Resultados de los Modelos Entrenados", className="text-center text-primary fw-bold mb-4"),

                                    # Selector de Modelo centrado
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Card([
                                                dbc.CardHeader("Selecciona un modelo para auditar:", className="bg-white fw-bold"),
                                                dbc.CardBody([
                                                    dcc.Dropdown(
                                                        id="modelo-dropdown",
                                                        options=[
                                                            {"label": "Regresión Logística", "value": "best_lg_pack.pkl"},
                                                            {"label": "Árbol de Decisión", "value": "best_dt_pack.pkl"},
                                                            {"label": "SVM", "value": "best_svm_pack.pkl"},
                                                            {"label": "XGBoost", "value": "best_xgb_pack.pkl"},
                                                        ],
                                                        placeholder="Selecciona archivo .pkl...",
                                                        clearable=True
                                                    )
                                                ])
                                            ], className="shadow-sm border-light")
                                        ], md=6),
                                    ], justify="center", className="mb-4"),

                                    # Tabla de Métricas
                                    html.Div(id="metrics-table", className="mb-4"),

                                    # Gráficas de Evaluación
                                    dbc.Row([
                                        dbc.Col(dbc.Card([
                                            dbc.CardHeader("Matriz de Confusión Normalizada", className="text-center bg-light"),
                                            dbc.CardBody(dcc.Graph(id="ytrue-vs-ypred", style={"height": "500px"}))
                                        ], className="h-100 shadow-sm"), md=6),
                                        
                                        dbc.Col(dbc.Card([
                                            dbc.CardHeader("Análisis de Residuos / Errores", className="text-center bg-light"),
                                            dbc.CardBody(dcc.Graph(id="residuals-plot", style={"height": "500px"}))
                                        ], className="h-100 shadow-sm"), md=6),
                                    ], className="mb-4"),

                                    html.Div(id="modelo-interpretacion", className="mt-4 text-center alert alert-light")
                                ])
                            ], className="shadow-sm border-0", style={"borderRadius": "12px"})
                        ], fluid=True, className="py-4")
                    ]),

                    # ==========================
                    # PESTAÑA 5: INDICADORES
                    # ==========================
                    dbc.Tab(label="Indicadores del Modelo", children=[
                        # Asumiendo que la función retorna un layout, lo envolvemos para mantener consistencia
                        dbc.Container([
                            dbc.Card([
                                dbc.CardBody([
                                    generar_pestaña_indicadores()
                                ])
                            ], className="shadow-sm border-0", style={"borderRadius": "12px"})
                        ], fluid=True, className="py-4")
                    ])

                ]) # Fin de dbc.Tabs
            ]) # Fin de dbc.CardBody principal
        ], className="border-0") # Fin de dbc.Card principal

    # --- Btn 8: Conclusiones / Créditos ---
    elif boton_id == "btn-8":
        return html.Div([
            
            # ---------------------------------------------------------
            # ENCABEZADO: RESUMEN EJECUTIVO
            # ---------------------------------------------------------
            dbc.Card([
                dbc.CardBody([
                    html.H3([html.I(className="bi bi-clipboard-check-fill me-2"), "Conclusiones del Proyecto"], 
                            className="text-center text-primary fw-bold mb-3"),
                    html.P("""
                        Tras procesar más de 100,000 registros clínicos y evaluar cuatro arquitecturas de Machine Learning, 
                        se ha desarrollado una herramienta capaz de identificar patrones de riesgo en la readmisión de pacientes diabéticos. 
                        El análisis confirma que la readmisión no es un evento aleatorio, sino una variable predecible influenciada 
                        fuertemente por el historial de hospitalizaciones previas y la complejidad del diagnóstico.
                    """, className="lead text-center text-muted fs-5")
                ])
            ], className="shadow-sm border-0 mb-5 bg-light"),

            # ---------------------------------------------------------
            # FILA 1: LOS 3 PILARES DEL HALLAZGO (TARJETAS VISUALES)
            # ---------------------------------------------------------
            dbc.Row([
                # Hallazgo 1: El Modelo Ganador
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.Div(html.I(className="bi bi-trophy-fill text-warning display-4"), className="mb-3"),
                        html.H5("Modelo Ganador: XGBoost", className="fw-bold"),
                        html.P("Superó a los demás algoritmos con un ROC-AUC de 0.70 y el mejor Recall (62.9%).", className="text-muted small"),
                        dbc.Badge("Mejor Desempeño", color="warning", text_color="dark", className="mt-2")
                    ], className="text-center")
                ], className="h-100 shadow-sm border-top border-4 border-warning"), width=12, md=4, className="mb-4"),

                # Hallazgo 2: La Métrica Clave
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.Div(html.I(className="bi bi-bullseye text-danger display-4"), className="mb-3"),
                        html.H5("Enfoque en Sensibilidad", className="fw-bold"),
                        html.P("Se priorizó maximizar el Recall para minimizar los Falsos Negativos (pacientes riesgosos no detectados).", className="text-muted small"),
                        dbc.Badge("Prioridad Clínica", color="danger", className="mt-2")
                    ], className="text-center")
                ], className="h-100 shadow-sm border-top border-4 border-danger"), width=12, md=4, className="mb-4"),

                # Hallazgo 3: Calidad de Datos
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.Div(html.I(className="bi bi-database-check text-success display-4"), className="mb-3"),
                        html.H5("Ingeniería de Características", className="fw-bold"),
                        html.P("La agrupación de diagnósticos (ICD-9) y el manejo de medicamentos fueron decisivos para reducir el ruido.", className="text-muted small"),
                        dbc.Badge("Datos Robustos", color="success", className="mt-2")
                    ], className="text-center")
                ], className="h-100 shadow-sm border-top border-4 border-success"), width=12, md=4, className="mb-4"),
            ]),

            # ---------------------------------------------------------
            # FILA 2: IMPACTO Y RECOMENDACIONES (LAYOUT ASIMÉTRICO)
            # ---------------------------------------------------------
            dbc.Row([
                
                # Columna Izquierda: Impacto en Negocio
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Impacto en la Gestión Hospitalaria", className="m-0 fw-bold text-primary"), className="bg-white"),
                        dbc.CardBody([
                            html.Ul([
                                html.Li("Identificación Proactiva: Permite intervenir antes del alta médica en pacientes de alto riesgo.", className="mb-2"),
                                html.Li("Optimización de Recursos: Focaliza los programas de seguimiento ambulatorio en el 11% de la población más crítica.", className="mb-2"),
                                html.Li("Reducción de Costos: Disminuir la tasa de readmisión impacta directamente en la rentabilidad y evita penalizaciones regulatorias.", className="mb-2"),
                            ], className="text-muted")
                        ])
                    ], className="h-100 shadow-sm border-start border-4 border-primary")
                ], width=12, lg=7, className="mb-4"),

                # Columna Derecha: Próximos Pasos
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Próximos Pasos", className="m-0 fw-bold text-dark"), className="bg-light"),
                        dbc.CardBody([
                            html.Div([
                                html.I(className="bi bi-arrow-right-circle-fill text-info me-2"),
                                "Incorporar variables socioeconómicas."
                            ], className="mb-2"),
                            html.Div([
                                html.I(className="bi bi-arrow-right-circle-fill text-info me-2"),
                                "Probar redes neuronales profundas."
                            ], className="mb-2"),
                            html.Div([
                                html.I(className="bi bi-arrow-right-circle-fill text-info me-2"),
                                "Despliegue en tiempo real (API)."
                            ], className="mb-2"),
                        ])
                    ], className="h-100 shadow-sm")
                ], width=12, lg=5, className="mb-4"),
            ]),

            # ---------------------------------------------------------
            # FOOTER: CRÉDITOS Y FUENTE
            # ---------------------------------------------------------
            html.Hr(),
            dbc.Alert([
                html.H6("Fuente de Datos y Créditos", className="alert-heading fw-bold"),
                html.P("Dataset: Diabetes 130-US hospitals for years 1999–2008 (UCI Machine Learning Repository).", className="mb-0 small"),
                html.P("Este dashboard fue desarrollado como herramienta de soporte a la decisión clínica.", className="mb-0 small")
            ], color="secondary", className="small")

        ], className="px-2")

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
        "best_lg_pack.pkl": "La regresión logística muestra un rendimiento moderado en este problema, con métricas alrededor del 0.60. Aunque no es el modelo más fuerte, sí logra capturar una parte importante de los pacientes que realmente regresan al hospital (Recall ≈ 0.594). Esto es relevante porque en la gestión hospitalaria es más grave no identificar a un paciente que sí terminará reingresando, ya que se perdería la oportunidad de intervenir a tiempo. La matriz de confusión confirma esto: aunque el modelo se equivoca en ambos sentidos, logra identificar correctamente a más de la mitad de los pacientes que sí reingresarán (5572 casos), lo cual es útil para priorizar seguimientos o monitoreo adicional. \n Por otro lado, el AUC de 0.602 y la curva ROC muestran que el modelo distingue los casos positivos un poco mejor que el azar, pero todavía queda margen de mejora. Esto implica que, aunque el modelo ofrece señales útiles, su capacidad para separar pacientes de alto y bajo riesgo no es completamente sólida, por lo que decisiones estrictamente basadas en él deberían complementarse con criterio clínico y otras herramientas. En general, este modelo sirve como una base estable y explicativa, pero en la práctica hospitalaria lo ideal sería utilizarlo como apoyo, no como única guía para gestionar el riesgo de reingreso.",
        "best_dt_pack.pkl": "El modelo de Árbol de Decisión ofrece un desempeño moderado para predecir el reingreso de pacientes con diabetes. Aunque clasifica bien a quienes no regresan al hospital, tiene dificultades para identificar a quienes sí lo harán (recall ≈ 0.487), lo que es crítico porque muchos reingresos diabéticos están asociados a complicaciones que podrían prevenirse con seguimiento oportuno. La matriz de confusión muestra que el modelo deja pasar a casi la mitad de los pacientes que realmente reingresan, lo que limita su utilidad clínica. Aun así, el AUC de 0.674 indica que logra distinguir mejor que el azar entre pacientes de alto y bajo riesgo, capturando algunos patrones relevantes, pero no con la fuerza necesaria para usarlo como herramienta principal; funciona mejor como apoyo y no como única guía de decisión.",
        "best_svm_pack.pkl": "El modelo SVM muestra un desempeño moderado para identificar el riesgo de reingreso en pacientes con diabetes. Aunque su accuracy es de 0.623, lo más relevante es que logra un recall de 0.594, es decir, identifica correctamente a más de la mitad de los pacientes que realmente volverán al hospital, algo clave porque muchos reingresos diabéticos pueden prevenirse con seguimiento oportuno. La matriz de confusión confirma esto: el modelo detecta 5572 pacientes que sí reingresan, aunque aún deja pasar a una cantidad importante (3809). Esto indica que el SVM, incluso con el balanceo de clases, todavía tiene dificultades para separar completamente a quienes están en alto riesgo. El AUC de 0.602 muestra que el modelo distingue los casos mejor que el azar, pero sin llegar a ser un predictor fuerte. En general, el SVM aporta señales útiles para priorizar pacientes diabéticos que podrían necesitar más acompañamiento, pero no es suficientemente preciso como para basar decisiones clínicas únicamente en él.",
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
        # TABLA DE MÉTRICAS
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
        # MATRIZ DE CONFUSIÓN
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
        # CURVA ROC
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
        # PARÁMETROS ÓPTIMOS
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
        print("Error al cargar o procesar el modelo:", e)
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
    Input("btn-7", "n_clicks")  
)
def actualizar_comparativa_modelos(_):
    modelos = modelos_pack 

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
            print(f"Error total al abrir {path}: {e2}")
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

    # Aplicar filtros 
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
