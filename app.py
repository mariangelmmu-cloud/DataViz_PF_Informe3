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


# -------------------------------------
# Botonera horizontal (idéntica a codigo_juan.txt)
# -------------------------------------
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
                                dcc.Graph(id="uni-num-box", style={"height": "320px"}),
                                html.Div(id="uni-num-table", className="mt-3")
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
                                dcc.Graph(id="uni-cat-bar", style={"height": "320px"}),
                                html.Div(id="uni-cat-table", className="mt-3")
                            ])
                        ])
                    ], md=6),
                ])
            ]),

            # ==========================
            # EDA: BIVARIADO
            # ==========================
            dcc.Tab(label="EDA - Bivariado", children=[
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
                        dcc.Graph(id="biv-catcat", style={"height": "550px", "width": "95%"})  # gráfico más grande
                    ])
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
                        dcc.Graph(id="biv-scatter", style={"height": "350px"})
                    ])
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
                        dcc.Graph(id="biv-boxplot", style={"height": "350px"})
                    ])
                ], className="mb-4"),
            ]),

            # ==========================
            # EDA: MULTIVARIADO
            # ==========================
            dcc.Tab(label="EDA - Multivariado", children=[
                html.Br(),
                html.H3("Análisis Multivariado"),
                html.P("Mapa de calor de correlaciones (variables numéricas)"),
                dcc.Graph(id="corr-heatmap", style={"height": "320px"})
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
            # EVALUACIÓN DE MODELOS
            # ==========================
            dcc.Tab(label="Evaluación de Modelos", children=[
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

                # --- Gráfica: valores reales vs. predichos ---
                dbc.Row([
                    dbc.Col(dcc.Graph(id="ytrue-vs-ypred", style={"height": "350px"}), md=6),
                    dbc.Col(dcc.Graph(id="residuals-plot", style={"height": "350px"}), md=6),
                ])
            ]),

                    # ==========================
        # INDICADORES DE EVALUACIÓN
        # ==========================
        dcc.Tab(label="Indicadores del Modelo", children=[
            html.Br(),
            html.H3("Indicadores de Evaluación del Modelo", className="text-center mb-4"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Selecciona un modelo entrenado")),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id="indicadores-dropdown",
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
            dbc.Row([
                dbc.Col(html.Div(id="indicadores-table"), md=6),
                dbc.Col(html.Div(id="indicadores-interpretacion"), md=6)
            ])
        ]),




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

    stats_df = df[var_num].describe().to_frame().T.round(2)
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
    Input("uni-num-dropdown", "value")  # trigger when selection changes (or any other)
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
# Callbacks - Evaluación de Modelos (Clasificación)
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
            fig_cm.update_layout(template="plotly_white", height=350)

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
                template="plotly_white", height=350
            )

        # ===========================
        # 4️⃣ PARÁMETROS ÓPTIMOS
        # ===========================
        params_df = pd.DataFrame(best_params.items(), columns=["Parámetro", "Valor"])
        if best_score_cv is not None:
            params_df.loc[len(params_df)] = ["Mejor Puntuación CV", round(best_score_cv, 3)]
        params_table = dbc.Table.from_dataframe(params_df, striped=True, bordered=True, hover=True)

        return (
            html.Div([
                html.H5("Métricas de Evaluación:"),
                metricas_table,
                html.Br(),
                html.H5("⚙️ Parámetros Óptimos del Modelo:"),
                params_table
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
    Output("indicadores-table", "children"),
    Output("indicadores-interpretacion", "children"),
    Input("indicadores-dropdown", "value")
)
def indicadores_clasificacion(pkl_file):
    if not pkl_file:
        return html.Div(), html.Div()

    try:
        data = safe_joblib_load_full(pkl_file)
        if not data:
            raise ValueError("Archivo .pkl ilegible o vacío")

        y_test = pd.Series(data.get("y_test", []))
        y_pred = pd.Series(data.get("y_pred", []))
        y_pred_proba = data.get("y_pred_proba", None)

        if y_pred_proba is None:
            return html.Div([html.P("El modelo no tiene probabilidades predichas (`y_pred_proba`).")]), html.Div()

        # --- Calcular métricas probabilísticas ---
        indicadores = {
            "Log Loss": log_loss(y_test, y_pred_proba),
            "Brier Score": brier_score_loss(y_test, y_pred_proba),
            "ROC-AUC": roc_auc_score(y_test, y_pred_proba),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "Accuracy": accuracy_score(y_test, y_pred),
        }

        df_ind = pd.DataFrame(indicadores.items(), columns=["Indicador", "Valor"]).round(4)
        tabla = dbc.Table.from_dataframe(df_ind, striped=True, bordered=True, hover=True)

        interpretacion = html.Div([
            html.H5("Interpretación de los indicadores:"),
            html.P("""
                • Un menor Log Loss y Brier Score indican un modelo más preciso en sus probabilidades.  
                • Un ROC-AUC más alto muestra mejor capacidad de discriminación entre clases.  
                • F1-Score y Accuracy miden el desempeño general: valores cercanos a 1 son mejores.  
                • En modelos desbalanceados, prioriza el F1-Score sobre el Accuracy.
            """)
        ])

        return html.Div([html.H5("Indicadores de Evaluación del Modelo:"), tabla]), interpretacion

    except Exception as e:
        print("❌ Error cargando modelo:", e)
        return html.Div([html.P(f"Error procesando {pkl_file}: {e}")]), html.Div()


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
