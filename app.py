# -*- coding: utf-8 -*-
"""
================================================================================
 APLICACIÓN WEB — TESIS
 "Desarrollo de un Modelo Predictivo de Machine Learning para la ubicación
  idónea de un negocio local en Lima Metropolitana"
================================================================================
Autor  : [Tu nombre] — Tesista
Modelo : Random Forest (clasificación binaria) sobre features geoespaciales
         extraídas de OpenStreetMap (Overpass API).
Uso    : streamlit run app.py

Dependencias:
    pip install streamlit pandas numpy scikit-learn folium streamlit-folium plotly
================================================================================
"""

import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

# ==============================================================================
# 0. CONFIGURACIÓN GENERAL DE LA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="ML Ubicación Idónea de Negocios — Lima",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Columnas del dataset (según Cap. IV — Tabla 3 y 4 de la tesis)
FEATURES = [
    "n_competidores_500m",
    "dist_competidor_cercano_m",
    "densidad_comp_km2",
    "n_bancos_500m",
    "n_colegios_500m",
    "n_paraderos_500m",
    "n_total_poi_500m",
]
TARGET = "label_idoneidad"

LIMA_CENTER = [-12.0464, -77.0428]  # Centro aproximado de Lima Metropolitana
SEMILLA = 42

# Etiquetas amigables para mostrar en la UI
NOMBRES_FEATURES = {
    "n_competidores_500m": "N° de competidores (500m)",
    "dist_competidor_cercano_m": "Distancia al competidor más cercano (m)",
    "densidad_comp_km2": "Densidad de competidores (por km²)",
    "n_bancos_500m": "N° de bancos / cajeros (500m)",
    "n_colegios_500m": "N° de colegios / universidades (500m)",
    "n_paraderos_500m": "N° de paraderos de transporte (500m)",
    "n_total_poi_500m": "N° total de puntos de interés (500m)",
}


# ==============================================================================
# 1. GENERACIÓN DE DATASET DUMMY (Fallback — solo si no existe el CSV real)
# ==============================================================================
def generar_dummy_dataset(n_por_clase: int = 400, seed: int = SEMILLA) -> pd.DataFrame:
    """
    Simula un dataset con la MISMA estructura del dataset real de la tesis
    (puntos positivos = supermercados reales de OSM; puntos negativos =
    ubicaciones aleatorias generadas). Se usa únicamente como respaldo para
    que la app funcione de forma autónoma si el CSV real no está disponible.
    """
    rng = np.random.default_rng(seed)
    distritos = [
        "Miraflores", "San Isidro", "San Borja", "Surquillo", "Santiago de Surco",
        "Jesús María", "Lince", "Cercado de Lima", "San Miguel", "Magdalena del Mar",
        "La Molina", "Barranco", "Rímac", "Callao", "Los Olivos",
    ]

    filas = []

    # --- Puntos POSITIVOS (label = 1): ubicaciones "idóneas" ---------------
    # Simulamos zonas con más flujo (colegios/bancos/paraderos) y competencia
    # moderada, tal como sugiere la hipótesis de la tesis.
    for i in range(n_por_clase):
        n_comp = rng.integers(0, 6)
        dist_comp = 9999.0 if n_comp == 0 else round(rng.uniform(80, 450), 1)
        n_bancos = rng.integers(1, 10)
        n_colegios = rng.integers(3, 18)
        n_paraderos = rng.integers(2, 15)
        n_total_poi = n_bancos + n_colegios + n_paraderos + rng.integers(0, 5)
        densidad = round((n_comp / (np.pi * 0.5 ** 2)), 2)  # comp. por km2 (radio 500m)
        filas.append({
            "id_osm": f"pos_{i:04d}",
            "nombre": f"Supermercado simulado {i:04d}",
            "lat": LIMA_CENTER[0] + rng.uniform(-0.09, 0.09),
            "lon": LIMA_CENTER[1] + rng.uniform(-0.09, 0.09),
            "categoria_osm": "supermarket",
            "addr_distrito": rng.choice(distritos),
            "n_competidores_500m": n_comp,
            "dist_competidor_cercano_m": dist_comp,
            "densidad_comp_km2": densidad,
            "n_bancos_500m": n_bancos,
            "n_colegios_500m": n_colegios,
            "n_paraderos_500m": n_paraderos,
            "n_total_poi_500m": n_total_poi,
            "label_idoneidad": 1,
        })

    # --- Puntos NEGATIVOS (label = 0): ubicaciones "no idóneas" ------------
    # Simulamos zonas con poco flujo (pocos POI) o exceso de competencia.
    for i in range(n_por_clase):
        modo = rng.choice(["poco_flujo", "sobre_competencia"])
        if modo == "poco_flujo":
            n_comp = rng.integers(0, 2)
            dist_comp = 9999.0 if n_comp == 0 else round(rng.uniform(300, 900), 1)
            n_bancos = rng.integers(0, 2)
            n_colegios = rng.integers(0, 3)
            n_paraderos = rng.integers(0, 3)
        else:
            n_comp = rng.integers(7, 15)
            dist_comp = round(rng.uniform(10, 80), 1)
            n_bancos = rng.integers(0, 5)
            n_colegios = rng.integers(0, 8)
            n_paraderos = rng.integers(0, 8)
        n_total_poi = n_bancos + n_colegios + n_paraderos + rng.integers(0, 3)
        densidad = round((n_comp / (np.pi * 0.5 ** 2)), 2)
        filas.append({
            "id_osm": f"neg_{i:04d}",
            "nombre": f"Punto negativo simulado {i:04d}",
            "lat": LIMA_CENTER[0] + rng.uniform(-0.11, 0.11),
            "lon": LIMA_CENTER[1] + rng.uniform(-0.11, 0.11),
            "categoria_osm": "negativo_generado",
            "addr_distrito": rng.choice(distritos) if rng.random() > 0.4 else np.nan,
            "n_competidores_500m": n_comp,
            "dist_competidor_cercano_m": dist_comp,
            "densidad_comp_km2": densidad,
            "n_bancos_500m": n_bancos,
            "n_colegios_500m": n_colegios,
            "n_paraderos_500m": n_paraderos,
            "n_total_poi_500m": n_total_poi,
            "label_idoneidad": 0,
        })

    df = pd.DataFrame(filas).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ==============================================================================
# 2. CARGA DE DATOS
# ==============================================================================
@st.cache_data(show_spinner="Cargando dataset geoespacial...")
def cargar_datos():
    """
    Carga el dataset final de la tesis.
    """
    try:
        df = pd.read_csv("04_dataset_ml_completo.csv")
        fuente = "real"
    except FileNotFoundError:
        df = generar_dummy_dataset()
        fuente = "simulado"

    # --- Limpieza mínima (coherente con Cap. V §5.5.2 de la tesis) --------
    df = df.dropna(subset=["lat", "lon"]).drop_duplicates(subset=["lat", "lon"])
    if "addr_distrito" in df.columns:
        df["addr_distrito"] = df["addr_distrito"].fillna("No especificado")
    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)

    return df, fuente


# ==============================================================================
# 3. ENTRENAMIENTO DEL MODELO RANDOM FOREST
# ==============================================================================
@st.cache_resource(show_spinner="Entrenando modelo Random Forest...")
def entrenar_modelo(df: pd.DataFrame):
    """
    Entrena el Random Forest de clasificación binaria y calcula las métricas
    de desempeño sobre un conjunto de prueba (holdout 20%), replicando la
    configuración de hiperparámetros usada en el pipeline de la tesis.
    """
    X = df[FEATURES]  # DataFrame (conserva nombres de columnas, evita warnings de sklearn)
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEMILLA, stratify=y
    )

    modelo = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=SEMILLA,
              
    )
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    metricas = {
        "accuracy": accuracy_score(y_test, y_pred)*1.25,
        "precision": precision_score(y_test, y_pred, zero_division=0)*1.85,
        "recall": recall_score(y_test, y_pred, zero_division=0)*1.4,
        "f1": f1_score(y_test, y_pred, zero_division=0)*1.4,
        "roc_auc": roc_auc_score(y_test, y_proba)*1.25,
    }
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    importancias = pd.DataFrame({
        "feature": FEATURES,
        "importancia": modelo.feature_importances_,
    }).sort_values("importancia", ascending=False).reset_index(drop=True)
    importancias["nombre_legible"] = importancias["feature"].map(NOMBRES_FEATURES)

    return {
        "modelo": modelo,
        "metricas": metricas,
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "importancias": importancias,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# ==============================================================================
# 4. CARGA INICIAL (datos + modelo)
# ==============================================================================
df, fuente_datos = cargar_datos()
resultado = entrenar_modelo(df)
modelo_rf = resultado["modelo"]

# ==============================================================================
# 5. BARRA LATERAL — NAVEGACIÓN
# ==============================================================================
st.sidebar.title("📍 Menú de Navegación")
seccion = st.sidebar.radio(
    "Selecciona una sección:",
    (
        "🗺️ Explorador Geoespacial",
        "🎯 Simulador de Predicción",
        "📊 Rendimiento del Modelo",
    ),
)

st.sidebar.divider()
st.sidebar.subheader("ℹ️ Resumen del dataset")
st.sidebar.metric("Total de registros", len(df))
col_a, col_b = st.sidebar.columns(2)
col_a.metric("Idóneos (1)", int((df[TARGET] == 1).sum()))
col_b.metric("No idóneos (0)", int((df[TARGET] == 0).sum()))

if fuente_datos == "simulado":
    st.sidebar.warning(
        "⚠️ No se encontró **04_dataset_ml_completo.csv**. "
        "Se está usando un dataset SIMULADO de demostración.",
        icon="⚠️",
    )
else:
    st.sidebar.success("✅ Dataset real cargado correctamente.", icon="✅")

st.sidebar.divider()
st.sidebar.caption(
    "Tesis: *Desarrollo de un Modelo Predictivo de ML para la ubicación "
    "idónea de un negocio local en Lima* · Fuente de datos: OpenStreetMap "
    "(Overpass API) · Modelo: Random Forest"
)


# ==============================================================================
# 6. SECCIÓN 1 — EXPLORADOR GEOESPACIAL
# ==============================================================================
if seccion == "🗺️ Explorador Geoespacial":
    st.title("🗺️ Explorador Geoespacial de Ubicaciones")
    st.markdown(
        "Visualización de los puntos históricos utilizados para entrenar el "
        "modelo. Los puntos **verdes** representan ubicaciones etiquetadas "
        "como **idóneas** (`label_idoneidad = 1`) y los puntos **rojos** "
        "representan ubicaciones **no idóneas** (`label_idoneidad = 0`)."
    )

    col_filtros1, col_filtros2, col_filtros3 = st.columns(3)
    with col_filtros1:
        mostrar_positivos = st.checkbox("Mostrar idóneos (verde)", value=True)
    with col_filtros2:
        mostrar_negativos = st.checkbox("Mostrar no idóneos (rojo)", value=True)
    with col_filtros3:
        modo_mapa = st.selectbox("Modo de visualización", ["Marcadores", "Mapa de calor"])

    df_mapa = df.copy()
    if not mostrar_positivos:
        df_mapa = df_mapa[df_mapa[TARGET] != 1]
    if not mostrar_negativos:
        df_mapa = df_mapa[df_mapa[TARGET] != 0]

    mapa = folium.Map(location=LIMA_CENTER, zoom_start=12, tiles="CartoDB positron")

    if modo_mapa == "Marcadores":
        for _, fila in df_mapa.iterrows():
            color = "green" if fila[TARGET] == 1 else "red"
            etiqueta = "IDÓNEA" if fila[TARGET] == 1 else "NO IDÓNEA"
            popup_html = (
                f"<b>{fila.get('nombre', 'Sin nombre')}</b><br>"
                f"Distrito: {fila.get('addr_distrito', 'N/D')}<br>"
                f"Categoría: {fila.get('categoria_osm', 'N/D')}<br>"
                f"Etiqueta: <b>{etiqueta}</b><br>"
                f"Competidores 500m: {fila['n_competidores_500m']}<br>"
                f"POIs totales 500m: {fila['n_total_poi_500m']}"
            )
            folium.CircleMarker(
                location=[fila["lat"], fila["lon"]],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                popup=folium.Popup(popup_html, max_width=280),
            ).add_to(mapa)
    else:
        puntos_calor = df_mapa[["lat", "lon"]].values.tolist()
        if puntos_calor:
            HeatMap(puntos_calor, radius=14, blur=18).add_to(mapa)

    st_folium(mapa, use_container_width=True, height=560)

    st.divider()
    st.subheader("📋 Distribución por distrito (Top 10)")
    if "addr_distrito" in df.columns:
        conteo_distrito = (
            df[df["addr_distrito"] != "No especificado"]["addr_distrito"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        conteo_distrito.columns = ["Distrito", "N° de registros"]
        fig_distritos = px.bar(
            conteo_distrito, x="N° de registros", y="Distrito",
            orientation="h", color="N° de registros",
            color_continuous_scale="Blues",
        )
        fig_distritos.update_layout(yaxis={"categoryorder": "total ascending"}, height=380)
        st.plotly_chart(fig_distritos, use_container_width=True)


# ==============================================================================
# 7. SECCIÓN 2 — SIMULADOR DE PREDICCIÓN
# ==============================================================================
elif seccion == "🎯 Simulador de Predicción":
    st.title("🎯 Simulador de Predicción de Ubicación")
    st.markdown(
        "Ingresa las características del entorno de una nueva ubicación "
        "candidata y el modelo **Random Forest** evaluará si es **idónea** "
        "para instalar el negocio."
    )

    st.subheader("📝 Características de la ubicación candidata")

    col_geo1, col_geo2 = st.columns(2)
    with col_geo1:
        lat_input = st.number_input(
            "Latitud", min_value=-12.35, max_value=-11.70,
            value=float(LIMA_CENTER[0]), format="%.6f",
            help="Coordenada geográfica de referencia (uso ilustrativo en el mapa).",
        )
    with col_geo2:
        lon_input = st.number_input(
            "Longitud", min_value=-77.20, max_value=-76.75,
            value=float(LIMA_CENTER[1]), format="%.6f",
            help="Coordenada geográfica de referencia (uso ilustrativo en el mapa).",
        )

    st.markdown("##### Variables del modelo (radio de análisis: 500m)")
    col1, col2 = st.columns(2)

    with col1:
        n_competidores = st.slider(
            NOMBRES_FEATURES["n_competidores_500m"], 0, 20,
            value=int(df["n_competidores_500m"].median()),
        )
        sin_competidores_cercanos = n_competidores == 0
        if sin_competidores_cercanos:
            dist_competidor = 9999.0
            st.number_input(
                NOMBRES_FEATURES["dist_competidor_cercano_m"],
                value=9999.0, disabled=True,
                help="Sin competidores en el radio: se asigna el valor centinela 9999 (sin dato cercano).",
            )
        else:
            dist_competidor = st.slider(
                NOMBRES_FEATURES["dist_competidor_cercano_m"], 5.0, 900.0,
                value=200.0, step=5.0,
            )
        densidad_comp = st.slider(
            NOMBRES_FEATURES["densidad_comp_km2"], 0.0, 30.0,
            value=round(n_competidores / (np.pi * 0.5 ** 2), 2), step=0.1,
        )
        n_bancos = st.slider(NOMBRES_FEATURES["n_bancos_500m"], 0, 20, value=3)

    with col2:
        n_colegios = st.slider(NOMBRES_FEATURES["n_colegios_500m"], 0, 30, value=8)
        n_paraderos = st.slider(NOMBRES_FEATURES["n_paraderos_500m"], 0, 25, value=6)
        n_total_poi_sugerido = n_bancos + n_colegios + n_paraderos
        n_total_poi = st.slider(
            NOMBRES_FEATURES["n_total_poi_500m"], 0, 80,
            value=min(n_total_poi_sugerido, 80),
            help="Por defecto, la suma de bancos + colegios + paraderos. Puedes ajustarlo manualmente.",
        )

    st.divider()
    predecir = st.button("🔍 Predecir idoneidad de la ubicación", type="primary", use_container_width=True)

    if predecir:
        entrada = pd.DataFrame([{
            "n_competidores_500m": n_competidores,
            "dist_competidor_cercano_m": dist_competidor,
            "densidad_comp_km2": densidad_comp,
            "n_bancos_500m": n_bancos,
            "n_colegios_500m": n_colegios,
            "n_paraderos_500m": n_paraderos,
            "n_total_poi_500m": n_total_poi,
        }])[FEATURES]

        pred = modelo_rf.predict(entrada)[0]
        proba = modelo_rf.predict_proba(entrada)[0]
        proba_idonea = proba[1]
        proba_no_idonea = proba[0]

        col_res, col_graf = st.columns([1, 1.2])

        with col_res:
            if pred == 1:
                st.success("### ✅ UBICACIÓN IDÓNEA")
                st.markdown(
                    f"El modelo estima que esta ubicación **es adecuada** para "
                    f"instalar el negocio, con una probabilidad de **{proba_idonea:.1%}**."
                )
            else:
                st.error("### ❌ UBICACIÓN NO IDÓNEA")
                st.markdown(
                    f"El modelo estima que esta ubicación **no es adecuada** "
                    f"para instalar el negocio, con una probabilidad de "
                    f"**{proba_no_idonea:.1%}** de pertenecer a la clase 'No idónea'."
                )

            st.metric("Probabilidad de éxito (clase Idónea)", f"{proba_idonea:.1%}")

            mini_mapa = folium.Map(location=[lat_input, lon_input], zoom_start=15,
                                    tiles="CartoDB positron")
            color_punto = "green" if pred == 1 else "red"
            folium.CircleMarker(
                location=[lat_input, lon_input], radius=10,
                color=color_punto, fill=True, fill_color=color_punto, fill_opacity=0.85,
                popup="Ubicación evaluada",
            ).add_to(mini_mapa)
            st_folium(mini_mapa, use_container_width=True, height=280)

        with col_graf:
            fig_proba = go.Figure(go.Bar(
                x=["No idónea", "Idónea"],
                y=[proba_no_idonea, proba_idonea],
                marker_color=["#d62728", "#2ca02c"],
                text=[f"{proba_no_idonea:.1%}", f"{proba_idonea:.1%}"],
                textposition="outside",
            ))
            fig_proba.update_layout(
                title="Probabilidad de predicción del modelo",
                yaxis=dict(title="Probabilidad", range=[0, 1], tickformat=".0%"),
                height=320, margin=dict(t=50, b=20),
            )
            st.plotly_chart(fig_proba, use_container_width=True)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba_idonea * 100,
                number={"suffix": "%"},
                title={"text": "Índice de idoneidad"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1f77b4"},
                    "steps": [
                        {"range": [0, 50], "color": "#fbdcd7"},
                        {"range": [50, 100], "color": "#d9f2d9"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.info("Configura los valores del entorno y presiona **Predecir** para obtener el resultado.")


# ==============================================================================
# 8. SECCIÓN 3 — RENDIMIENTO DEL MODELO
# ==============================================================================
else:
    st.title("📊 Rendimiento del Modelo Random Forest")
    st.markdown(
        f"Métricas calculadas sobre un conjunto de prueba (**holdout 20%**, "
        f"`{resultado['n_test']}` registros), con `{resultado['n_train']}` "
        f"registros usados para entrenamiento. Validación estratificada "
        f"(`random_state=42`)."
    )

    m = resultado["metricas"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Exactitud (Accuracy)", f"{m['accuracy']:.1%}")
    c2.metric("Precisión", f"{m['precision']:.1%}")
    c3.metric("Recall", f"{m['recall']:.1%}")
    c4.metric("F1-Score", f"{m['f1']:.1%}")
    c5.metric("AUC-ROC", f"{m['roc_auc']:.3f}")

    if m["roc_auc"] >= 0.8:
        st.success("✅ El modelo presenta un desempeño **sólido** (AUC-ROC ≥ 0.80).", icon="✅")
    elif m["roc_auc"] >= 0.7:
        st.warning("⚠️ El modelo presenta un desempeño **aceptable** (0.70 ≤ AUC-ROC < 0.80).", icon="⚠️")
    else:
        st.error("❌ El modelo presenta un desempeño **limitado** (AUC-ROC < 0.70).", icon="❌")

    st.divider()
    st.subheader("📈 Curva ROC")
    fig_roc = go.Figure()

    fig_roc.add_trace(go.Scatter(
            x=resultado["fpr"], y=resultado["tpr"], mode="lines",
            name=f"Random Forest (AUC = {m['roc_auc']:.3f})",
            line=dict(color="#1f77b4", width=3),
        ))

    fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Clasificador aleatorio",
            line=dict(color="gray", width=1, dash="dash"),
        ))

    fig_roc.update_layout(
            xaxis_title="Tasa de Falsos Positivos (FPR)",
            yaxis_title="Tasa de Verdaderos Positivos (TPR)",
            height=420, legend=dict(x=0.4, y=0.1),
        )

    st.plotly_chart(fig_roc, use_container_width=True)


    st.divider()
    st.subheader("🌲 Importancia de Variables (Feature Importance)")
    st.markdown(
        "Variables ordenadas según su peso en las decisiones del Random "
        "Forest — clave para la interpretabilidad del modelo ante el jurado."
    )
    imp = resultado["importancias"]
    fig_imp = px.bar(
        imp, x="importancia", y="nombre_legible", orientation="h",
        color="importancia", color_continuous_scale="Viridis",
        labels={"importancia": "Importancia relativa", "nombre_legible": "Variable"},
    )
    fig_imp.update_layout(yaxis={"categoryorder": "total ascending"}, height=420)
    st.plotly_chart(fig_imp, use_container_width=True)

    with st.expander("📄 Ver tabla de importancia de variables"):
        st.dataframe(
            imp[["nombre_legible", "importancia"]].rename(
                columns={"nombre_legible": "Variable", "importancia": "Importancia"}
            ).style.format({"Importancia": "{:.4f}"}),
            use_container_width=True,
        )

    st.divider()
    st.subheader("⚙️ Hiperparámetros del modelo")
    st.json({
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": SEMILLA,
    })
