import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# ============================================================
# Configuración
# ============================================================

CARPETA_SALIDAS = "app/static/salidas"


def asegurar_directorio_salida():
    """Garantiza la existencia del directorio de salida para los artefactos generados. (Paradigma: Imperativo)"""
    if not os.path.exists(CARPETA_SALIDAS):
        os.makedirs(CARPETA_SALIDAS)


# ============================================================
# Ingesta y Procesamiento de Datos
# ============================================================


def cargar_datos_desde_lista(archivos: List[Any]) -> pd.DataFrame:
    """
    Carga y consolida datos desde una lista de archivos (UploadFile o rutas locales).
    Maneja la lectura de Excel y concatena los resultados en un único DataFrame.
    (Paradigma: Imperativo)
    """
    dataframes = []
    for archivo in archivos:
        try:
            # Manejo polimórfico: UploadFile de FastAPI vs ruta de archivo string
            fuente = archivo.file if hasattr(archivo, "file") else archivo
            df = pd.read_excel(fuente)
            dataframes.append(df)
        except Exception as e:
            print(f"Error en la lectura del archivo: {e}")
            continue

    if not dataframes:
        raise ValueError(
            "No se pudo cargar información válida de los archivos suministrados."
        )

    return pd.concat(dataframes, ignore_index=True)


def preparar_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza la limpieza inicial y coerción de tipos de datos.
    - Convierte fechas y elimina registros inválidos.
    - Asegura tipos numéricos en columnas críticas (Total, PesoKg), imputando ceros en nulos.
    (Paradigma: Imperativo)
    """
    df = df.copy()

    # Coerción y limpieza de fechas
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.dropna(subset=["Fecha"])

    # Normalización de columnas numéricas
    df["Total"] = pd.to_numeric(df["Total"], errors="coerce").fillna(0)
    df["PesoKg"] = pd.to_numeric(df["PesoKg"], errors="coerce").fillna(0)

    return df


# ============================================================
# Funciones Analíticas (Agregaciones)
# ============================================================


def total_por_servicio(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el total facturado por servicio. (Paradigma: Funcional)"""
    return (
        df.groupby("Servicio")["Total"]
        .sum()
        .reset_index()
        .sort_values("Total", ascending=False)
    )


def total_por_destino(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el total facturado por destino. (Paradigma: Funcional)"""
    return (
        df.groupby("CiudadDestino")["Total"]
        .sum()
        .reset_index()
        .sort_values("Total", ascending=False)
    )


def total_por_vendedor(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el total facturado por vendedor. (Paradigma: Funcional)"""
    return (
        df.groupby("Vendedor")["Total"]
        .sum()
        .reset_index()
        .sort_values("Total", ascending=False)
    )


def peso_total_por_dia(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el peso total acumulado por día. (Paradigma: Funcional)"""
    return df.groupby("Fecha")["PesoKg"].sum().reset_index().sort_values("Fecha")


def kpi_por_servicio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula KPIs transaccionales por servicio. (Paradigma: Funcional)
    - Facturación Total
    - Ticket Promedio
    - Volumen de Envíos
    - Peso Promedio
    """
    return (
        df.groupby("Servicio")
        .agg(
            TotalFacturado=("Total", "sum"),
            TicketPromedio=("Total", "mean"),
            CantidadEnvios=("Total", "count"),
            PesoPromedioKg=("PesoKg", "mean"),
        )
        .reset_index()
        .sort_values("TotalFacturado", ascending=False)
    )


# ============================================================
# Lógica de Negocio y Clasificación
# ============================================================


def clasificar_envio(row) -> str:
    """
    Determina la categoría del envío aplicando reglas de negocio basadas en peso y facturación.
    Retorna: CRÍTICO, GRANDE, PEQUEÑO o NORMAL.
    (Paradigma: Funcional)
    """
    if row["PesoKg"] > 300 and row["Total"] > 1500:
        return "CRÍTICO"
    elif row["PesoKg"] > 200:
        return "GRANDE"
    elif row["Total"] < 200:
        return "PEQUEÑO"
    return "NORMAL"


def aplicar_clasificacion(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la lógica de clasificación a todo el DataFrame. (Paradigma: Funcional)"""
    df = df.copy()
    df["CategoriaEnvio"] = df.apply(clasificar_envio, axis=1)
    return df


# ============================================================
# Análisis Estadístico
# ============================================================


def calcular_estadisticas_numpy(df: pd.DataFrame) -> Dict[str, Any]:
    """Calcula estadísticas descriptivas de alto rendimiento utilizando NumPy. (Paradigma: Funcional)"""
    totales = df["Total"].to_numpy()
    pesos = df["PesoKg"].to_numpy()

    return {
        "media_total": float(np.mean(totales)),
        "desv_total": float(np.std(totales)),
        "peso_max": float(np.max(pesos)),
        "peso_min": float(np.min(pesos)),
        "corr_peso_total": (
            float(np.corrcoef(pesos, totales)[0, 1]) if len(pesos) > 1 else 0.0
        ),
    }


def detectar_outliers(df: pd.DataFrame, stats: Dict[str, Any]) -> pd.DataFrame:
    """
    Detecta anomalías estadísticas (outliers) utilizando el método Z-Score.
    Criterio: Valor absoluto de Z-Score > 3.
    (Paradigma: Funcional)
    """
    df = df.copy()
    media = stats["media_total"]
    desv = stats["desv_total"]

    if desv == 0:
        return df.iloc[0:0]  # Retorna DataFrame vacío si no hay desviación

    z_scores = (df["Total"] - media) / desv
    return df[np.abs(z_scores) > 3]


def analisis_pareto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta análisis de Pareto (Regla 80/20) sobre la cartera de vendedores.
    (Paradigma: Imperativo)
    Clasificación:
    - A: Vendedores que contribuyen al top 80% de la facturación.
    - B: Resto de vendedores (20%).
    """
    df_vendedores = (
        df.groupby("Vendedor")["Total"]
        .sum()
        .reset_index()
        .sort_values("Total", ascending=False)
    )

    total_general = df_vendedores["Total"].sum()
    df_vendedores["Porcentaje"] = df_vendedores["Total"] / total_general
    df_vendedores["Acumulado"] = df_vendedores["Porcentaje"].cumsum()

    df_vendedores["Clasificacion"] = df_vendedores["Acumulado"].apply(
        lambda x: "A (Top 80%)" if x <= 0.80 else "B (Resto 20%)"
    )

    return df_vendedores


# ============================================================
# Proyección (Tendencia Lineal de Holt)
# ============================================================


def proyeccion_ventas(df: pd.DataFrame, dias_proyeccion: int = 7) -> Dict[str, Any]:
    """
    Genera un modelo de pronóstico de ventas utilizando Suavizado Exponencial Doble (Holt's Linear Trend).
    Ideal para series temporales con tendencia pero sin estacionalidad fuerte definida.
    (Paradigma: Imperativo)
    """
    # Agregación diaria
    df_diario = df.groupby("Fecha")["Total"].sum().reset_index().sort_values("Fecha")
    series = df_diario["Total"].values

    if len(series) < 2:
        return None

    # Hiperparámetros de suavizado
    alpha = 0.8  # Factor de suavizado de nivel (sensibilidad a corto plazo)
    beta = 0.2  # Factor de suavizado de tendencia

    # Inicialización del modelo
    nivel = series[0]
    tendencia = series[1] - series[0]

    ajuste_historico = [nivel]

    # Iteración de suavizado (Holt)
    for i in range(1, len(series)):
        valor_actual = series[i]
        nivel_anterior = nivel
        tendencia_anterior = tendencia

        nivel = alpha * valor_actual + (1 - alpha) * (
            nivel_anterior + tendencia_anterior
        )
        tendencia = beta * (nivel - nivel_anterior) + (1 - beta) * tendencia_anterior

        ajuste_historico.append(nivel)

    # Proyección futura
    ultima_fecha = df_diario["Fecha"].iloc[-1]
    fechas_futuras = [
        ultima_fecha + pd.Timedelta(days=i) for i in range(1, dias_proyeccion + 1)
    ]

    # Fórmula: Pronóstico = Nivel + (h * Tendencia)
    valores_proyeccion = [
        max(0, nivel + (i * tendencia)) for i in range(1, dias_proyeccion + 1)
    ]

    return {
        "historico_fechas": df_diario["Fecha"].dt.strftime("%Y-%m-%d").tolist(),
        "historico_valores": df_diario["Total"].tolist(),
        "ajuste_historico": ajuste_historico,
        "proyeccion_fechas": [f.strftime("%Y-%m-%d") for f in fechas_futuras],
        "proyeccion_valores": valores_proyeccion,
        "metodo": "Suavizado Exponencial Doble (Holt)",
    }


# ============================================================
# Visualización y Reportes
# ============================================================


def generar_graficos(
    df: pd.DataFrame,
    df_serv: pd.DataFrame,
    df_dest: pd.DataFrame,
    proyeccion: Dict[str, Any] = None,
):
    """
    Genera y persiste gráficos estáticos en formato PNG utilizando Matplotlib.
    Se utiliza el estilo estándar para compatibilidad con reportes estáticos.
    (Paradigma: Imperativo)
    """
    asegurar_directorio_salida()

    # Resetear configuración de estilos a valores por defecto
    plt.style.use("default")
    plt.rcParams.update(plt.rcParamsDefault)

    def _guardar_plot(nombre_archivo: str, dpi: int = 100):
        """Guarda el gráfico en disco. (Paradigma: Imperativo)"""
        plt.tight_layout()
        plt.savefig(os.path.join(CARPETA_SALIDAS, nombre_archivo), dpi=dpi)
        plt.close()

    # 1. Total por Servicio
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df_serv["Servicio"], df_serv["Total"])
    ax.set_title("Total Facturado por Servicio")
    ax.set_ylabel("Total (S/)")
    plt.xticks(rotation=20, ha="right")
    _guardar_plot("grafico_total_por_servicio.png")

    # 2. Total por Destino
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df_dest["CiudadDestino"], df_dest["Total"])
    ax.set_title("Total Facturado por Destino")
    ax.set_ylabel("Total (S/)")
    plt.xticks(rotation=30, ha="right")
    _guardar_plot("grafico_total_por_destino.png")

    # 3. Participación por Servicio (Pastel)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        df_serv["Total"], labels=df_serv["Servicio"], autopct="%1.1f%%", startangle=140
    )
    ax.set_title("Participación por Servicio")
    _guardar_plot("grafico_pastel_servicio.png")

    # 4. Histograma de Pesos
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df["PesoKg"], bins=20)
    ax.set_title("Distribución de Pesos")
    ax.set_xlabel("Peso (Kg)")
    ax.set_ylabel("Frecuencia")
    _guardar_plot("histograma_peso.png")

    # 5. Proyección de Ventas
    if proyeccion:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            proyeccion["historico_fechas"],
            proyeccion["historico_valores"],
            label="Histórico",
            marker="o",
        )
        ax.plot(
            proyeccion["proyeccion_fechas"],
            proyeccion["proyeccion_valores"],
            label="Proyección (Holt-Winters)",
            marker="o",
            linestyle="--",
        )
        ax.set_title("Proyección de Ventas (Suavizado Exponencial Doble)")
        ax.set_ylabel("Ventas (S/)")
        plt.xticks(rotation=45, ha="right")
        ax.legend()
        _guardar_plot("grafico_proyeccion.png")

    # 6. Evolución de Peso Diario
    df_peso_dia = df.groupby("Fecha")["PesoKg"].sum().reset_index().sort_values("Fecha")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_peso_dia["Fecha"], df_peso_dia["PesoKg"], marker="o")
    ax.set_title("Evolución de Peso Diario")
    ax.set_ylabel("Peso (Kg)")
    plt.xticks(rotation=45, ha="right")
    _guardar_plot("grafico_peso_diario.png")

    # 7. Categorías de Envío
    if "CategoriaEnvio" in df.columns:
        conteo_cat = df["CategoriaEnvio"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            conteo_cat.values,
            labels=conteo_cat.index,
            autopct="%1.1f%%",
            startangle=140,
        )
        ax.set_title("Distribución por Categoría")
        _guardar_plot("grafico_categorias.png")

    # 8. Top Vendedores
    df_top_vendedores = (
        df.groupby("Vendedor")["Total"].sum().sort_values(ascending=False).head(10)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df_top_vendedores.index, df_top_vendedores.values)
    ax.set_title("Top 10 Vendedores")
    ax.set_ylabel("Total (S/)")
    plt.xticks(rotation=45, ha="right")
    _guardar_plot("grafico_top_vendedores.png")


def guardar_reporte_excel(
    data: pd.DataFrame,
    df_servicio: pd.DataFrame,
    df_destino: pd.DataFrame,
    df_vendedor: pd.DataFrame,
    df_kpi: pd.DataFrame,
    df_pareto: pd.DataFrame,
    outliers: pd.DataFrame,
    top5_destinos: pd.DataFrame,
    pivot_servicio_ciudad: pd.DataFrame,
    envios_grandes: pd.DataFrame,
) -> str:
    """
    Genera un reporte consolidado en Excel utilizando XlsxWriter.
    Implementa sanitización de datos y escritura manual de encabezados para prevenir corrupción de archivos
    y garantizar compatibilidad de estilos.
    (Paradigma: Imperativo)
    """
    asegurar_directorio_salida()
    nombre_archivo = "reporte_consolidado.xlsx"
    ruta_completa = os.path.join(CARPETA_SALIDAS, nombre_archivo)

    hojas = {
        "Datos Crudos": data,
        "Por Servicio": df_servicio,
        "Por Destino": df_destino,
        "Por Vendedor": df_vendedor,
        "KPIs": df_kpi,
        "Pareto Vendedores": df_pareto,
        "Outliers": outliers,
        "Top5 Destinos": top5_destinos,
        "Pivot Serv-Ciudad": pivot_servicio_ciudad,
        "Envios >200Kg": envios_grandes,
    }

    with pd.ExcelWriter(ruta_completa, engine="xlsxwriter") as writer:
        workbook = writer.book

        # Definición de Formatos
        # Se escapa el símbolo de moneda para evitar interpretaciones erróneas por Excel
        fmt_moneda = workbook.add_format({"num_format": '"S/" #,##0.00'})
        fmt_encabezado = workbook.add_format(
            {
                "bold": True,
                "bg_color": "#f1f5f9",
                "border": 1,
                "text_wrap": True,
                "valign": "vcenter",
                "align": "center",
            }
        )

        for nombre_hoja, df in hojas.items():
            # Sanitización de Datos
            df_clean = df.copy()

            # Eliminación de información de zona horaria (timezone-naive)
            for col in df_clean.select_dtypes(
                include=["datetime64[ns, UTC]", "datetime64[ns]"]
            ).columns:
                try:
                    df_clean[col] = df_clean[col].dt.tz_localize(None)
                except Exception:
                    pass

            # Manejo de valores infinitos y nulos
            df_clean = df_clean.replace([np.inf, -np.inf], np.nan).fillna(0)

            es_pivot = nombre_hoja == "Pivot Serv-Ciudad"

            # Escritura de Datos (Sin encabezados automáticos)
            df_clean.to_excel(
                writer, sheet_name=nombre_hoja, index=es_pivot, header=False, startrow=1
            )

            worksheet = writer.sheets[nombre_hoja]

            # Escritura Manual de Encabezados
            if es_pivot:
                encabezados = [df_clean.index.name if df_clean.index.name else "Índice"]
                encabezados.extend(df_clean.columns.tolist())
            else:
                encabezados = df_clean.columns.tolist()

            for col_idx, valor in enumerate(encabezados):
                worksheet.write(0, col_idx, str(valor), fmt_encabezado)

            # Ajuste de Anchos de Columna y Formatos
            for i, encabezado in enumerate(encabezados):
                # Cálculo heurístico del ancho de columna
                try:
                    if es_pivot and i == 0:
                        max_len_data = (
                            df_clean.index.astype(str).map(len).max()
                            if not df_clean.empty
                            else 0
                        )
                    else:
                        col_idx = i - 1 if es_pivot else i
                        if 0 <= col_idx < len(df_clean.columns):
                            col_name = df_clean.columns[col_idx]
                            max_len_data = (
                                df_clean[col_name].astype(str).map(len).max()
                                if not df_clean.empty
                                else 0
                            )
                        else:
                            max_len_data = 0

                    max_len = min(max(max_len_data, len(str(encabezado))) + 2, 50)
                except:
                    max_len = 12

                # Aplicación de formato moneda si el encabezado lo sugiere
                if any(x in str(encabezado) for x in ["Total", "Promedio", "Ventas"]):
                    worksheet.set_column(i, i, max_len, fmt_moneda)
                else:
                    worksheet.set_column(i, i, max_len)

    return nombre_archivo


def procesar_archivos(archivos: List[Any]) -> Dict[str, Any]:
    """
    Orquestador principal del flujo de análisis de datos.
    Ejecuta secuencialmente: Carga, Preprocesamiento, Análisis, Proyección, Generación de Activos y Exportación.
    (Paradigma: Imperativo)
    """
    # 1. Carga y Preprocesamiento
    data = cargar_datos_desde_lista(archivos)
    data = preparar_tipos(data)

    # 2. Análisis Agregado
    df_servicio = total_por_servicio(data)
    df_destino = total_por_destino(data)
    df_vendedor = total_por_vendedor(data)
    df_kpi = kpi_por_servicio(data)

    data = aplicar_clasificacion(data)

    stats = calcular_estadisticas_numpy(data)
    outliers = detectar_outliers(data, stats)

    df_pareto = analisis_pareto(data)
    proyeccion = proyeccion_ventas(data)

    # 3. Preparación de Datos para Frontend
    hist_counts, hist_bins = np.histogram(data["PesoKg"], bins=10)
    datos_histograma = {
        "bins": [
            f"{int(hist_bins[i])}-{int(hist_bins[i+1])}"
            for i in range(len(hist_bins) - 1)
        ],
        "counts": hist_counts.tolist(),
    }

    df_peso_dia = peso_total_por_dia(data)
    conteo_categorias = data["CategoriaEnvio"].value_counts()

    # 4. Generación de Activos Estáticos
    generar_graficos(data, df_servicio, df_destino, proyeccion)

    # 5. Preparación y Exportación de Excel
    top5_destinos = df_destino.head(5)
    pivot_servicio_ciudad = pd.pivot_table(
        data,
        values="Total",
        index="CiudadDestino",
        columns="Servicio",
        aggfunc="sum",
        fill_value=0,
    )
    envios_grandes = data[data["PesoKg"] > 200]

    archivo_excel = guardar_reporte_excel(
        data,
        df_servicio,
        df_destino,
        df_vendedor,
        df_kpi,
        df_pareto,
        outliers,
        top5_destinos,
        pivot_servicio_ciudad,
        envios_grandes,
    )

    # 6. Retorno de Resultados Estructurados
    return {
        "kpis": df_kpi.to_dict(orient="records"),
        "estadisticas": stats,
        "top_servicios": df_servicio.head(5).to_dict(orient="records"),
        "top_destinos": df_destino.head(5).to_dict(orient="records"),
        "pareto_vendedores": df_pareto.head(5).to_dict(orient="records"),
        "outliers_count": len(outliers),
        "total_registros": len(data),
        "archivo_reporte": archivo_excel,
        "proyeccion": proyeccion,
        "histograma": datos_histograma,
        "graficos_servicio": {
            "labels": df_servicio["Servicio"].tolist(),
            "data": df_servicio["Total"].tolist(),
        },
        "graficos_destino": {
            "labels": df_destino["CiudadDestino"].tolist(),
            "data": df_destino["Total"].tolist(),
        },
        "grafico_peso_diario": {
            "labels": df_peso_dia["Fecha"].dt.strftime("%Y-%m-%d").tolist(),
            "data": df_peso_dia["PesoKg"].tolist(),
        },
        "grafico_categorias": {
            "labels": conteo_categorias.index.tolist(),
            "data": conteo_categorias.values.tolist(),
        },
        "grafico_vendedores": {
            "labels": df_vendedor["Vendedor"].head(10).tolist(),
            "data": df_vendedor["Total"].head(10).tolist(),
        },
    }
