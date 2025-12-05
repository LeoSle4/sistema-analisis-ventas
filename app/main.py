from fastapi import FastAPI, Request, UploadFile, File, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List
from app.logica import procesar_archivos
import os
import traceback

app = FastAPI(title="Sistema de Análisis de Transportes")

# Montar archivos estáticos (para servir las imágenes generadas)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Configurar plantillas
templates = Jinja2Templates(directory="app/templates")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=b"", media_type="image/x-icon")


@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
async def chrome_devtools():
    return JSONResponse(content={})


@app.get("/", response_class=HTMLResponse)
async def inicio(request: Request):
    """
    Renderiza la página de inicio.
    (Paradigma: Imperativo)
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analizar", response_class=HTMLResponse)
async def analizar(request: Request, archivos: List[UploadFile] = File(...)):
    """
    Recibe los archivos, ejecuta el análisis y muestra los resultados.
    (Paradigma: Imperativo)
    """
    try:
        # Ejecutar la lógica de negocio
        resultados = procesar_archivos(archivos)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "resultados": resultados,
                "mensaje": "✅ Análisis completado con éxito.",
            },
        )
    except Exception as e:
        traceback.print_exc()
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"❌ Ocurrió un error: {str(e)}"},
        )
