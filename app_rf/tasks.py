# app_rf/tasks.py
from celery import shared_task
from .utils_run import run_analysis_and_capture
from .mongo_client import db
import gridfs, io, pandas as pd

@shared_task
def tarea_procesar():
    fs = gridfs.GridFS(db)
    f = fs.find_one({"filename":"dataset.csv"})
    df = pd.read_csv(io.BytesIO(f.read()))
    stdout, img_b64 = run_analysis_and_capture(df)
    # guardar resultados en una colección de Mongo a modo de cache
    db.results.insert_one({"stdout": stdout, "img_b64": img_b64})
    return "ok"
