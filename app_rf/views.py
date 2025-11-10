# app_rf/views.py (ejemplo de lectura)
from django.http import HttpResponse
from .mongo_client import db
import gridfs
import pandas as pd
import io

def ver_dataset(request):
    fs = gridfs.GridFS(db)
    file = fs.find_one({"filename":"dataset.csv"})
    data_bytes = file.read()
    df = pd.read_csv(io.BytesIO(data_bytes))
    # ahora usa df como en tus notebooks
    return HttpResponse(f"Filas: {len(df)}")
