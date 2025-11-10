# scripts/upload_to_gridfs.py
from pymongo import MongoClient
import gridfs
import os

uri = os.environ.get("mongodb+srv://jesusgabo2002_db_user:<db_password>@cluster0.l7t6q6j.mongodb.net/")
client = MongoClient(uri)
db = client.get_default_database()
fs = gridfs.GridFS(db)

file_path = "TotalFeatures-ISCXFlowMeter.csv"   # cambia nombre
with open(file_path, "rb") as f:
    file_id = fs.put(f, filename=os.path.basename(file_path))
    print("Uploaded file_id:", file_id)
