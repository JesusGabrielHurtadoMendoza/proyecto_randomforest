# app_rf/mongo_client.py
from pymongo import MongoClient
import os

client = MongoClient(os.environ.get("mongodb+srv://jesusgabo2002_db_user:<db_password>@cluster0.l7t6q6j.mongodb.net/"))
db = client.get_default_database()  # o client['nombre_db']
