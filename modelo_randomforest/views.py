import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import requests
import os
import pandas as pd

def index(request):
    # URL de descarga directa
    url = "https://drive.google.com/uc?export=download&id=1lfOIXsQfaRNyB-VLiGbTtX5qq5DkDHoQ"
    local_filename = os.path.join(os.path.dirname(__file__), 'TotalFeatures-ISCXFlowMeter.csv')

    # Si el archivo no existe localmente, lo descarga
    if not os.path.exists(local_filename):
        r = requests.get(url, stream=True)
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    # Ahora carga con pandas
    df = pd.read_csv(local_filename)
    # … resto de tu código …

    # Mostrar información general del dataset
    dataset_info = {
        'filas': df.shape[0],
        'columnas': df.shape[1],
        'columnas_nombres': list(df.columns)
    }

    # ============================
    # 2️⃣ LIMPIEZA DE DATOS (fix NaN)
    # ============================
    df = df.dropna(subset=['calss'])
    df = df.fillna(0)

    # ============================
    # 3️⃣ SEPARAR VARIABLES
    # ============================
    X = df.drop('calss', axis=1)
    y = df['calss']

    # Codificar etiquetas si son texto
    y = pd.factorize(y)[0]

    # ============================
    # 4️⃣ DIVISIÓN TRAIN / TEST
    # ============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # ============================
    # 5️⃣ ENTRENAR EL MODELO
    # ============================
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ============================
    # 6️⃣ MÉTRICAS
    # ============================
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted')
    }

    # ============================
    # 7️⃣ REPORTE DE CLASIFICACIÓN
    # ============================
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # ============================
    # 8️⃣ MATRIZ DE CONFUSIÓN
    # ============================
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicciones')
    plt.ylabel('Reales')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    conf_matrix_img = base64.b64encode(buf.getvalue()).decode('utf-8')

    # ============================
    # 9️⃣ IMPORTANCIA DE CARACTERÍSTICAS
    # ============================
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(20)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importances)
    plt.title('Importancia de Características')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    feature_importance_img = base64.b64encode(buf.getvalue()).decode('utf-8')

    # ============================
    # 🔟 DISTRIBUCIÓN DE CLASES
    # ============================
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y)
    plt.title('Distribución de Clases')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    class_distribution_img = base64.b64encode(buf.getvalue()).decode('utf-8')

    # ============================
    # 11️⃣ MATRIZ DE CORRELACIÓN
    # ============================
    corr = df.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm')
    plt.title('Matriz de Correlación')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    correlation_img = base64.b64encode(buf.getvalue()).decode('utf-8')

    # ============================
    # 12️⃣ ESTADÍSTICAS DESCRIPTIVAS
    # ============================
    stats = df.describe().to_html(classes='table table-bordered')

    # ============================
    # 13️⃣ 10 PRIMERAS FILAS
    # ============================
    primeras_filas = df.head(10).to_html(classes='table table-striped')

    # ============================
    # 14️⃣ CONTEXTO FINAL
    # ============================
    context = {
        'dataset_info': dataset_info,
        'primeras_filas': primeras_filas,
        'stats': stats,
        'metrics': metrics,
        'classification_report': report_df.to_html(classes='table table-striped'),
        'conf_matrix_img': conf_matrix_img,
        'feature_importance_img': feature_importance_img,
        'class_distribution_img': class_distribution_img,
        'correlation_img': correlation_img,
    }

    return render(request, 'index.html', context)
