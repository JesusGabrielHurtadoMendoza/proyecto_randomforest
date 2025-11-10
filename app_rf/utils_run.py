# app_rf/utils_run.py
import matplotlib.pyplot as plt
import io, base64, sys
import pandas as pd
from contextlib import redirect_stdout

def run_analysis_and_capture(df):
    buf = io.StringIO()
    with redirect_stdout(buf):
        # Aquí pones tu código que imprimía en el notebook
        print("Iniciando análisis...")
        print("Shape:", df.shape)
        # ejemplo gráfico
        fig, ax = plt.subplots()
        df.iloc[:100].hist(ax=ax)  # ejemplo
        # Guardar figura a bytes
        img_buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(img_buf, format='png')
        plt.close(fig)
        img_buf.seek(0)
        img_b64 = base64.b64encode(img_buf.read()).decode('utf-8')
    stdout_text = buf.getvalue()
    return stdout_text, img_b64
