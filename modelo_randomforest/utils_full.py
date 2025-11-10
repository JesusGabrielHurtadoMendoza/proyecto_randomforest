from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

def generar_curvas_roc_pr(y_test, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

    # Crear las gráficas
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (área = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('Tasa de falsos positivos')
    ax_roc.set_ylabel('Tasa de verdaderos positivos')
    ax_roc.set_title('Curva ROC')
    ax_roc.legend(loc="lower right")

    fig_pr, ax_pr = plt.subplots()
    ax_pr.plot(recall, precision, color='purple', lw=2)
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Curva Precision-Recall')

    return {'roc': ax_roc, 'pr': ax_pr}
