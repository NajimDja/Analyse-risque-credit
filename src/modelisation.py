# Modèles de prédiction
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

class Models:
    
    def regression_logistique(self, 
                              penalty : str = "l2", 
                              max_iter : int = 100, 
                              tol : float = 0.0001, 
                              solver : str = "lbfgs"):

        log_reg = LogisticRegression(
            penalty=penalty,
            tol = tol,
            random_state=100,
            solver=solver,
            max_iter=max_iter,
            verbose=1
        )

        return log_reg


class Metrics:
    
    def evaluate_model(self, y_true, y_pred, y_prob, model_name):

        print(f"\n{'='*50}")
        print(f"MODEL : {model_name}")
        print(f"{'='*50}")

        # Metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_prob)

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")

        print("\nClassification Report:\n")
        print(classification_report(y_true, y_pred))

        # Figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[0]
        )

        axes[0].set_title("Confusion Matrix")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)

        axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        axes[1].plot([0, 1], [0, 1], linestyle="--")

        axes[1].set_title("ROC Curve")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].legend()

        # 3. Probability Distribution
        axes[2].hist(
            y_prob[y_true == 0],
            bins=30,
            alpha=0.6,
            label="No Default"
        )

        axes[2].hist(
            y_prob[y_true == 1],
            bins=30,
            alpha=0.6,
            label="Default"
        )

        axes[2].set_title("Predicted Probabilities")
        axes[2].set_xlabel("Probability")
        axes[2].set_ylabel("Frequency")
        axes[2].legend()

        plt.tight_layout()
        plt.show()