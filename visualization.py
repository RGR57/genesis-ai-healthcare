import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

def plot_feature_importance(model, feature_names):
    importances = model.model.feature_importances_

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances)
    plt.xlabel("Importance")
    plt.title("Feature Importance (Global Explanation)")
    plt.show()


def plot_prediction_probabilities(model, sample):
    probs = model.model.predict_proba([sample])[0]

    plt.figure()
    plt.bar(["No Disease", "Disease"], probs)
    plt.title("Prediction Probabilities")
    plt.ylabel("Probability")
    plt.show()


def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i][j], ha="center", va="center")

    plt.colorbar()
    plt.show()


def plot_roc_curve(model, X_test, y_test):
    y_probs = model.model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1])

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


def plot_evolution_progress(fitness_history):
    plt.figure()
    plt.plot(fitness_history)

    plt.title("Evolutionary Optimization Progress")
    plt.xlabel("Generation")
    plt.ylabel("Best Accuracy")

    plt.show()


def plot_confidence_distribution(model, X_test):
    confidences = []

    for sample in X_test[:50]:
        prob = np.max(model.model.predict_proba([sample]))
        confidences.append(prob)

    plt.figure()
    plt.hist(confidences)

    plt.title("Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")

    plt.show()