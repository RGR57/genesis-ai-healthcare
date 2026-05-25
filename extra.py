import shap
import matplotlib.pyplot as plt
import numpy as np

class ExplainableAI:
    def __init__(self, model, feature_names):
        self.model = model.model
        self.explainer = shap.TreeExplainer(self.model)
        self.feature_names = feature_names

    def explain(self, X_sample):

        if not isinstance(X_sample, np.ndarray):
            X_sample = np.array(X_sample)

        if len(X_sample.shape) == 1:
            X_sample = X_sample.reshape(1, -1)

        shap_values = self.explainer(X_sample)

        explanation = shap.Explanation(
            values=shap_values.values[0, :, 1],
            base_values=shap_values.base_values[0, 1],
            data=X_sample[0],
            feature_names=self.feature_names
        )

        plt.figure(figsize=(12, 6))
        shap.plots.waterfall(explanation)
        plt.tight_layout()
        plt.show()


        return shap_values.values