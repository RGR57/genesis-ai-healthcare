import numpy as np

class MetaCognition:
    def __init__(self, model, threshold=0.75):
        self.model = model
        self.threshold = threshold

    def confidence_score(self, sample):
        probabilities = self.model.model.predict_proba([sample])
        return np.max(probabilities)

    def self_reflect(self, confidence):
        if confidence < self.threshold:
            return "Low confidence → Escalate to human expert"
        else:
            return "High confidence → Autonomous decision"