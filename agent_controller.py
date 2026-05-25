from logger_module import log_decision
class CoordinatorAgent:
    def __init__(self, model, xai, meta, rag):
        self.model = model
        self.xai = xai
        self.meta = meta
        self.rag = rag

    def process_patient(self, sample):


        # 🔹 Convert sample to query (simple version)
        query = "heart disease risk factors"

        retrieved_knowledge = self.rag.retrieve(query)


        prediction = self.model.predict(sample)
        confidence = self.meta.confidence_score(sample)
        reflection = self.meta.self_reflect(confidence)

        explanation = self.xai.explain(sample.reshape(1, -1))

        return {
            "Prediction": prediction[0],
            "Confidence": confidence,
            "Reflection": reflection,
            "Retrieved Knowledge": retrieved_knowledge,
            "Explanation": explanation
        }