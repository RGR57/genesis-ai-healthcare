from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class NLPModule:

    def __init__(self):

        # =========================
        # Intent Training Examples
        # =========================
        self.intent_examples = {

            "clinical_assessment": [
                "I have chest pain and high blood pressure",
                "I smoke and have diabetes",
                "predict heart disease risk",
                "I feel fatigue and dizziness",
                "shortness of breath and obesity",
                "high cardiovascular risk",
                "heart attack symptoms",
                "blood pressure is high"
            ],

            "physiological_analysis": [
                "heart rate is 120",
                "my pulse is low",
                "blood oxygen level",
                "temperature is high",
                "glucose level is 180",
                "my bpm is 45",
                "heart beat is abnormal"
            ],

            "knowledge": [
                "what causes cancer",
                "what causes heart disease",
                "symptoms of hypertension",
                "treatment for diabetes",
                "how to prevent stroke",
                "what is obesity",
                "tell me about cholesterol"
            ],

            "explainability": [
                "explain previous diagnosis",
                "why was this prediction made",
                "show feature importance",
                "explain the reasoning",
                "why did the system predict this"
            ]
        }

        # =========================
        # Build Training Dataset
        # =========================
        self.training_sentences = []
        self.training_labels = []

        for label, examples in self.intent_examples.items():

            for sentence in examples:

                self.training_sentences.append(sentence)
                self.training_labels.append(label)

        # =========================
        # TF-IDF Vectorizer
        # =========================
        self.vectorizer = TfidfVectorizer()

        self.training_vectors = self.vectorizer.fit_transform(
            self.training_sentences
        )

    # =========================
    # Semantic Intent Detection
    # =========================
    def detect_intent(self, query):

        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vector,
            self.training_vectors
        )

        best_index = similarities.argmax()

        confidence = similarities[0][best_index]

        # =========================
        # Low Confidence Fallback
        # =========================
        if confidence < 0.15:
            return "general"

        return self.training_labels[best_index]