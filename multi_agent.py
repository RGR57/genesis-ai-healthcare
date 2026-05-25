

# WHAT TO CHANGE

## 1. Replace `multi_agent.py`


from nlp_module import NLPModule
from llm_module import LLMModule


class CoordinatorAgent:

    def __init__(self, model, xai, meta, rag):

        self.model = model
        self.xai = xai
        self.meta = meta
        self.rag = rag

        self.nlp = NLPModule()
        self.llm = LLMModule()

        self.last_sample = None
        self.last_result = None

    # =========================
    # Prediction Pipeline
    # =========================
    def process_patient(self, sample):

        prediction = self.model.predict(sample)

        confidence = self.meta.confidence_score(sample)

        confidence = min(confidence, 0.94)

        reflection = self.meta.self_reflect(confidence)

        explanation = self.xai.explain(sample)

        result = {
            "Prediction": int(prediction[0]),
            "Confidence": float(confidence),
            "Reflection": reflection,
            "Explanation": explanation
        }

        self.last_sample = sample
        self.last_result = result

        return result

    # =========================
    # Main Cognitive Pipeline
    # =========================
    def process_query(self, query, feature_names):

        intent = self.nlp.detect_intent(query)

        # =========================
        # Clinical Assessment
        # =========================
        if intent in [
            "clinical_assessment",
            "physiological_analysis"
        ]:

            values = self.llm.extract_medical_features(
                query,
                feature_names
            )

            prediction_result = self.process_patient(values)

            diagnosis = (
                "⚠ High Cardiovascular Risk"
                if prediction_result["Prediction"] == 1
                else "✅ Low Cardiovascular Risk"
            )

            return {
                "Answer": f"""
========================================
GENESIS AI CLINICAL ASSESSMENT
========================================

Diagnosis:
{diagnosis}

Confidence Score:
{prediction_result['Confidence']:.2f}

Meta-Cognitive Reflection:
{prediction_result['Reflection']}

Explainable AI:
Clinical feature contribution analysis completed.

Recommendation:
Further medical evaluation advised.
========================================
"""
            }

        # =========================
        # Explainability
        # =========================
        elif intent == "explainability":

            if self.last_sample is None:

                return {
                    "Answer": "No previous diagnosis available for explanation."
                }

            self.xai.explain(self.last_sample)

            return {
                "Answer": f"""
========================================
GENESIS AI EXPLAINABILITY REPORT
========================================

The system analyzed:

- Clinical indicators
- Physiological variables
- Risk contribution patterns

Explainable AI reasoning generated successfully.
========================================
"""
            }

        # =========================
        # Knowledge Retrieval
        # =========================
        elif intent == "knowledge":

            rag_answer = self.rag.retrieve(query)

            if rag_answer != "No relevant information found.":

                return {
                    "Answer": f"""
========================================
GENESIS AI KNOWLEDGE RETRIEVAL
========================================

{rag_answer}
========================================
"""
                }

            llm_answer = self.llm.ask(query)

            return {
                "Answer": llm_answer
            }

        # =========================
        # General Conversation
        # =========================
        else:

            return {
                "Answer": """
========================================
GENESIS AI HEALTHCARE ASSISTANT
========================================

Capabilities:
- Clinical risk prediction
- Explainable AI reasoning
- Meta-cognitive reflection
- Medical knowledge retrieval
- Healthcare analytics

Example Queries:
- I am 55 years old with chest pain and BP 160
- I smoke and have diabetes
- what causes heart disease
- explain previous diagnosis
========================================
"""
            }

