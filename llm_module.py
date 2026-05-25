import re
import requests


class LLMModule:

    # =========================
    # Dynamic Medical Reasoning
    # =========================
    def ask(self, query):

        try:

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5-coder:1.5b",
                    "prompt": f"""
You are Genesis AI, an advanced healthcare cognitive assistant.

Provide medically informative, concise, and professional responses.

User Query:
{query}
""",
                    "stream": False
                },
                timeout=120
            )

            result = response.json()

            if "response" in result:

                return result["response"]

            elif "message" in result:

                return result["message"]

            else:

                return f"""
Genesis AI received an unexpected LLM response.

Raw Output:
{result}
"""

        except Exception as e:

            return f"""
Genesis AI LLM connection failed.

Error:
{e}

Please ensure:
- Ollama is running
- qwen2.5-coder:1.5b model is installed
- Ollama server is active
"""

    # =========================
    # Dynamic Feature Extraction
    # =========================
    def extract_medical_features(self, query, feature_names):

        query = query.lower()

        values = []

        # =========================
        # Numerical Extraction
        # =========================
        age = 0
        bp = 0
        chol = 0
        glucose = 0
        heart_rate = 0

        age_match = re.search(
            r'(\d+)\s*years?',
            query
        )

        bp_match = re.search(
            r'(bp|blood pressure)\s*(\d+)',
            query
        )

        chol_match = re.search(
            r'cholesterol\s*(\d+)',
            query
        )

        glucose_match = re.search(
            r'(glucose|sugar)\s*(\d+)',
            query
        )

        hr_match = re.search(
            r'(heart rate|pulse|bpm)\s*(is)?\s*(\d+)',
            query
        )

        # =========================
        # Assign Numerical Values
        # =========================
        if age_match:
            age = float(age_match.group(1))

        if bp_match:
            bp = float(bp_match.group(2))

        if chol_match:
            chol = float(chol_match.group(1))

        if glucose_match:
            glucose = float(glucose_match.group(2))

        if hr_match:
            heart_rate = float(hr_match.group(3))

        # =========================
        # Semantic Clinical States
        # =========================
        smoking = (
            "smoke" in query or
            "smoking" in query
        )

        diabetes = (
            "diabetes" in query or
            "diabetic" in query
        )

        obesity = (
            "obesity" in query or
            "obese" in query
        )

        chest_pain = (
            "chest pain" in query
        )

        fatigue = (
            "fatigue" in query
        )

        hypertension = (
            bp >= 140 or
            "hypertension" in query
        )

        male = "male" in query

        # =========================
        # Dynamic Feature Mapping
        # =========================
        for feature in feature_names:

            feature_lower = feature.lower()

            val = 0

            # AGE
            if "age" in feature_lower:
                val = age

            # BLOOD PRESSURE
            elif (
                "blood pressure" in feature_lower or
                "bp" in feature_lower
            ):
                val = bp

            # CHOLESTEROL
            elif "cholesterol" in feature_lower:
                val = chol

            # GLUCOSE
            elif (
                "glucose" in feature_lower or
                "sugar" in feature_lower
            ):
                val = glucose

            # HEART RATE
            elif (
                "heart rate" in feature_lower or
                "pulse" in feature_lower
            ):
                val = heart_rate

            # SMOKING
            elif "smoking" in feature_lower:
                val = 1 if smoking else 0

            # DIABETES
            elif "diabetes" in feature_lower:
                val = 1 if diabetes else 0

            # OBESITY
            elif "obesity" in feature_lower:
                val = 1 if obesity else 0

            # CHEST PAIN
            elif (
                "chest pain" in feature_lower or
                "angina" in feature_lower
            ):
                val = 1 if chest_pain else 0

            # FATIGUE
            elif "fatigue" in feature_lower:
                val = 1 if fatigue else 0

            # HYPERTENSION
            elif "hypertension" in feature_lower:
                val = 1 if hypertension else 0

            # GENDER
            elif "male" in feature_lower:
                val = 1 if male else 0

            values.append(val)

        return values