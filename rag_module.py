import re


class RAGModule:

    def __init__(self, file_path):

        with open(file_path, "r") as f:
            self.knowledge = [
                line.strip()
                for line in f.readlines()
                if line.strip()
            ]

    # =========================
    # Clinical Query Processing
    # =========================
    def preprocess(self, text):

        text = text.lower()

        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        return text.split()

    # =========================
    # Semantic Retrieval
    # =========================
    def retrieve(self, query):

        query_words = set(self.preprocess(query))

        best_match = None
        best_score = 0

        for line in self.knowledge:

            line_words = set(self.preprocess(line))

            overlap = query_words.intersection(line_words)

            score = len(overlap)

            # Weighted medical relevance
            medical_keywords = [
                "heart",
                "blood",
                "pressure",
                "cholesterol",
                "smoking",
                "diabetes",
                "obesity",
                "cardiovascular",
                "hypertension"
            ]

            for keyword in medical_keywords:

                if keyword in overlap:
                    score += 2

            if score > best_score:

                best_score = score
                best_match = line

        # =========================
        # Relevance Threshold
        # =========================
        if best_score >= 2:

            return f"""
Retrieved Medical Knowledge:
{best_match}
"""

        return "No relevant information found."