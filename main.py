from visualization import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from data_loader import load_data
from model import HealthcareModel
from xai_module import ExplainableAI
from meta_cognition import MetaCognition
from multi_agent import CoordinatorAgent
from rag_module import RAGModule


# =========================
# SYSTEM BANNER
# =========================
print("""
========================================
      GENESIS AI HEALTHCARE SYSTEM
========================================
Self-Evolving Cognitive Agent Framework
for Intelligent Healthcare Assistance

Modules Enabled:
- Explainable AI (XAI)
- Meta-Cognitive Reflection
- Agentic AI Coordination
- Retrieval-Augmented Generation
- Healthcare Risk Prediction
========================================
""")


# =========================
# Knowledge Base Setup
# =========================
if not os.path.exists("knowledge_base.txt"):

    with open("knowledge_base.txt", "w") as f:

        f.write("High blood pressure increases heart disease risk.\n")

        f.write("Smoking increases cardiovascular disease risk.\n")

        f.write("Obesity is a major risk factor for heart disease.\n")

        f.write("Diabetes can increase cardiovascular complications.\n")

        f.write("Regular exercise improves heart health.\n")


# =========================
# Load Dataset
# =========================
(data, feature_names) = load_data("dataset.csv")

X_train, X_test, y_train, y_test = data


# =========================
# RAG MODULE
# =========================
rag = RAGModule("knowledge_base.txt")


# =========================
# Fixed Optimized Depth
# =========================
best_depth = 5

print(f"\nUsing optimized decision depth: {best_depth}")


# =========================
# Train Model
# =========================
model = HealthcareModel(max_depth=best_depth)

model.train(X_train, y_train)


# =========================
# Evaluate Model
# =========================
metrics = model.evaluate(X_test, y_test)

print("\n===== MODEL PERFORMANCE =====")

for key, value in metrics.items():

    print(f"{key}: {value}")


# =========================
# Visualizations
# =========================
try:

    plot_feature_importance(model, feature_names)

    plot_confusion_matrix(metrics["Confusion Matrix"])

    plot_roc_curve(model, X_test, y_test)

    plot_confidence_distribution(model, X_test)

except Exception as e:

    print(f"\nVisualization Warning: {e}")


# =========================
# Initialize Cognitive Modules
# =========================
xai = ExplainableAI(model, feature_names)

meta = MetaCognition(model, threshold=0.75)

agent = CoordinatorAgent(model, xai, meta, rag)


# =========================
# CHAT SYSTEM
# =========================
def chat_system():

    print("\n========================================")
    print("     GENESIS AI INTERACTIVE CONSOLE")
    print("========================================")

    print("\nSample Queries:")

    print("- I am 55 years old with chest pain and BP 160")

    print("- heart rate is 120")

    print("- what causes heart disease")

    print("- I smoke and have high cholesterol")

    print("- explain previous diagnosis")

    while True:

        query = input("\nPatient/Doctor Query > ")

        if query.lower() == "exit":

            print("\nExiting Genesis AI...")

            break

        try:

            response = agent.process_query(
                query,
                feature_names
            )

            print("\n========================================")
            print("          GENESIS AI RESPONSE")
            print("========================================")

            # -------------------------
            # Prediction Output
            # -------------------------
            if "Prediction" in response:

                if response["Prediction"] == 1:

                    print("\n⚠ HIGH RISK OF HEART DISEASE")

                else:

                    print("\n✅ LOW RISK OF HEART DISEASE")

                print(f"\nConfidence Score: {response['Confidence']:.2f}")

                print(f"\nMeta-Cognitive Reflection:")

                print(response["Reflection"])

                print("\nExplainable AI Analysis:")

                print(response["Explanation"])

            # -------------------------
            # General Output
            # -------------------------
            else:

                print(response["Answer"])

            print("========================================")

        except Exception as e:

            print(f"\nSystem Error: {e}")


# =========================
# RUN SYSTEM
# =========================
chat_system()