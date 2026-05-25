import datetime

def log_decision(result):
    with open("decision_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now()} | {result}\n")