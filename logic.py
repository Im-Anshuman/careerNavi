import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_vectorized(path="careers_vectorized.json"):
    with open(path, "r", encoding="utf-8") as f:
        careers = json.load(f)
    for c in careers:
        c["_vector"] = np.array(c["_vector"], dtype=float)
    return careers

def preprocess_input_list(text):
    if isinstance(text, str):
        items = [t.strip() for t in text.split(",") if t.strip()]
    else:
        items = list(text)
    return [t.lower() for t in items]

def user_to_vector(user_skills, user_interests, model):
    text = " . ".join(user_skills + user_interests)
    return model.encode(text)

def rank_careers_by_embedding(user_vec, careers, top_n=3):
    career_vecs = np.stack([c["_vector"] for c in careers])
    sims = cosine_similarity([user_vec], career_vecs)[0]
    idxs = np.argsort(sims)[::-1][:top_n]
    results = []
    for i in idxs:
        results.append((careers[i], float(sims[i])))
    return results

def rule_based_score(user_skills, user_interests, career, skill_weight=0.7, interest_weight=0.3):
    skills = [s.lower() for s in (career.get("required_skills") or career.get("skills") or [])]
    interests = [s.lower() for s in (career.get("interest_tags") or career.get("interests") or [])]
    skill_match = len(set(user_skills) & set(skills)) / max(1, len(skills))
    interest_match = len(set(user_interests) & set(interests)) / max(1, len(interests))
    return skill_weight * skill_match + interest_weight * interest_match

def get_personalized_phase(user_skills, career):
    skills = [s.lower() for s in (career.get("required_skills") or career.get("skills") or [])]
    match_ratio = len(set(user_skills) & set(skills)) / max(1, len(skills))
    if match_ratio < 0.3:
        phase = "Beginner"
    elif match_ratio < 0.7:
        phase = "Intermediate"
    else:
        phase = "Advanced"
    return phase

def missing_skills(user_skills, career):
    skills = [s.lower() for s in (career.get("required_skills") or career.get("skills") or [])]
    return list(set(skills) - set(user_skills))
