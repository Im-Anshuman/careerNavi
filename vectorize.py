import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "all-MiniLM-L6-v2"

def normalize_text_list(lst):
    return [s.strip().lower() for s in lst]

def career_to_text(career):
    parts = []
    skills = career.get("required_skills") or career.get("skills") or []
    interests = career.get("interest_tags") or career.get("interests") or []
    parts.extend(skills)
    parts.extend(interests)
    if "career" in career:
        parts.append(career["career"])
    if "description" in career:
        parts.append(career["description"])
    return " . ".join(normalize_text_list(parts))

def main():
    model = SentenceTransformer(MODEL_NAME)
    with open("careers_data.json", "r", encoding="utf-8") as f:
        careers = json.load(f)

    for c in tqdm(careers, desc="Embedding careers"):
        text = career_to_text(c)
        vec = model.encode(text).tolist()
        c["_vector"] = vec

    with open("careers_vectorized.json", "w", encoding="utf-8") as f:
        json.dump(careers, f, indent=2, ensure_ascii=False)

    print("Saved careers_vectorized.json (with _vector fields)")

if __name__ == "__main__":
    main()
