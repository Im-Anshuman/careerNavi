import streamlit as st
from logic import (
    load_vectorized, preprocess_input_list, user_to_vector,
    rank_careers_by_embedding, get_personalized_phase, missing_skills, rule_based_score
)
from sentence_transformers import SentenceTransformer
import numpy as np

st.set_page_config(page_title="Career Roadmap Generator", layout="wide")
st.title("🚀 AI Career Roadmap Generator")

st.markdown("Enter your current skills and interests (comma-separated). Example: Python, SQL, statistics")

with st.form("input_form"):
    skills_input = st.text_input("Your skills (comma-separated)", placeholder="Python, SQL, Excel")
    interests_input = st.text_input("Your interests (comma-separated)", placeholder="data, finance, ai")
    use_embedding = st.checkbox("Use semantic AI matching (recommended)", value=True)
    submitted = st.form_submit_button("Generate Roadmap")

if not submitted:
    st.info("Fill the form and click Generate Roadmap.")
    st.stop()

user_skills = preprocess_input_list(skills_input)
user_interests = preprocess_input_list(interests_input)

st.info("Loading careers...")
careers = load_vectorized("careers_vectorized.json")

if use_embedding:
    st.info("Encoding input with embedding model (this may take a few seconds)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    user_vec = user_to_vector(user_skills, user_interests, model)
    top = rank_careers_by_embedding(user_vec, careers, top_n=4)
    results = []
    for career_dict, sim in top:
        phase = get_personalized_phase(user_skills, career_dict)
        missing = missing_skills(user_skills, career_dict)
        results.append({"career": career_dict["career"], "score": float(sim), "career_obj": career_dict, "phase": phase, "missing": missing})
else:
    scored = []
    for c in careers:
        score = rule_based_score(user_skills, user_interests, c)
        scored.append((c, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    results = []
    for c, score in scored[:4]:
        phase = get_personalized_phase(user_skills, c)
        missing = missing_skills(user_skills, c)
        results.append({"career": c["career"], "score": float(score), "career_obj": c, "phase": phase, "missing": missing})

st.header("Top recommendations")
for i, r in enumerate(results, 1):
    st.subheader(f"{i}. {r['career']} — Score: {round(r['score']*100, 1)}%")
    st.markdown(f"**Recommended Level:** {r['phase']}")
    ms = r["missing"][:8]
    if ms:
        st.markdown(f"**Missing skills (priority):** {', '.join(ms)}")
    else:
        st.markdown("**You already have most required skills — you can aim for Advanced tasks.**")
    st.markdown("**Roadmap (key steps):**")
    roadmap = r["career_obj"].get("roadmap", {})
    if roadmap:
        for step in roadmap.get(r["phase"], []):
            st.write("- " + step)
    st.markdown("**Resources:**")
    for res in r["career_obj"].get("resources", []):
        st.write("- " + res)
    st.divider()
