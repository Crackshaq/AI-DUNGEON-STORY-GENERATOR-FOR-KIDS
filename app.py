
import streamlit as st
from story_generator import load_model, generate_story

# Load model once
tokenizer, model = load_model()

st.title("🧙 AI Dungeon Story Generator")
genre = st.selectbox("Choose a genre:", ["Fantasy", "Mystery", "Sci-Fi", "Horror"])
prompt = st.text_area("Enter your story prompt:", "Once upon a time in a distant kingdom...")

if st.button("Generate Story"):
    full_prompt = f"[Genre: {genre}]\n{prompt}"
    st.write("Generating story...")
    stories = generate_story(full_prompt, tokenizer, model)

    for idx, story in enumerate(stories):
        st.subheader(f"Story {idx+1}")
        st.text_area("Generated Text", value=story, height=200)

        file_name = f"story_{idx+1}.txt"
        st.download_button("Download Story", story, file_name)
