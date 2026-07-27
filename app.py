import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


st.title("  AI Story Generator")


theme = st.text_input("Enter theme (e.g., dragon and princess)")
genre = st.selectbox("Genre:", ["Adventure", "Mystery","sci-fi", "Romance"])
tone = st.selectbox("Tone:", ["Light-hearted", "Dark", "Whimsical", "casual", "formal"])
length = st.selectbox("Length", ["short ","medium","long"])


length_map ={
    "short": 50,
    "medium": 100,
    "long": 200
}

#generator = pipeline("text-generation", model="gpt2", truncation=True)
generator = pipeline(
    "text-generation",
    model="gpt2",
    truncation=True,
    model_kwargs={"low_cpu_mem_usage": True}
)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", low_cpu_mem_usage=True)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, truncation=True)
# RUN COMMAND
# streamlit run app.py
if st.button("Generate Story"):
    if theme:
        prompt = f"write a {tone} {genre} story about {theme}"
        story = generator(
            prompt,
            max_length=length_map[length],
            temperature=0.8,
            pad_token_id=50256
        )
        st.subheader("Generated Story")
        st.write(story[0]['generated_text'])
    else:
        st.warning("Please enter a theme to generate story")
        

