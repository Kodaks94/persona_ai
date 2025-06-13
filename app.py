import streamlit as st
from rag_pipeline import answer_query



st.set_page_config(page_title="Persona.AI", layout='wide')
st.title("Persona.AI -- Ask Mahrad's Assistant")

query = st.text_input("Ask a question about Mahrad:", "What are some of Mahrad's projects?")

if query:
    with st.spinner("Thinking like Mahrad... "):
        answer = answer_query(query)
    st.markdown("---")
    st.markdown(f"**Answer:** {answer}")



