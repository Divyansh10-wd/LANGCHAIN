from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)
import streamlit as st
st.header("Research Tool")

user_input=st.text_input("Enter your query:")

if st.button("Summarize"):

    result = model.invoke(user_input)
    st.write(result.content)
