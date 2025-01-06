import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def get_movie_chain():
    prompt_template = """
    You are a knowledgeable movie assistant. Answer movie-related questions with detailed recommendations and insights. If a movie is not in the dataset, recommend similar ones based on genre, rating, or director.
    For non-movie-related questions, respond with: "I don't know. I only handle movie-related queries."

    Question: {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

def main():
    st.set_page_config(page_title="MovieGPT", layout="wide")
    st.header("🎥 ASK ME ANYTHING ABOUT MOVIES 🤖")

    question = st.text_input("Enter your question:")
    if question:
        chain = get_movie_chain()
        response = chain.run({"question": question})
        st.write("Response:", response)

    st.markdown("""
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            text-align: center;
            padding: 10px 0;
        }
        .profile-image {
            border-radius: 50%;
            width: 40px;
            height: 40px;
            vertical-align: middle;
            margin-right: 10px;
        }
        </style>
        <div class="footer">
            
            </a>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
