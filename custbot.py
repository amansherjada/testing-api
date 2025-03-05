import os
import streamlit as st
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pinecone import Pinecone
import re

load_dotenv()

# Initialize Hugging Face embeddings
embedding_model = HuggingFaceEmbeddings()

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("ahlchatbot-customer")

# Load Vector Store from Pinecone
vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Custom Prompt Template
custom_prompt_template = """
# American Hairline Customer Support AI Assistant

## Core Objective
Provide exceptional, personalized customer support for non-surgical hair replacement solutions, guiding potential clients through their hair restoration journey.

## Interaction Protocol

### Communication Principles
- Be empathetic and understanding
- Provide clear, concise information
- Prioritize customer comfort and confidence
- Maintain a professional yet warm tone

### Handling Customer Interactions

#### Price Inquiries
- CRITICAL INSTRUCTION: When asked about specific pricing
  1. DO NOT share exact price details
  2. Respond with a standardized message:
     "Thank you for your interest in our hair replacement solutions! Pricing is personalized based on individual needs and hair characteristics. Our dedicated customer support team can provide a tailored consultation to give you precise pricing information. Please contact us at [Customer Support Phone Number/Email] to schedule a free, no-obligation consultation."

#### General Inquiry Handling
1. Listen carefully to customer concerns
2. Provide informative, supportive responses
3. Focus on benefits and personalization
4. Guide towards booking a consultation

## Key Conversation Areas

### Product Information
- Explain non-surgical hair replacement concepts
- Highlight natural appearance
- Emphasize customization options
- Focus on confidence restoration

### Consultation Booking
- Provide clear steps to schedule
- Offer multiple contact methods
- Assure a supportive, pressure-free environment

## Prohibited Actions
- No medical diagnoses
- No comparisons with other providers
- No sharing of personal client information
- NEVER disclose specific pricing details

## Response Structure
- Use clear, simple language
- Break down complex information
- Provide actionable next steps
- Maintain a positive, encouraging tone

## Technical Constraints
- Context-aware responses
- Adaptable to various customer queries
- Seamless integration with support workflow

## Customer Confidence Builders
- Share general success stories
- Emphasize technological advancements
- Highlight personalized approach
- Build trust through transparent communication

## Consultation Encouragement
Always guide conversations towards:
1. Understanding customer needs
2. Scheduling a free consultation
3. Addressing initial concerns
4. Building confidence in the solution

{context}

{question}

"""

prompt = PromptTemplate(
    template=custom_prompt_template,
    input_variables=["context", "question"]
)

# Create RetrievalQA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# Initialize session state #
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to process query and update chat
def process_query(query):
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Add spinner while generating response
    with st.spinner('Generating response...'):
        # Get AI response
        result = qa.invoke({"query": query})
        answer = result["result"]

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    return answer

# Streamlit Chat Interface

st.set_page_config(page_title="AHL Trainer Chatbot by Aman Khan", page_icon="🦱", layout="centered")
st.title(f"AHL Trainer Chatbot")
st.write("Ask questions related to training materials, Developed by [Aman Khan](https://github.com/amansherjada)")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Your message:"):
    answer = process_query(prompt)
    
    # Extract options using regex
    options_pattern = re.findall(r"\((.)\) (.*?)\n", answer)
    
    if options_pattern:
        # Create columns for buttons
        cols = st.columns(len(options_pattern))
        
        # Create buttons for each option
        for i, (key, value) in enumerate(options_pattern):
            if cols[i].button(value, key=f"btn_{len(st.session_state.messages)}_{i}"):
                process_query(value)
                st.rerun()

# Only show buttons for the most recent assistant message
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    last_message = st.session_state.messages[-1]["content"]
    options_pattern = re.findall(r"\((.)\) (.*?)\n", last_message)
    
    if options_pattern:
        cols = st.columns(len(options_pattern))
        for i, (key, value) in enumerate(options_pattern):
            if cols[i].button(value, key=f"last_btn_{i}"):
                process_query(value)
                st.rerun()