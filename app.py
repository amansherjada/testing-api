import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pinecone import Pinecone

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Hugging Face embeddings
embedding_model = HuggingFaceEmbeddings()

# Initialize Pinecone
PINECONE_API_KEY = "pcsk_475ix6_QNMj2etqYWbrUz2aKFQebCPzCepmZEsZFoWsMG3wjYvFaxdUFu73h7GWbieTeti"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("ahlchatbot-customer")

# Load Vector Store from Pinecone
vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key="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")

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

prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Create RetrievalQA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        event_type = data.get("event", "")
        contact = data.get("contact", {})
        message = data.get("message", {})
        user_message = message.get("text", "")
        user_phone = contact.get("phone", "")

        if not user_message:
            return jsonify({"error": "No message content provided"}), 400

        # Process user message
        result = qa.invoke({"query": user_message})
        answer = result["result"]

        # Log the interaction
        print(f"Received message from {user_phone}: {user_message}")
        print(f"Bot response: {answer}")

        # Here, you can add code to send the 'answer' back to the user via Gallabox's messaging API

        return jsonify({"messages": [{"text": answer}]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Render's provided PORT
    app.run(host="0.0.0.0", port=port, debug=True)
