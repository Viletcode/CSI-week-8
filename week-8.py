import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load and clean data
df = pd.read_csv("loan_data.csv")
df = df.fillna("Unknown")

# Prepare documents
documents = []
for _, row in df.iterrows():
    doc = f"""
    Applicant Info: Gender = {row['Gender']}, Married = {row['Married']}, Education = {row['Education']}, 
    ApplicantIncome = {row['ApplicantIncome']}, LoanAmount = {row['LoanAmount']}, 
    Credit_History = {row['Credit_History']}, Loan_Status = {row['Loan_Status']}
    """
    documents.append(doc.strip())

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(documents, show_progress_bar=True)

# Create FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# Load generative model (Flan-T5)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Retrieve top-k relevant chunks
def retrieve_context(query, k=5):
    query_vector = embedder.encode([query])
    distances, indices = index.search(query_vector, k)
    return "\n\n".join([documents[i] for i in indices[0]])

# Generate answer using Flan-T5
def generate_answer(query):
    context = retrieve_context(query)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# CLI interaction
if __name__ == "__main__":
    print("📘 Loan Approval Q&A Chatbot (Type 'exit' to quit)\n")
    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        answer = generate_answer(user_query)
        print(f"🤖 Bot: {answer}\n")
