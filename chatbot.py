import os
import requests
import re
import pickle
from sentence_transformers import SentenceTransformer, util
import sys
# -- CONFIG --
OPENROUTER_API_KEY = "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"
MODEL_NAME = "mistralai/mixtral-8x7b-instruct"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "https://chat.openai.com",
    "Content-Type": "application/json"
}
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Initialize components
embedder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')


# -- Document Processing --
def load_docs(path):
    """Load and clean documents"""
    with open(path, encoding="utf-8") as f:
        chunks = f.read().split("\n\n")
    return [re.sub(r'\s+', ' ', c.strip()) for c in chunks if c.strip()]


def load_or_create_embeddings():
    """Load or create document embeddings"""
    english_docs = load_docs("malla_reddy_summary.txt")

    if os.path.exists("english_embeds.pkl"):
        with open("english_embeds.pkl", "rb") as f:
            english_embeds = pickle.load(f)
    else:
        english_embeds = embedder.encode(english_docs, convert_to_tensor=True)
        with open("english_embeds.pkl", "wb") as f:
            pickle.dump(english_embeds, f)

    return english_docs, english_embeds


english_docs, english_embeds = load_or_create_embeddings()


# -- RAG Retriever --
def retrieve_top_k(query, k=4):
    """Retrieve relevant document chunks (English only)"""
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, english_embeds, top_k=k)[0]
    return "\n\n".join([english_docs[hit['corpus_id']] for hit in hits])


# -- LLM Integration (Malla Reddy Persona) --
def generate_answer_as_malla_reddy(query):
    """Generate response using OpenRouter API (English only)"""
    system_prompt = (
        "You are Chamakura Malla Reddy, a respected politician, educationist, and businessman from Telangana, India. "
        "Speak in the first person as Malla Reddy. "
        "I was born on September 9, 1953, and raised in Bowenpally. "
        "I attended Wesley Boys High School and completed my intermediate education at Mahabubia Junior Boys College. "
        "I am married to Kalpana Reddy and have three children: two sons, Mahender Reddy and Dr. Bhadra Reddy, and a daughter, Mamatha Reddy. "
        "I began my journey humbly as a milk seller and through dedication and hard work, I became the founder and chairman of the Malla Reddy Educational Society, "
        "which operates multiple schools, colleges, universities, and research institutes such as Malla Reddy Health City, Malla Reddy Engineering College, and Malla Reddy University. "
        "Politically, I have served as the Member of Parliament for Malkajgiri and currently serve as the Member of the Legislative Assembly from Medchal constituency. "
        "From 2019 to 2023, I was the Minister of Labour and Employment in Telangana, also overseeing Factories, Skill Development, and Women & Child Welfare. "
        "I initially joined Telugu Desam Party in 2014 and was the only TDP MP elected from Telangana, later joining Bharat Rashtra Samithi in 2016. "
        "I strongly believe in providing affordable, quality education primarily for the middle class and empowering youth by embedding practical skills such as AI, coding, and data science into the curriculum from the first year. "
        "Answer all questions respectfully and confidently, as if you are Malla Reddy himself. "
        "Keep answers concise and to the point"
        "Answers should be one line."
        "If you are unsure about a fact, respond gracefully and honestly."
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
        data = response.json()

        if 'choices' in data:
            return data['choices'][0]['message']['content'].strip()
        else:
            print("API ERROR:", data)
            return "Sorry, no valid response from the model."
    except Exception as e:
        print(f"API Exception: {e}")
        return "Sorry, I couldn't process your request."


# -- Simplified Main Loop --
def main():
    print("Malla Reddy Bot (English Mode) | Type 'exit' to quit\n")
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() in ['exit', 'quit']:
            break

        # Retrieve context (optional)
        context = retrieve_top_k(query) if english_docs else ""

        # Generate and print response
        response = generate_answer_as_malla_reddy(query)
        print(f"\nMalla Reddy: {response}\n")


if __name__ == "__main__":
    main()