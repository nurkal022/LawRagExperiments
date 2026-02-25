"""
Generate questions with gold passages using GPT-4o-mini.
Samples random chunks and asks GPT to generate legal questions.
"""

import json
import random
import os
import sys
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

# Config
NUM_QUESTIONS = 100
CHUNKS_FILE = Path(__file__).parent.parent / "data" / "chunks" / "all_chunks.json"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "questions.json"
MODEL = "gpt-4o-mini"

# Prompt for question generation
GENERATION_PROMPT = """Ты - эксперт по казахстанскому праву. На основе данного юридического текста:

1. Сформулируй один конкретный юридический вопрос на КАЗАХСКОМ языке, на который этот текст отвечает
2. Дай краткий эталонный ответ на КАЗАХСКОМ языке (2-3 предложения)

Текст:
---
{text}
---

Ответь СТРОГО в JSON формате:
{{
  "question": "вопрос на казахском",
  "gold_answer": "ответ на казахском"
}}

Если текст не содержит полезной юридической информации или слишком короткий, верни:
{{
  "question": null,
  "gold_answer": null
}}"""


def load_chunks():
    """Load all chunks."""
    print(f"Loading chunks from {CHUNKS_FILE}...")
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")
    return chunks


def sample_chunks(chunks, n=NUM_QUESTIONS * 2):
    """Sample random chunks, preferring longer ones."""
    # Filter chunks with reasonable length (at least 200 chars)
    good_chunks = [c for c in chunks if len(c.get('text', '')) >= 300]
    print(f"Found {len(good_chunks)} chunks with 300+ chars")
    
    # Sample more than needed to account for failures
    sampled = random.sample(good_chunks, min(n, len(good_chunks)))
    return sampled


def generate_question(client, chunk_text):
    """Generate question using GPT-4o-mini."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Ты помощник для генерации юридических вопросов."},
                {"role": "user", "content": GENERATION_PROMPT.format(text=chunk_text[:3000])}
            ],
            temperature=0.7,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"Error: {e}")
        return {"question": None, "gold_answer": None}


def main():
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env")
        return
    
    client = OpenAI(api_key=api_key)
    
    # Load and sample chunks
    chunks = load_chunks()
    sampled = sample_chunks(chunks, NUM_QUESTIONS * 2)
    
    # Build chunk lookup for doc_id -> chunk indices
    chunk_lookup = {}
    for i, chunk in enumerate(chunks):
        doc_id = chunk['doc_id']
        if doc_id not in chunk_lookup:
            chunk_lookup[doc_id] = []
        chunk_lookup[doc_id].append(i)
    
    # Generate questions
    questions = []
    print(f"\nGenerating {NUM_QUESTIONS} questions...")
    
    for chunk in tqdm(sampled):
        if len(questions) >= NUM_QUESTIONS:
            break
        
        # Generate question
        result = generate_question(client, chunk['text'])
        
        if result.get('question') and result.get('gold_answer'):
            # Find global chunk index
            doc_id = chunk['doc_id']
            chunk_idx = chunk.get('chunk_idx', 0)
            
            # Get global index from chunk_lookup
            doc_chunks = chunk_lookup.get(doc_id, [])
            global_idx = doc_chunks[chunk_idx] if chunk_idx < len(doc_chunks) else doc_chunks[0] if doc_chunks else 0
            
            question_obj = {
                "id": f"q{len(questions)+1:03d}",
                "question": result['question'],
                "gold_answer": result['gold_answer'],
                "gold_passages": [
                    {"doc_id": doc_id, "chunk_id": chunk_idx, "global_idx": global_idx}
                ]
            }
            questions.append(question_obj)
            
            if len(questions) % 10 == 0:
                print(f"\nGenerated {len(questions)} questions")
                # Save intermediate results
                save_questions(questions)
    
    # Final save
    save_questions(questions)
    print(f"\nDone! Generated {len(questions)} questions")
    print(f"Saved to {OUTPUT_FILE}")


def save_questions(questions):
    """Save questions to JSON file."""
    output = {
        "questions": questions,
        "_meta": {
            "total": len(questions),
            "model": MODEL,
            "note": "Generated automatically, review recommended"
        }
    }
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()
