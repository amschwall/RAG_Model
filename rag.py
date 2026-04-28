import os
import re
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict
from datasets import DatasetDict, load_dataset
import faiss
from tqdm import tqdm
from gensim.models import Word2Vec
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
import torch


# Skip-gram embedding model
def tokenize_code(text):
    """simple whitespace tokenizer, lowercase"""
    return text.lower().split()

class CodeEmbedder:
    """train word2vec on code, then mean-pool token vectors to get sequence embeddings"""

    def __init__(self, vector_size=128, window=5, min_count=2, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def train(self, corpus):
        """corpus: list of strings (code snippets). tokenizes and trains word2vec."""
        tokenized = [tokenize_code(doc) for doc in corpus]
        print(f"training word2vec on {len(tokenized)} documents...")
        self.model = Word2Vec(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=1,  # skip-gram
            epochs=10
        )
        print(f"vocabulary size: {len(self.model.wv)}")
        print(f"embedding dimension: {self.vector_size}")

    def encode(self, texts):
        """mean-pool token vectors to get one vector per text"""
        embeddings = []
        for text in texts:
            tokens = tokenize_code(text)
            vecs = [self.model.wv[t] for t in tokens if t in self.model.wv]
            if vecs:
                embeddings.append(np.mean(vecs, axis=0))
            else:
                embeddings.append(np.zeros(self.vector_size))
        return np.array(embeddings, dtype=np.float32)

def process_codexglue_example(example):
    buggy = example["buggy"]
    fixed = example["fixed"]

    return {
        "buggy": buggy,
        "fixed": fixed,
        "embed_text": buggy   # what you embed for retrieval
    }

class CodeRAGRetriever:
    """RAG retriever for CodeXGLUE bug-fixing dataset (Java only)."""

    def __init__(self, kb_dir=os.getcwd()):
        print(f"Loading KB from: {kb_dir}")

        # Load config
        with open(os.path.join(kb_dir, "config.json"), "r") as f:
            self.config = json.load(f)

        # Load FAISS index
        self.index = faiss.read_index(os.path.join(kb_dir, "java_index.faiss"))

        # Load metadata (buggy/fixed pairs)
        with open(os.path.join(kb_dir, "java_metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)

        # Load embedder
        self.embedder = CodeEmbedder(vector_size=self.config["vector_size"])
        self.embedder.model = Word2Vec.load(os.path.join(kb_dir, "word2vec.model"))

        print(f"Loaded {self.config['java_examples']} Java examples")

    def retrieve(self, query, top_k=3):
        """Retrieve top-k similar buggy methods."""
        # Encode query
        query_embedding = self.embedder.encode([query])

        # Search FAISS index
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= len(self.metadata):
                continue

            item = self.metadata[idx]

            results.append({
                "buggy": item["buggy"],
                "fixed": item["fixed"],
                "similarity": 1 / (1 + dist)
            })

        return results

def build_rag_context(retrieved_examples):
    """Build few-shot context string from retrieved buggy→fixed examples."""
    if not retrieved_examples:
        return ""

    parts = ["Here are similar Java bug-fix examples for reference:\n"]

    for i, ex in enumerate(retrieved_examples, 1):
        parts.append(f"--- Example {i} ---")
        parts.append("Buggy code:\n```java")
        parts.append(ex["buggy"])
        parts.append("```\n")
        parts.append("Fixed code:\n```java")
        parts.append(ex["fixed"])
        parts.append("```\n")

    parts.append("Now fix the following Java method:\n")

    return "\n".join(parts)

def build_prompt(task_input, tokenizer, rag_context=""):
    # System message: always Java
    system = (
        "You are an expert Java developer. "
        "Generate the corrected Java method based on the buggy input."
    )

    if rag_context:
        system += " Use the provided examples as reference for how to fix similar bugs."

    # User content: either context + task or just task
    user_content = f"{rag_context}\n{task_input}" if rag_context else task_input

    chat = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    return tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True
    )

def generate_code(prompt, tokenizer, model, num_samples=1, max_new_tokens=512, temperature=0.0):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    input_length = inputs['input_ids'].shape[1]

    generations = []
    for _ in range(num_samples):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=(temperature > 0.0),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][input_length:]
        code = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        if "Human:" in code:
            code = code.split("Human:")[0].strip()

        generations.append(code)

    return generations

def main():
    # paths
    dataset = load_dataset("google/code_x_glue_cc_code_refinement", name="medium")
    OUTPUT_DIR = os.getcwd()

    print("loading dataset...")
    print(f"train: {len(dataset['train'])} examples")

    # quick look at the format
    example = dataset['train'][0]
    print("\nbuggy:\n", example['buggy'][:300], "...")
    print("\nfixed:\n", example['fixed'][:300], "...")

    print("Imports complete")

    # Process data for knowledge base
    print("processing java...")
    java_data = [process_codexglue_example(ex) for ex in tqdm(dataset['train'])]
    print(f"Methods: {len(java_data)} examples")

    # Train Word2Vec and generate embeddings
    # Build corpus for training the embedder
    code_corpus = [item['buggy'] for item in dataset] + [item['fixed'] for item in dataset]

    embed_texts = [item['embed_text'] for item in dataset]

    # Train embedder
    embedder = CodeEmbedder(vector_size=128, window=5, min_count=2)
    embedder.train(code_corpus + embed_texts)

    # Encode the entire KB (Java only)
    embeddings = embedder.encode([item['embed_text'] for item in dataset])

    print(f"embeddings: {embeddings.shape}")

    # Build FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"index: {index.ntotal} vectors")

    # Save everything into the current working directory
    OUTPUT_DIR = os.getcwd()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, os.path.join(OUTPUT_DIR, "java_index.faiss"))

    # Save metadata (buggy/fixed pairs)
    with open(os.path.join(OUTPUT_DIR, "java_metadata.pkl"), "wb") as f:
        pickle.dump(java_data, f)

    # Save the trained word2vec model
    embedder.model.save(os.path.join(OUTPUT_DIR, "word2vec.model"))

    # Save config
    config = {
        "embedding_method": "word2vec",
        "vector_size": embedder.vector_size,
        "dimension": dimension,
        "java_examples": len(java_data)
    }

    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved all artifacts to: {OUTPUT_DIR}")

    # Instantiate retriever
    w2v_retriever = CodeRAGRetriever()

    # Test retrieval
    java_query = "Sort an array of integers in ascending order\npublic void sortArray(int[] arr)"
    print(f"query: {java_query}\n")

    java_results = w2v_retriever.retrieve(java_query, language='java', top_k=3)

    print(f"retrieved {len(java_results)} java examples:")
    for i, r in enumerate(java_results, 1):
        print(f"\n  {i}. similarity: {r['similarity']:.4f}")
        print(f"     summary: {r['summary'][:80]}...")
        print(f"     signature: {r['signature'][:60]}...")

    # Load generator model
    MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"loaded {MODEL_NAME} on {model.device}")

if __name__ == "__main__":
    main()