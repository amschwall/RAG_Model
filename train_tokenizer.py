import sentencepiece as sp
from transformers import T5Tokenizer
import os
from datasets import load_dataset

def main():
    csn = load_dataset("code_search_net", "java")
    methods = csn["train"].shuffle(seed=42).select(range(50000))

    with open("java_corpus.txt", "w") as f:
        for m in methods:
            f.write(m["whole_func_string"].strip().replace("\n", " ") + "\n")

    save_dir = "t5_tokenizer"
    os.makedirs(save_dir, exist_ok=True)

    # Build the list of special tokens
    special_tokens = [
        "<pad>",
    ] + [f"<extra_id_{i}>" for i in range(100)]

    # Join into comma-separated string for SentencePiece
    user_defined = ",".join(special_tokens)

    sp.SentencePieceTrainer.Train(
        input="java_corpus.txt", # this is wrong
        model_prefix=f"{save_dir}/tokenizer",
        vocab_size=16384,
        model_type="unigram",
        user_defined_symbols=user_defined,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )

    tok = T5Tokenizer(vocab_file="t5_tokenizer/tokenizer.model", extra_ids=100)

    tok.save_pretrained("t5_tokenizer")
    print("Model has been trained!")

if __name__ == "__main__":
    main()