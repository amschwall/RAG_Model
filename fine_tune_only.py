from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
import random
import numpy as np

class BugFixDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        buggy = self.data[idx]["buggy"]
        fixed = self.data[idx]["fixed"]

        enc = self.tokenizer(
            buggy,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        dec = self.tokenizer(
            fixed,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "labels": dec["input_ids"].squeeze()
        }

def span_corruption(input_ids, tokenizer, corruption_rate=0.15):
    """
    T5-style span corruption (Raffel et al., 2020). Selects ~15% of tokens,
    groups consecutive ones into spans, replaces each span with a unique
    sentinel. Returns (encoder_input, decoder_target).
    """
    input_ids = input_ids.squeeze()
    length = input_ids.size(0)

    # tokens we should never corrupt
    special_ids = set()
    for tok_id in [tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id]:
        if tok_id is not None:
            special_ids.add(tok_id)

    # which positions are eligible for corruption
    eligible = [i for i in range(length) if input_ids[i].item() not in special_ids]

    if len(eligible) == 0:
        sentinel_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
        return input_ids.unsqueeze(0), torch.tensor([sentinel_id]).unsqueeze(0)

    # select ~15% of eligible positions
    num_to_corrupt = max(1, int(len(eligible) * corruption_rate))
    num_to_corrupt = min(num_to_corrupt, len(eligible))
    corrupted_positions = set(sorted(random.sample(eligible, num_to_corrupt)))

    # group consecutive corrupted positions into spans
    spans = []
    current_span = []
    for pos in sorted(corrupted_positions):
        if current_span and pos != current_span[-1] + 1:
            spans.append(current_span)
            current_span = []
        current_span.append(pos)
    if current_span:
        spans.append(current_span)

    # sentinel IDs count downward: extra_id_0, extra_id_1, extra_id_2, ...
    sentinel_base = tokenizer.convert_tokens_to_ids("<extra_id_0>")

    # build encoder input: replace each span with one sentinel
    encoder_tokens = []
    span_idx = 0
    i = 0
    while i < length:
        if span_idx < len(spans) and i == spans[span_idx][0]:
            encoder_tokens.append(sentinel_base - span_idx)
            i = spans[span_idx][-1] + 1
            span_idx += 1
        else:
            encoder_tokens.append(input_ids[i].item())
            i += 1

    # build decoder target: sentinel + dropped tokens for each span, final sentinel at end
    decoder_tokens = []
    for idx, span in enumerate(spans):
        decoder_tokens.append(sentinel_base - idx)
        for pos in span:
            decoder_tokens.append(input_ids[pos].item())
    decoder_tokens.append(sentinel_base - len(spans))

    return (
        torch.tensor(encoder_tokens, dtype=torch.long).unsqueeze(0),
        torch.tensor(decoder_tokens, dtype=torch.long).unsqueeze(0)
    )

class SpanCorruptionDataset(Dataset):
    """
    Tokenizes code upfront, applies fresh span corruption per call.
    Returns encoder input (corrupted) and decoder target (spans only).
    """
    def __init__(self, code_snippets, tokenizer, max_length=512, corruption_rate=0.15):
        self.tokenizer = tokenizer
        self.corruption_rate = corruption_rate

        self.token_ids = [
            tokenizer.encode(code, max_length=max_length, truncation=True)
            for code in code_snippets
        ]
        print(f"Dataset: {len(self.token_ids)} samples, max_length={max_length}")

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.token_ids[idx], dtype=torch.long)
        encoder_input, decoder_target = span_corruption(
            input_ids, self.tokenizer, self.corruption_rate
        )
        return {
            "input_ids": encoder_input.squeeze(0),
            "labels": decoder_target.squeeze(0)
        }
    
def collate_fn(batch, tokenizer):
    """Pad variable-length encoder inputs and decoder targets within each batch."""
    input_ids_list = [item["input_ids"] for item in batch]
    labels_list = [item["labels"] for item in batch]

    # pad encoder inputs with pad_token_id
    max_input_len = max(x.size(0) for x in input_ids_list)
    padded_inputs = torch.full((len(batch), max_input_len), tokenizer.pad_token_id, dtype=torch.long)
    attention_masks = torch.zeros((len(batch), max_input_len), dtype=torch.long)
    for i, ids in enumerate(input_ids_list):
        padded_inputs[i, :ids.size(0)] = ids
        attention_masks[i, :ids.size(0)] = 1

    # pad decoder targets with -100 (ignored by CrossEntropyLoss)
    max_label_len = max(x.size(0) for x in labels_list)
    padded_labels = torch.full((len(batch), max_label_len), -100, dtype=torch.long)
    for i, lbl in enumerate(labels_list):
        padded_labels[i, :lbl.size(0)] = lbl

    return {
        "input_ids": padded_inputs,
        "attention_mask": attention_masks,
        "labels": padded_labels
    }

def main():
    # reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset
    dataset = load_dataset("google/code_x_glue_cc_code_refinement", name="medium")

    # load tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5_tokenizer")

     # define model
    t5_config = T5Config(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
        d_model=512,
        d_ff=2048,
        d_kv=64,
        num_heads=8,
        num_layers=6,
        num_decoder_layers=6
    )
    model = T5ForConditionalGeneration(config=t5_config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Build dataset
    train_data = dataset["train"]
    train_dataset = BugFixDataset(train_data, tokenizer)

    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda batch: collate_fn(batch, tokenizer))

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print("Starting 3‑epoch finetuning...")

    for epoch in range(3):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch["input_ids"].to(device),
                labels=batch["labels"].to(device)
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} — Loss: {avg_loss:.4f}")

    model.save_pretrained("finetuned_only_model")
    tokenizer.save_pretrained("finetuned_only_model")

if __name__ == "__main__":
    main()