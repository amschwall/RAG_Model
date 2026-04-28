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

    # Load model and tokenizer from checkpoint
    model = T5ForConditionalGeneration.from_pretrained("pretrained_t5").to(device)
    tokenizer = T5Tokenizer.from_pretrained("pretrained_t5")

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

    model.save_pretrained("finetuned_pretrained_model")
    tokenizer.save_pretrained("finetuned_pretrained_model")

if __name__ == "__main__":
    main()