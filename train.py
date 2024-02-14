import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from hedgehogTransformer.hedgehogT import (
    HedgehogTransformer,
)  # Adjust import path as needed
import pickle


# Ensure the checkpoint directory exists
checkpoint_path = "hedgehog_transformer_checkpoints"
os.makedirs(checkpoint_path, exist_ok=True)

# Load the dataset
dataset = load_dataset("wikitext", "wikitext-103-v1")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# Function to tokenize and prepare labels
def tokenize_and_prepare_labels(examples):
    tokenized_inputs = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    labels = tokenized_inputs["input_ids"][:, 1:].clone()
    tokenized_inputs["input_ids"] = tokenized_inputs["input_ids"][:, :-1]
    # print(tokenized_inputs["input_ids"], tokenized_inputs["input_ids"])
    return {"input_ids": tokenized_inputs["input_ids"], "labels": labels}


# with open('processed_dataset.pkl', 'rb') as f:
#    tokenized_datasets = pickle.load(f)

# Tokenize dataset and set format for PyTorch
tokenized_datasets = dataset.map(tokenize_and_prepare_labels, batched=True)
tokenized_datasets.set_format(type="torch", columns=["input_ids", "labels"])


# with open('processed_dataset.pkl', 'wb') as f:
#  pickle.dump(tokenized_datasets, f)

with open("processed_dataset.pkl", "rb") as f:
    tokenized_datasets = pickle.load(f)

# DataLoader
train_loader = DataLoader(tokenized_datasets["train"], batch_size=8, shuffle=True)

# Model configuration
vocab_size = tokenizer.vocab_size
model = HedgehogTransformer(
    dim=512, depth=6, heads=8, dim_head=64, mlp_dim=2048, vocab_size=vocab_size
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(3):  # Adjust the number of epochs as needed
    for batch_idx, batch in enumerate(train_loader):
        inputs, labels = batch["input_ids"], batch["labels"]

        # print('1inputs, labels)
        print(f"1; {inputs.shape},2: {labels.shape}")

        # inputs = inputs.unsqueeze(-1)
        # labels = labels.unsqueeze(-1)

        print(f"3; {inputs.shape},4: {labels.shape}")

        optimizer.zero_grad()
        outputs = model(inputs)

        # Adjust output dimensions for calculating loss
        outputs = outputs.view(-1, outputs.size(-1))
        labels = labels.view(-1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch id: {epoch + 1}, Batch id: {batch_idx}, Loss: {loss.item()}")

        # Save checkpoints periodically
        if batch_idx % 100 == 0:
            checkpoint = {
                "epoch": epoch,
                "batch_idx": batch_idx,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            }
            checkpoint_filename = f"checkpoint_epoch_{epoch}_batch_{batch_idx}.pth"
            checkpoint_filepath = os.path.join(checkpoint_path, checkpoint_filename)
            torch.save(checkpoint, checkpoint_filepath)
            print(f"Checkpoint saved to {checkpoint_filepath}")

# Save the final model
final_model_path = os.path.join(checkpoint_path, "final_hedgehog_transformer_model.pth")
torch.save(model.state_dict(), final_model_path)
print("Training complete. Model saved to", final_model_path)
