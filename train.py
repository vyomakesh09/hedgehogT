import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from hedgehogTransformer.hht import HedgehogTransformer  # Adjust import path as needed

def save_checkpoint(epoch, batch_idx, model, optimizer, loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['batch_idx'], checkpoint['loss']

# Ensure the checkpoint directory exists
checkpoint_path = "/home/v/hedgehog_transformer_checkpoints"
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
    return {"input_ids": tokenized_inputs["input_ids"], "labels": labels}


# Tokenize dataset and set format for PyTorch
tokenized_datasets = dataset.map(tokenize_and_prepare_labels, batched=True)
tokenized_datasets.set_format(type="torch", columns=["input_ids", "labels"])

# DataLoader for both training and testing
train_loader = DataLoader(tokenized_datasets["train"], batch_size=8, shuffle=True)
test_loader = DataLoader(
    tokenized_datasets["validation"], batch_size=8
)  # Adjust as needed for test set

# Model configuration
vocab_size = tokenizer.vocab_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HedgehogTransformer(
    src_vocab_size=vocab_size,  # Adjust according to the actual vocab size
    embed_size=512,
    num_layers=6,
    heads=8,
    device=device,
    forward_expansion=4,
    dropout=0.1,
    max_length=512,  # Adjust according to the actual max length needed
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Implement learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Check if resuming from checkpoint
start_epoch = 0
start_batch = 0
if os.path.exists("path_to_checkpoint.pth"):
    start_epoch, start_batch, _ = load_checkpoint("path_to_checkpoint.pth", model, optimizer)

# Training loop
model.train()
for epoch in range(start_epoch, 3):  # Adjust number of epochs as needed
    for batch_idx, batch in enumerate(train_loader, start=start_batch):
        if batch_idx < start_batch:
            continue  # Skip to the next batch if resuming

        inputs, labels = batch["input_ids"].to(device), batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.view(-1, outputs.size(-1))
        labels = labels.view(-1)

        loss = criterion(outputs, labels)
        if torch.isnan(loss):
            save_checkpoint(epoch, batch_idx, model, optimizer, loss.item(), os.path.join(checkpoint_path, "checkpoint_nan_loss.pth"))
            print("NaN loss encountered. Checkpoint saved before exiting.")
            break  # Exit training loop

        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch: {epoch + 1}, Batch: {batch_idx}, Loss: {loss.item()}")

        # Save checkpoints periodically
        if batch_idx % 100 == 0:
            checkpoint_filename = f"checkpoint_epoch_{epoch}_batch_{batch_idx}.pth"
            save_checkpoint(epoch, batch_idx, model, optimizer, loss.item(), os.path.join(checkpoint_path, checkpoint_filename))
            print(f"Checkpoint saved to {checkpoint_filename}")

    scheduler.step()  # Adjust the learning rate based on the scheduler

    # Reset start_batch for the next epoch
    start_batch = 0

    if torch.isnan(loss):
        break  # Exit if NaN loss was encountered


# Save the final model
final_model_path = os.path.join(checkpoint_path, "final_hedgehog_transformer_model.pth")
torch.save(model.state_dict(), final_model_path)
print("Training and evaluation complete. Model saved to", final_model_path)

# Evaluation loop
model.eval()
with torch.no_grad():
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch_idx, batch in enumerate(test_loader):
        inputs, labels = batch["input_ids"].to(device), batch["labels"].to(device)
        outputs = model(inputs)
        outputs = outputs.view(-1, outputs.size(-1))
        labels = labels.view(-1)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_samples
    print(f"Test Loss: {avg_loss}, Accuracy: {accuracy}")


