import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from hedgehogTransformer.hht import HedgehogTransformer

'''# Assuming `BaseAttention` is the class for the pre-trained model's attention mechanism
class BaseAttention(nn.Module):
    def __init__(self, head_dim):
        super(BaseAttention, self).__init__()
        self.head_dim = head_dim
        self.q_proj = nn.Linear(head_dim, head_dim)
        self.k_proj = nn.Linear(head_dim, head_dim)

    def forward(self, hidden_states, output_attentions=False):
        # A placeholder for the base attention's forward pass
        # For demonstration, we'll just return random tensors simulating attention weights
        random_attention = torch.rand(
            hidden_states.size(0), hidden_states.size(1), hidden_states.size(1)
        )
        return random_attention


# Replace the ... with your actual attention module
base_attn = BaseAttention(head_dim=64) # Your base attention module here

# Initialize HedgehogAttention with the base attention mechanism
hedgehog_attn = HedgehogAttention(base_attn, head_dim=64)

# Simulate hidden states tensor as the input to the attention mechanism
hidden_states = torch.rand(2, 10, 64)  # (batch_size, seq_length, head_dim)

# Forward pass through HedgehogAttention
outputs = hedgehog_attn(hidden_states, output_attentions=True)

# Handle the outputs based on the training mode
if isinstance(outputs, tuple) and len(outputs) == 3:
    loss, pred_attn, true_attn = outputs
else:
    pred_attn = outputs
    
# Print the shapes to verify the output (the shapes depend on your specific implementation details)
print("Predicted Attention Shape:", pred_attn.shape)
print("True Attention Shape:", true_attn.shape)
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Example input text
input_text = 'Hello, world'

# Tokenize the input
tokens = tokenizer.encode(input_text, return_tensors='pt', add_special_tokens=True)

# Ensure the model and tokens are on the same device
tokens = tokens.to(device)


model = HedgehogTransformer(
    src_vocab_size=10000,  # Example Vocab Size
    embed_size=512,
    num_layers=6,
    heads=8,
    device=device,
    forward_expansion=4,
    dropout=0.1,
    max_length=100,  # Example Max Length
).to(device)


'''# Get model output 
output = model(tokens)

# Convert output logits to probabilities 
probabilities = F.softmax(output, dim=-1)

# Optionally, get the predicted toekn IDs
predicted_token_id = torch.argmax(probabilities, dim=-1)

print(predicted_token_id)'''

