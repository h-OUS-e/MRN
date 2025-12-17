import torch
import torch.nn as nn


class TRM(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=2, n_heads=8, max_seq_len=512):
        """
        Initializes the TRM with a tiny network (2 layers) as per.
        """
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # Learnable initial states for y (answer) and z (latent reasoning)
        self.y_init = nn.Parameter(torch.randn(1, 1, d_model))
        self.z_init = nn.Parameter(torch.randn(1, 1, d_model))

        # The "Tiny Network": A single network used for both tasks
        # Using 2 layers of Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, 
                                                   batch_first=True, norm_first=True)
        self.net = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output Heads
        self.output_head = nn.Linear(d_model, vocab_size)
        self.halt_head = nn.Linear(d_model, 1) # ACT Halting probability 

    def forward_step(self, x_emb, y_emb, z_emb):
        """
        A single pass of the network.
        The paper states the task is specified by the inclusion of x.
        update z: input = x + y + z
        update y: input = y + z
        """
        # We sum the embeddings 
        combined_input = x_emb + y_emb + z_emb
        return self.net(combined_input)

    def latent_recursion(self, x, y, z, n=6):
        """
        Recursively improves latent z, then refines y[cite: 189].
        n: Number of latent reasoning steps (default 6) [cite: 301]
        """
        # 1. Update latent z recursively 'n' times
        for _ in range(n):
            # Input includes x (question) to update reasoning z
            z = self.forward_step(x, y, z) 
            
        # 2. Refine output answer y ONCE
        # Input excludes x to update answer y (y <- net(y, z)) 
        # We pass zeros_like(x) to simulate the absence of x
        y = self.forward_step(torch.zeros_like(x), y, z)
        
        return y, z

    def deep_recursion(self, x, y, z, n=6, T=3):
        """
        The core TRM logic described in Figure 3[cite: 192].
        x: Embedded input question
        y: Current embedded answer
        z: Current latent reasoning
        T: Total recursion cycles (default 3) [cite: 301]
        """
        
        # Recurse T-1 times WITHOUT gradients to improve states [cite: 192]
        with torch.no_grad():
            for _ in range(T - 1):
                y, z = self.latent_recursion(x, y, z, n)
        
        # Recurse 1 time WITH gradients to learn [cite: 196]
        y, z = self.latent_recursion(x, y, z, n)
        
        # Calculate predictions
        logits = self.output_head(y)
        halt_score = self.halt_head(y).squeeze(-1) # Logits for stopping
        
        # Return detached states for the next supervision step, 
        # but keep gradients on predictions for the current loss.
        # Note: In the paper's loop, they return detached y, z for the NEXT loop iteration.
        return (y.detach(), z.detach()), logits, halt_score

    def forward(self, x_input_ids, n_sup=16, n=6, T=3):
        """
        Simulates the training loop with Deep Supervision[cite: 199].
        Returns a list of outputs for loss calculation.
        """
        B, L = x_input_ids.shape
        
        # 1. Embed Input
        x = self.embedding(x_input_ids) + self.pos_embedding[:, :L, :]
        
        # 2. Initialize y and z [cite: 201]
        # Expand init states to batch size and sequence length
        y = self.y_init.expand(B, L, self.d_model)
        z = self.z_init.expand(B, L, self.d_model)
        
        outputs = []
        
        # 3. Deep Supervision Loop [cite: 202]
        for step in range(n_sup):
            # Run deep recursion
            (y_next, z_next), logits, halt_score = self.deep_recursion(x, y, z, n, T)
            
            outputs.append({
                "logits": logits,
                "halt_score": halt_score,
                "step": step
            })
            
            # Update states for next step
            y = y_next
            z = z_next
            
            # Note: In inference, you would break here if sigmoid(halt_score) > 0.5
            
        return outputs


    

def train_step(model, optimizer, x_input, y_true, n_sup=16):
    """
    Performs one training step with Deep Supervision.
    
    Args:
        x_input: Input sequences (Batch, Seq_Len)
        y_true: Target sequences (Batch, Seq_Len)
        n_sup: Max supervision steps (default 16 as per paper)
    """
    model.train()
    
    # 1. Prediction Loss: Standard Cross Entropy for tokens
    # defined here to handle batch/seq dimensions easily
    criterion_pred = nn.CrossEntropyLoss()
    
    # 2. Halting Loss: Binary Cross Entropy for the halt score
    # We use BCEWithLogitsLoss because our model outputs raw scores
    criterion_halt = nn.BCEWithLogitsLoss()
    
    B, L = x_input.shape
    
    # Initial embeddings setup (matches model.forward structure)
    x = model.embedding(x_input) + model.pos_embedding[:, :L, :]
    y = model.y_init.expand(B, L, model.d_model)
    z = model.z_init.expand(B, L, model.d_model)
    
    total_step_loss = 0
    
    # --- Deep Supervision Loop ---
    # As per Algorithm 3 in Figure 3 of the paper
    for step in range(n_sup):
        optimizer.zero_grad()
        
        # A. Run one recursion cycle (Deep Recursion)
        # Returns detached states for next step, but graph-connected logits for this step
        (y_next, z_next), logits, halt_score = model.deep_recursion(x, y, z)
        
        # --- B. Calculate Prediction Loss ---
        # Reshape logits to (Batch * Seq_Len, Vocab)
        # Reshape targets to (Batch * Seq_Len)
        loss_pred = criterion_pred(logits.reshape(-1, logits.size(-1)), y_true.reshape(-1))
        
        # --- C. Calculate Halting Loss ---
        # Determine if the model is currently correct (Token-level accuracy)
        # Paper implies sentence-level or token-level; usually for puzzles it's full sequence.
        # Here we implement token-level accuracy for simplicity.
        predictions = torch.argmax(logits, dim=-1)
        is_correct = (predictions == y_true).float() # 1.0 if correct, 0.0 if wrong
        
        # The target for the halt head is: "Are we correct yet?"
        loss_halt = criterion_halt(halt_score.reshape(-1), is_correct.reshape(-1))
        
        # --- D. Backpropagation ---
        # Sum losses
        loss = loss_pred + loss_halt
        
        # Backpropagate immediately for this step
        loss.backward()
        optimizer.step()
        
        total_step_loss += loss.item()
        
        # --- E. Prepare for next step ---
        # CRITICAL: We use the DETACHED states for the next iteration.
        # If we didn't detach (returned by deep_recursion), gradients would 
        # flow back through time infinitely, causing memory explosions.
        y = y_next
        z = z_next
        
        # Early stopping during training check (optional/efficiency)
        # The paper notes using "early stopping" if confidence is high [cite: 209]
        # if torch.sigmoid(halt_score).mean() > 0.5: break

    return total_step_loss / n_sup


# --- Example Usage ---
if __name__ == "__main__":
    # Hyperparameters from the paper
    vocab_size = 1000
    trm = TRM(vocab_size=vocab_size, d_model=128, n_layers=2)
    
    # Dummy Input (Batch=2, Seq_Len=10)
    x_input = torch.randint(0, vocab_size, (2, 10))
    
    # Forward pass (simulates N_sup supervision steps)
    results = trm(x_input, n_sup=4) # Reduced n_sup for demo
    
    print(f"Output for supervision step 1 shape: {results[0]['logits'].shape}")
    print(f"Halting score for step 1: {results[0]['halt_score']}")
    