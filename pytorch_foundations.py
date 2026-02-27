# ============================================================
# PYTORCH FOUNDATIONS - COMPLETE CODE REFERENCE
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ============================================================
# 1. TENSORS - CREATION
# ============================================================

# --- Basic Creation ---
t1 = torch.tensor([1, 2, 3])                        # from list
t2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])         # 2D tensor
t3 = torch.zeros(3, 4)                              # all zeros
t4 = torch.ones(2, 3)                               # all ones
t5 = torch.full((2, 3), 7.0)                        # filled with value
t6 = torch.eye(4)                                   # identity matrix
t7 = torch.empty(3, 3)                              # uninitialized
t8 = torch.rand(3, 3)                               # uniform [0, 1)
t9 = torch.randn(3, 3)                              # normal dist
t10 = torch.randint(0, 10, (3, 3))                  # random integers

# --- Range & Linspace ---
t11 = torch.arange(0, 10, 2)                        # [0, 2, 4, 6, 8]
t12 = torch.linspace(0, 1, 5)                       # [0, .25, .5, .75, 1]

# --- From NumPy ---
arr = np.array([1.0, 2.0, 3.0])
t13 = torch.from_numpy(arr)                         # shares memory
t14 = torch.tensor(arr)                             # copies data

# --- Like tensors ---
x = torch.rand(2, 3)
t15 = torch.zeros_like(x)
t16 = torch.ones_like(x)
t17 = torch.rand_like(x)

# --- Dtype & Device ---
t18 = torch.tensor([1, 2], dtype=torch.float32)
t19 = torch.tensor([1, 2], dtype=torch.int64)
device = "cuda" if torch.cuda.is_available() else "cpu"
t20 = torch.rand(3, 3).to(device)

print("=== Tensor Creation Done ===\n")

# ============================================================
# 2. TENSORS - INDEXING & SLICING
# ============================================================

x = torch.tensor([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]], dtype=torch.float32)

print("Full tensor:\n", x)
print("Row 0:", x[0])                               # first row
print("Element [1,2]:", x[1, 2])                    # scalar
print("Rows 0-1:\n", x[0:2])                        # slicing
print("Col 1:", x[:, 1])                             # all rows, col 1
print("Submatrix:\n", x[0:2, 1:3])                  # submatrix

# Boolean masking
mask = x > 4
print("Values > 4:", x[mask])

# Fancy indexing
idx = torch.tensor([0, 2])
print("Rows 0 and 2:\n", x[idx])

# ============================================================
# 3. TENSOR OPERATIONS
# ============================================================

a = torch.rand(3, 4)
b = torch.rand(3, 4)

# Arithmetic
print(a + b)
print(a - b)
print(a * b)                                        # element-wise
print(a / b)
print(a ** 2)

# Matrix operations
A = torch.rand(3, 4)
B = torch.rand(4, 5)
print("matmul:", torch.matmul(A, B).shape)          # (3, 5)
print("@:", (A @ B).shape)                          # same

# Reductions
print("sum:", a.sum())
print("sum along dim 0:", a.sum(dim=0))
print("mean:", a.mean())
print("max:", a.max())
print("argmax:", a.argmax())

# Comparison
print("a > 0.5:\n", a > 0.5)
print("torch.where:", torch.where(a > 0.5, a, torch.zeros_like(a)))

# Clamp
print("clamped:", a.clamp(min=0.2, max=0.8))

# ============================================================
# 4. BROADCASTING
# ============================================================

# Broadcasting rules: dimensions align from right; size-1 dims expand
x = torch.ones(3, 1)                               # (3, 1)
y = torch.ones(1, 4)                               # (1, 4)
print("Broadcast result shape:", (x + y).shape)    # (3, 4)

a = torch.rand(5, 3, 1)
b = torch.rand(   3, 4)
print("Broadcast:", (a + b).shape)                 # (5, 3, 4)

# ============================================================
# 5. RESHAPING
# ============================================================

x = torch.arange(12, dtype=torch.float32)
print("Original:", x.shape)                        # (12,)
print("Reshaped:", x.reshape(3, 4).shape)           # (3, 4)
print("View:", x.view(2, 6).shape)                  # (2, 6) - shares memory
print("Unsqueeze:", x.unsqueeze(0).shape)           # (1, 12)
print("Squeeze:", x.unsqueeze(0).squeeze(0).shape)  # (12,)

# Transpose & Permute
t = torch.rand(2, 3, 4)
print("Permute:", t.permute(2, 0, 1).shape)         # (4, 2, 3)
print("Transpose:", t.transpose(0, 1).shape)        # (3, 2, 4)

# Concatenate & Stack
a = torch.ones(2, 3)
b = torch.zeros(2, 3)
print("Cat dim0:", torch.cat([a, b], dim=0).shape)  # (4, 3)
print("Cat dim1:", torch.cat([a, b], dim=1).shape)  # (2, 6)
print("Stack:", torch.stack([a, b], dim=0).shape)   # (2, 2, 3)

# Split & Chunk
chunks = torch.chunk(torch.rand(12), 4)             # 4 equal chunks
splits = torch.split(torch.rand(12), 3)             # size-3 splits

# ============================================================
# 6. AUTOGRAD & COMPUTATIONAL GRAPHS
# ============================================================

# requires_grad enables gradient tracking
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 2 * x
y.backward()
print("\ndy/dx at x=2:", x.grad)                    # 3x^2 + 2 = 14

# Multi-variable gradient
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print("Gradient:", x.grad)                          # 2x = [2, 4, 6]

# Stopping gradient tracking
x = torch.rand(3, requires_grad=True)
with torch.no_grad():
    y = x * 2                                       # no grad computation
print("y.requires_grad:", y.requires_grad)          # False

# Detach from graph
z = x.detach()                                      # tensor without grad

# retain_graph for multiple backward passes
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward(retain_graph=True)
print("First grad:", x.grad)
x.grad.zero_()                                      # reset gradient
y.backward()
print("Second grad:", x.grad)

# ============================================================
# 7. nn.Module - BUILDING MODELS
# ============================================================

# --- Simple Linear Model ---
class LinearModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

model = LinearModel(4, 2)
x = torch.rand(8, 4)
out = model(x)
print("\nLinear model output:", out.shape)           # (8, 2)

# --- Multi-Layer Perceptron ---
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

mlp = MLP(10, 64, 3)
x = torch.rand(16, 10)
print("MLP output:", mlp(x).shape)                  # (16, 3)

# Inspecting model
print("\nModel parameters:")
for name, param in mlp.named_parameters():
    print(f"  {name}: {param.shape}")

total_params = sum(p.numel() for p in mlp.parameters())
print(f"Total params: {total_params}")

# ============================================================
# 8. COMMON LAYERS
# ============================================================

# Linear
linear = nn.Linear(in_features=10, out_features=5, bias=True)

# Convolutions
conv1d = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
conv2d = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
conv3d = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3)

# Transposed Convolution (upsampling)
convT = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)

# Pooling
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
avgpool = nn.AvgPool2d(kernel_size=2)
adaptpool = nn.AdaptiveAvgPool2d((1, 1))            # global average pool

# Normalization
bn1d = nn.BatchNorm1d(num_features=64)
bn2d = nn.BatchNorm2d(num_features=32)
ln = nn.LayerNorm(normalized_shape=64)
gn = nn.GroupNorm(num_groups=8, num_channels=64)

# Dropout
dropout = nn.Dropout(p=0.5)
dropout2d = nn.Dropout2d(p=0.5)

# Recurrent
rnn = nn.RNN(input_size=10, hidden_size=32, num_layers=2, batch_first=True)
lstm = nn.LSTM(input_size=10, hidden_size=32, num_layers=2, batch_first=True)
gru = nn.GRU(input_size=10, hidden_size=32, num_layers=2, batch_first=True)

# Embedding
embedding = nn.Embedding(num_embeddings=1000, embedding_dim=64)

# Activation functions
relu = nn.ReLU()
leaky = nn.LeakyReLU(negative_slope=0.01)
gelu = nn.GELU()
silu = nn.SiLU()
tanh = nn.Tanh()
sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=-1)

# ============================================================
# 9. LOSS FUNCTIONS
# ============================================================

# Classification
ce_loss = nn.CrossEntropyLoss()
bce_loss = nn.BCELoss()
bce_logits = nn.BCEWithLogitsLoss()                 # numerically stable
nll_loss = nn.NLLLoss()

# Regression
mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
huber_loss = nn.HuberLoss(delta=1.0)
smooth_l1 = nn.SmoothL1Loss()

# Usage example
logits = torch.rand(8, 3)
targets = torch.randint(0, 3, (8,))
loss = ce_loss(logits, targets)
print(f"\nCross-entropy loss: {loss.item():.4f}")

pred = torch.rand(8, 1)
true = torch.rand(8, 1)
loss_mse = mse_loss(pred, true)
print(f"MSE loss: {loss_mse.item():.4f}")

# ============================================================
# 10. OPTIMIZERS
# ============================================================

model = MLP(10, 64, 3)

sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
adam = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
adamw = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
rmsprop = optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99)
adagrad = optim.Adagrad(model.parameters(), lr=0.01)

# Per-layer learning rates
optimizer = optim.Adam([
    {"params": model.net[0].parameters(), "lr": 1e-4},
    {"params": model.net[3].parameters(), "lr": 1e-3},
], lr=1e-3)

# ============================================================
# 11. LEARNING RATE SCHEDULERS
# ============================================================

optimizer = optim.Adam(model.parameters(), lr=1e-3)

step_lr    = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
multi_step = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
cosine_lr  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
exp_lr     = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
plateau    = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
onecycle   = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, total_steps=100)
warmup     = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)

# ============================================================
# 12. DATASET & DATALOADER
# ============================================================

# Custom Dataset
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Generate dummy data
X_data = np.random.randn(200, 10).astype(np.float32)
y_data = np.random.randint(0, 3, 200)

dataset = TabularDataset(X_data, y_data)

# Train/val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

# DataLoader
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)

print(f"\nDataset size: {len(dataset)}")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# ============================================================
# 13. FULL TRAINING LOOP
# ============================================================

model = MLP(10, 64, 3)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        # Forward
        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (optional)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        # Metrics
        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

    return total_loss / total, correct / total


print("\n=== Training ===")
num_epochs = 10
best_val_loss = float("inf")

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    scheduler.step()

    print(f"Epoch {epoch:2d} | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f} | "
          f"LR: {scheduler.get_last_lr()[0]:.6f}")

    # Checkpoint best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "/tmp/best_model.pt")

# ============================================================
# 14. SAVING & LOADING MODELS
# ============================================================

# Save state dict (recommended)
torch.save(model.state_dict(), "/tmp/model_weights.pt")

# Load state dict
loaded_model = MLP(10, 64, 3)
loaded_model.load_state_dict(torch.load("/tmp/model_weights.pt", map_location="cpu"))
loaded_model.eval()
print("\nModel loaded successfully!")

# Save entire model (not recommended for deployment)
torch.save(model, "/tmp/full_model.pt")

# Save checkpoint (training state)
checkpoint = {
    "epoch": num_epochs,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "val_loss": best_val_loss,
}
torch.save(checkpoint, "/tmp/checkpoint.pt")

# Load checkpoint
ckpt = torch.load("/tmp/checkpoint.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
optimizer.load_state_dict(ckpt["optimizer_state_dict"])
start_epoch = ckpt["epoch"]
print(f"Resumed from epoch {start_epoch}")

# ============================================================
# 15. CUSTOM autograd.Function
# ============================================================

class MySigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        s = 1 / (1 + torch.exp(-x))
        ctx.save_for_backward(s)
        return s

    @staticmethod
    def backward(ctx, grad_output):
        s, = ctx.saved_tensors
        return grad_output * s * (1 - s)

x = torch.randn(4, requires_grad=True)
y = MySigmoid.apply(x)
y.sum().backward()
print("\nCustom sigmoid grad:", x.grad)

# ============================================================
# 16. CNN EXAMPLE
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # (B, 32, H, W)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # (B, 32, H/2, W/2)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # (B, 64, H/4, W/4)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),                  # (B, 64, 1, 1)
            nn.Flatten(),                                  # (B, 64)
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

cnn = SimpleCNN(num_classes=10)
x = torch.rand(4, 1, 28, 28)                              # batch of grayscale 28x28
print("\nCNN output:", cnn(x).shape)                      # (4, 10)

# ============================================================
# 17. RNN / LSTM EXAMPLE
# ============================================================

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (hn, cn) = self.lstm(x)
        # Use last hidden state
        return self.fc(hn[-1])

lstm_model = LSTMClassifier(10, 64, 2, 5)
x = torch.rand(8, 20, 10)                                 # (batch, seq_len, features)
print("LSTM output:", lstm_model(x).shape)                # (8, 5)

# ============================================================
# 18. INFERENCE MODE & PREDICTION
# ============================================================

model.eval()
sample = torch.rand(1, 10)

# Method 1: no_grad
with torch.no_grad():
    output = model(sample)
    probs = F.softmax(output, dim=1)
    pred_class = probs.argmax(dim=1)
    confidence = probs.max(dim=1).values
    print(f"\nPredicted class: {pred_class.item()}, confidence: {confidence.item():.4f}")

# Method 2: inference_mode (faster, more restrictive)
with torch.inference_mode():
    output = model(sample)
    print("Inference mode output shape:", output.shape)

# ============================================================
# 19. DEVICE MANAGEMENT
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# Move model to device
model = model.to(device)

# Move data to device
X = torch.rand(4, 10).to(device)
y_pred = model(X)                                         # auto runs on device

# Check device
print("Model device:", next(model.parameters()).device)

# GPU memory info
if torch.cuda.is_available():
    print("VRAM allocated:", torch.cuda.memory_allocated() / 1e6, "MB")
    print("VRAM cached:", torch.cuda.memory_reserved() / 1e6, "MB")
    torch.cuda.empty_cache()

# ============================================================
# 20. USEFUL UTILITIES
# ============================================================

# Reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Count parameters
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

total, trainable = count_params(mlp)
print(f"\nTotal params: {total:,} | Trainable: {trainable:,}")

# Freeze layers
for param in mlp.net[0].parameters():
    param.requires_grad = False

# Gradient norm inspection
x = torch.rand(4, 10, requires_grad=True)
y = mlp(x).sum()
y.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
print(f"Gradient norm: {grad_norm:.4f}")

# Weight initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

cnn.apply(init_weights)
print("Weights initialized!")

print("\n=== ALL PYTORCH FOUNDATIONS CODE COMPLETE ===")
