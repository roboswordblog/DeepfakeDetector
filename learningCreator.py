import pandas as pd
import torch.nn as nn
import torch
import pathlib
import torch.nn.functional as F

def PositionalEmbedding(seq_len, emb_size):
    embeddings = torch.ones(seq_len, emb_size)
    for i in range(seq_lean):
        for j in range(emb_size):
            embeddings[i][j] = np.sin(i / (pow(10000, j/emb_size)) if j % 2 == 0 else np.cos(i / pow(10000, (j-1) / emb_size)))
    return torch.tensor(embeddings)

class PatchEmbeddings(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int=16, emb_size:int = 768, img_size=224):
        self.patch_size = patch_size
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) -> b (h w) e')
        )
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_size))
        self.pos_embed = nn.Parameter(PositionalEmbedding((img_size // patch_size)**2 + 1, emb_size))
    
    def  forward(self, x:Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.embed(x)
        return x


class MultiHead(nn.Module):
    def __init__(self, emb_size, num_head):
        super().__init__()
        self.emb_size = emb_size
        self.num_head = num_head
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size, emb_size)
        self.query =  nn.Linear(emb_size, emb_size)
        self.att_dr = nn.Dropout(0.1)

    def forward(self, x):
        k = rearrange(self.key(x), 'b n (h e) - > b h n e', h=self.num_head)
        q = rearrange(self.query(x), 'b n (h e) - > b h n e', h=self.num_head)       
        v = rearrange(self.value(x), 'b n (h e) - > b h n e', h=self.num_head)
        wei = q@k.transpose(3,2)/self.head_dim ** 0.5
        wei = F.softmax(wei, dim=2)
        wei = self.att_dr(wei)
        out = wei@v
        out = rearrange(out, 'b h n e -> b n (h e)')
        return out

class FeedForward(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(emb_size, 4*emb_size),
            nn.Linear(emb_size, 4*emb_size)
        )

    def forward(self, x):
        return self.ff(x)

class Block(nn.Module):
    def __init__(self, emb_size, num_head):
        super().__init__()
        self.att = MultiHead(emb_size, num_head)
        self.ll = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(0.1)
        self.ff = FeedForward(emb_size)

    def forward(self, x):
        x = x + self.dropout(self.att(self.ll(x)))
        x = x +  self.dropout(self.ff(self.ll(x)))
        return x
        
class VisionTransformer(nn.Module):
    def __init__(self, num_layerss, img_size, emb_size, patch_size, num_head, num_class):
        super().__init__()
        self.attention = nn.Sequential(*[Block (emb_size, num_head) for _ in range(num_layers)])
        self.patchemb = PatchEmbedding(patch_size=patch_size, img_size=img_size)
        self.ff = nn.Linear(emb_size, num_class)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_layers = 8
emb_size = 768
num_head 
num_class = 10
patch_size = 16
model = VisionTransformer( num_layers = num_layers,
                            img_size = 224,
                            emb_size = emb_size
                            num_head= num_head,
                            patch_size = patch_size,
                            num_class = num_class).to(device)

transform = transforms.Compose([
    transforms.Resiize((224, 224)),
    transforms.ToTensor(),
    transforms.Normaliize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_set = torchvisoin.datasets.CIFAR10(root='./data' train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workesr=2)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

citerion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

model.train()
epochs = 20
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (input, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterioon(outputs, targets)
        loss.backward()
        optimmizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(train_loader)}] | "
                      f"Loss: {running_loss / (batch_idx + 1):.4f} | Acc: {100. * correct / total:.2f}%")
                
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    print(f"==> Epoch {epoch+1} Complete | Avg Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%\n")

model.eval()
test_loss, test_correct, test_total = 0.0, 0, 0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        test_total += targets.size(0)
        test_correct += predicted.eq(targets).sum().item()

final_test_loss = test_loss / len(test_loader)
final_test_acc = 100. * test_correct / test_total
print(f"🏁 Final Test Results | Loss: {final_test_loss:.4f} | Accuracy: {final_test_acc:.2f}%")
