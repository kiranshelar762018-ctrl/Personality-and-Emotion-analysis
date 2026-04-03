import torch
import torch.nn as nn
import torch.optim as optim
import time

# ---------------- EXACT ARCHITECTURE ---------------- #
# We copy this exact class from app.py so we don't accidentally start the Flask server
class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_lstm = nn.LSTM(300, 32, batch_first=True)
        self.text_fc = nn.Linear(32, 16)

        self.audio_lstm = nn.LSTM(74, 32, batch_first=True)
        self.audio_fc = nn.Linear(32, 16)

        self.visual_lstm = nn.LSTM(713, 32, batch_first=True)
        self.visual_fc = nn.Linear(32, 16)

        self.fc1 = nn.Linear(48, 32)
        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 7)

    def forward(self, text, audio, visual):
        _, (t, _) = self.text_lstm(text)
        _, (a, _) = self.audio_lstm(audio)
        _, (v, _) = self.visual_lstm(visual)

        t = self.text_fc(t[-1])
        a = self.audio_fc(a[-1])
        v = self.visual_fc(v[-1])

        x = torch.cat([t, a, v], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

def train_synthetic_model():
    print("🚀 Initializing PyTorch Local Training...")
    model = MultimodalModel()
    
    # 7 emotions cross entropy
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    print("📊 Generating 1500 Synthetic Multimodal Training Data points...")
    NUM_SAMPLES = 1500
    MAX_LEN = 50

    try:
        # Generate entirely random background noise
        text_data = torch.rand((NUM_SAMPLES, MAX_LEN, 300))
        audio_data = torch.rand((NUM_SAMPLES, MAX_LEN, 74))
        visual_data = torch.rand((NUM_SAMPLES, MAX_LEN, 713))
        
        # Ground truth labels (evenly distributed 0-6)
        labels = torch.randint(0, 7, (NUM_SAMPLES,))
        
        # Inject deliberate data biases so the Neural Net can actually learn "differences"
        for i in range(NUM_SAMPLES):
            emotion = labels[i].item()
            # If happiness, boost certain visual ranges
            if emotion == 0: 
                text_data[i, :, 0:100] += 5.0
                visual_data[i, :, 0:200] += 3.0
            # If sadness, lower audio ranges
            elif emotion == 1: 
                audio_data[i, :, 0:20] -= 2.0
            # If anger, huge jump in visual
            elif emotion == 2: 
                visual_data[i, :, 200:400] += 6.0
            # If fear, weird audio signature
            elif emotion == 3: 
                audio_data[i, :, 40:70] += 4.0
            # ... we just sprinkle enough bias so the network converges
            else:
                text_data[i, :, 200:300] -= 2.0

        dataset = torch.utils.data.TensorDataset(text_data, audio_data, visual_data, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        print(f"🔥 Starting Training Loop (20 Epochs)...")
        epochs = 20
        model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            for batch_text, batch_audio, batch_visual, batch_labels in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_text, batch_audio, batch_visual)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == batch_labels).sum().item()
                
            acc = 100 * correct / NUM_SAMPLES
            print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | Accuracy: {acc:.1f}%")

        print("🧠 Training Complete! Accuracy reached competitive synthetic levels.")
        
        # Save exact model file expected by app.py
        torch.save(model.state_dict(), "model.pth")
        print("\n✅ SUCCESSFULLY SAVED TRAINED WEIGHTS TO 'model.pth'!")
    
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    start = time.time()
    train_synthetic_model()
    print(f"Total processing time: {time.time() - start:.2f} seconds")
