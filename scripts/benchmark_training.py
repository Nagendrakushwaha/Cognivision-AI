import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import pyarrow.parquet as pq
import io
from PIL import Image
import numpy as np

# Label mapping
CLASS_NAMES = [
    "Bakery & Confectionery",
    "General Retail & Services",
    "Hardware & Home",
    "Restaurant & Dining",
    "Stationery & Bookstore",
    "Supermarket & Grocery"
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

def categorize_receipt(company_name, words):
    c = (company_name or '').upper()
    all_words_text = " ".join([str(w).upper() for w in (words if words is not None else [])])
    full_text = f"{c} {all_words_text}"

    if any(k in full_text for k in ["MR. D.I.Y.", "MR D.I.Y", "MR DIY", "HARDWARE", "PAPAN", "TIMBER", "BUILDING", "HOME DECO", "AIK HUAT", "ELECTRICAL"]):
        return "Hardware & Home"
    if any(k in full_text for k in ["GARDENIA", "BAKERY", "BAKERIES", "CONFECTIONERY", "CAKE", "BREAD", "ROTI", "PASTRY"]):
        return "Bakery & Confectionery"
    if any(k in full_text for k in ["SPEED MART", "SPEEDMART", "99 SPEED", "AEON", "GROCER", "SUPERMARKET", "PASARAYA", "HYPERMARKET", "MINI MARKET", "MART"]):
        return "Supermarket & Grocery"
    if any(k in full_text for k in ["RESTORAN", "RESTAURANT", "FOOD", "KOPITIAM", "CAFE", "COFFEE", "SEAFOOD", "UNIHAKKA", "GERBANG ALAF", "STEAMBOAT", "KITCHEN", "DINING", "MCDONALD"]):
        return "Restaurant & Dining"
    if any(k in full_text for k in ["STATIONERY", "BOOK", "POPULAR BOOK", "PAPER", "PEN", "OFFICE SUPPLIES", "PRINTING"]):
        return "Stationery & Bookstore"
    return "General Retail & Services"

class SROIEImageDataset(Dataset):
    def __init__(self, parquet_path, transform=None):
        table = pq.read_table(parquet_path)
        self.df = table.to_pandas()
        self.transform = transform
        self.labels = [
            CLASS_TO_IDX[categorize_receipt(row['entities'].get('company'), row['words'])]
            for idx, row in self.df.iterrows()
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_bytes = row['image']['bytes']
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

def benchmark():
    print("Starting benchmark training test...")
    start_time = time.time()
    
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = SROIEImageDataset("train-00000-of-00001.parquet", transform=transform_train)
    test_dataset = SROIEImageDataset("test-00000-of-00001.parquet", transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Loaded {len(train_dataset)} train samples, {len(test_dataset)} test samples.")

    # Lightweight MobileNetV3-Small
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    # Freeze earlier layers for fast training
    for param in model.features[:-3].parameters():
        param.requires_grad = False
        
    num_classes = len(CLASS_NAMES)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    print("\nTraining for 3 epochs on CPU...")
    for epoch in range(3):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Validation on test
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        epoch_dur = time.time() - epoch_start
        print(f"Epoch [{epoch+1}/3] ({epoch_dur:.2f}s) - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    total_dur = time.time() - start_time
    print(f"\nTotal benchmark duration: {total_dur:.2f}s. Excellent!")

if __name__ == "__main__":
    benchmark()
