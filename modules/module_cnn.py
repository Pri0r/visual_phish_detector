import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from utils import EarlyStopping

# use square transformation if true
SQUARE_CUT = True

class CropToSquareTop:
    """Transforms image into a square"""
    def __call__(self, img):
        w, h = img.size
        return img.crop((0, 0, w, min(h, w)))

def get_transforms() -> transforms.Compose:
    """Returns either a square or squash transform function"""
    if SQUARE_CUT:
        return transforms.Compose([
            CropToSquareTop(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])


class ScreenshotDataset(Dataset):
    """Represents a dataset of website screenshots with corresponding file paths"""
    def __init__(self, metadata_path, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform
        self.data =[]
        
        # get file paths from metadata json file
        with open(metadata_path, "r", encoding="utf-8") as f:
            raw_metadata = json.load(f)
        
        # get labels and check if the files exist
        for item in raw_metadata:
            rel_path = item.get("data_path")
            img_path = os.path.join(root_dir, rel_path, "screenshot.png")
            
            if os.path.exists(img_path):
                label = 1.0 if item.get("phishing") else 0.0
                self.data.append((img_path, label))
            else:
                print(f"Warning: path '{rel_path}' does not contain a valid screenshot!")

    def __len__(self):
        """Returns dataset size"""
        return len(self.data)

    def __getitem__(self, index):
        """Returns a single image along with its label"""
        img_path, label = self.data[index]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error: couldn't load {img_path}: {e}")
            # send black image instead
            return torch.zeros((3, 224, 224)), torch.tensor(label, dtype=torch.float32)


def build_model(use_resnet = True, deafult_weights = True):
    """Returns either a resnet50 or efficientnet-b4 model"""
    if use_resnet:
        weights = models.ResNet50_Weights.DEFAULT if deafult_weights else None
        model = models.resnet50(weights=weights)
        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
    else:
        weights = models.EfficientNet_B4_Weights.DEFAULT if deafult_weights else None
        model = models.efficientnet_b4(weights=weights)
        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 1)

    return model


class CNNClassifier:
    """Screenshot phishing classifier"""
    def __init__(self, model_path, use_resnet = True, device = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{self.__class__.__name__}] Loading model to {self.device}...")
        
        self.model = build_model(use_resnet, deafult_weights=False).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = get_transforms()

    def evaluate_single(self, image_path):
        """Classify a single screenshot"""
        if not os.path.exists(image_path):
            return {"status": "error", "reason": "missing_file", "is_phishing": False, "probability": 0.0}

        try:
            image = Image.open(image_path).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # inference
            with torch.no_grad():
                output = self.model(tensor)
                prob = torch.sigmoid(output.view(-1)).item()
                
            # return both score and a simple decision with 0.5 threshold
            return {
                "status": "success",
                "is_phishing": prob >= 0.5,
                "probability": prob
            }
        except Exception as e:
            return {"status": "error", "reason": str(e), "is_phishing": False, "probability": 0.0}


def evaluate_loader(model, loader, device, criterion = None):
    """Evaluates model given a dataloader, returns metrics and loss"""
    model.eval()
    all_preds, all_labels = [], []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            if criterion:
                loss = criterion(outputs.view(-1), labels)
                running_loss += loss.item() * images.size(0)

            preds = torch.round(torch.sigmoid(outputs.view(-1)))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    val_loss = running_loss / len(loader.dataset) if criterion and len(loader.dataset) > 0 else 0.0
    
    return acc, prec, rec, f1, val_loss


def train_cnn_model(metadata_path, root_dir, save_path= "cnn_model.pth", use_resnet = True, batch_size = 32, lr = 0.001, epochs = 50, patience = 10):
    """Train new CNN model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    full_dataset = ScreenshotDataset(metadata_path, root_dir, transform=get_transforms())
    if len(full_dataset) == 0:
        print("Error: Dataset is empty!")
        return
    
    # split 85:15 (training:validation)
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    model = build_model(use_resnet, deafult_weights=True).to(device)

    # calculase criterion weight in case of inbalanced training set
    num_pos = 0
    num_neg = 0
    for idx in train_dataset.indices:
        _, label = full_dataset.data[idx]
        if label == 1.0:
            num_pos += 1
        else:
            num_neg += 1
            
    pos_weight_val = (num_neg / num_pos) if num_pos > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device) 

    # initialize optimizer and loss criterion
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # start training
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=save_path)
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        train_loss = running_loss / len(train_loader.dataset)
        acc, prec, rec, val_f1, val_loss = evaluate_loader(model, val_loader, str(device), criterion)
        
        print(f"Epoch {epoch+1}/{epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stop triggered!")
            break

    print(f"Training finished, model saved to: {save_path}")
