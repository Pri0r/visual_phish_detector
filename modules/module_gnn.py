import os
import json
import torch
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from utils import rdf_to_pyg_data, EarlyStopping


class GNNModel(torch.nn.Module):
    """GNN model"""
    def __init__(self, num_node_features):
        super().__init__()
        
        self.conv1 = GATv2Conv(num_node_features, 64, heads=4, concat=True)
        self.conv2 = GATv2Conv(256, 128, heads=1, concat=False)
        self.classifier = torch.nn.Sequential(
            Linear(128 * 2, 128),
            ReLU(),
            Dropout(0.5),
            Linear(128, 64),
            ReLU(),
            Dropout(0.5),
            Linear(64, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        
        global_graph_embedding = torch.cat([x_mean, x_max], dim=1)
        
        return self.classifier(global_graph_embedding)


class GNNClassifier:
    """RDF graph phishing classifier"""
    def __init__(self, model_path, num_node_features, device = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{self.__class__.__name__}] Loading model to {self.device}...")
        
        self.model = GNNModel(num_node_features=num_node_features).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device)) 
        self.model.eval()

    def evaluate_single(self, graph_input):
        """Classify a single RDF graph"""
        try:
            if isinstance(graph_input, str):
                # file does not exist
                if not os.path.exists(graph_input):
                    return {"status": "error", "reason": "missing_file", "is_phishing": False, "probability": 0.0}
                
                # convert rdf graph to pyg format
                if graph_input.endswith(".ttl"):
                    data = rdf_to_pyg_data(graph_input, label = -1)
                # load already precomputed data
                elif graph_input.endswith(".pt"):
                    data = torch.load(graph_input, map_location=self.device, weights_only=False)
                else:
                    return {"status": "error", "reason": "wrong_format", "is_phishing": False, "probability": 0.0}
            else:
                data = graph_input
    
            data = data.to(self.device)
            # check if graph is valid
            if not hasattr(data, "x") or data.num_nodes == 0:
                return {"status": "error", "reason": "empty_graph", "is_phishing": False, "probability": 0.0}
    
            # create zero batch vector if needed
            if not hasattr(data, "batch") or data.batch is None:
                data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
    
            # inference
            with torch.no_grad():
                out = self.model(data)
                prob = torch.sigmoid(out).item()
    
            return {
                "status": "success",
                "is_phishing": prob >= 0.5,
                "probability": prob
            }
        except Exception as e:
            return {"status": "error", "reason": str(e), "is_phishing": False, "probability": 0.0}


def train_gnn_model(metadata_path, root_dir, save_path = "best_gnn_model.pt", epochs = 100, batch_size = 64, lr = 0.0005, patience = 10):
    """Train new GNN model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}...")

    def train_epoch(model, loader, optimizer, criterion, device):
        """Single train iteration"""
        model.train()
        total_loss = 0
        all_preds, all_labels = [],[]
        
        for data in tqdm(loader, desc="Training", leave=False):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            
            loss = criterion(out, data.y.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * data.num_graphs
            
            preds = (torch.sigmoid(out) > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(data.y.unsqueeze(1).cpu())

        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        train_f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        return total_loss / len(loader.dataset), train_f1
    
    @torch.no_grad()
    def test_epoch(model, loader, device, criterion = None):
        """Single test iteration"""
        model.eval()
        total_loss = 0
        all_preds, all_labels = [],[]
        
        for data in loader:
            data = data.to(device)
            out = model(data)
            
            if criterion:
                loss = criterion(out, data.y.unsqueeze(1).float())
                total_loss += loss.item() * data.num_graphs

            preds = (torch.sigmoid(out) > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(data.y.unsqueeze(1).cpu())
            
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        avg_loss = total_loss / len(loader.dataset) if criterion else 0
        
        return avg_loss, {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    # load data
    dataset = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
  
    for item in tqdm(metadata, desc="Loading .pt files"):
        path = os.path.join(root_dir, item["data_path"], "site.pt")
        label = 1 if item.get("phishing", False) else 0

        if os.path.exists(path):
            graph_data = torch.load(path, weights_only=False)
            if not graph_data or graph_data.num_nodes == 0:
                print(f"Warning: Empty graph data for path: {path}")
                continue
            if graph_data.y.item() != label:
                print(f"Warning: different labels for path: {path}")
            dataset.append(graph_data)
        else:
            print(f"Warning: wrong path: {path}")

    if not dataset:
        raise ValueError("Dataset is empty!")

    #split 85:15 (training:validation)
    total_len = len(dataset)
    train_idx = int(0.85 * total_len)
    train_loader = DataLoader(dataset[:train_idx], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset[train_idx:], batch_size=batch_size, shuffle=False)
    num_features = dataset[0].num_node_features
    model = GNNModel(num_node_features=num_features).to(device)

    # calculase criterion weight in case of inbalanced training set
    num_pos = sum([data.y.item() == 1 for data in dataset[:train_idx]])
    num_neg = sum([data.y.item() == 0 for data in dataset[:train_idx]])
    pos_weight_val = (num_neg / num_pos) if num_pos > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val]).to(device)
    
    # initialize optimizer and loss criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # start training
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=save_path)
    for epoch in range(epochs):
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = test_epoch(model, val_loader, device, criterion)
        val_f1 = val_metrics["f1"]

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stop triggered!")
            break

    print(f"Training finished, model saved to: {save_path}")
