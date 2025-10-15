import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import flwr as fl
from opacus import PrivacyEngine


df = pd.read_csv('TON_IoT_Train_Test_Network.csv')

for col in df.columns:
    null_pct = df[col].isnull().sum() / len(df) * 100
    if null_pct > 50:
        df = df.drop(col, axis=1)

label_col = 'type'
if label_col not in df.columns:
    raise ValueError(f"Label column '{label_col}' not found")

y = df[label_col]
X_df = df.drop(label_col, axis=1)

numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()

if non_numeric_cols:
    for col in non_numeric_cols:
        try:
            X_df[col] = X_df[col].fillna('MISSING')
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str))
        except:
            X_df = X_df.drop(col, axis=1)

if numeric_cols:
    for col in numeric_cols:
        if X_df[col].isnull().any():
            median_val = X_df[col].median()
            X_df[col] = X_df[col].fillna(median_val)

numeric_cols_present = X_df.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols_present:
    scaler = StandardScaler()
    X_df[numeric_cols_present] = scaler.fit_transform(X_df[numeric_cols_present])

X = X_df.values

if X.dtype == 'object':
    try:
        X = X.astype(np.float32)
    except Exception as e:
        for i, col in enumerate(X_df.columns):
            try:
                X_df[col].astype(np.float32)
            except:
                pass
        raise ValueError("Data contains non-numeric values")

le_label = LabelEncoder()
y = le_label.fit_transform(y.astype(str))
num_classes = len(le_label.classes_)

num_clients = 10
client_data = []
unique_labels = np.unique(y)
np.random.seed(42)

for i in range(num_clients):
    num_labels = min(np.random.randint(2, 5), len(unique_labels))
    selected_labels = np.random.choice(unique_labels, size=num_labels, replace=False)
    mask = np.isin(y, selected_labels)
    client_X = X[mask]
    client_y = y[mask]
    
    if len(client_y) > 5000:
        sample_size = min(5000, len(client_y))
        indices = np.random.choice(len(client_y), size=sample_size, replace=False)
        client_X = client_X[indices]
        client_y = client_y[indices]
    
    client_data.append((client_X.astype(np.float32), client_y.astype(np.int64)))

for i, (X_client, y_client) in enumerate(client_data):
    if X_client.dtype not in [np.float32, np.float64]:
        raise ValueError(f"Client {i} data is not numeric")
    if y_client.dtype not in [np.int32, np.int64]:
        raise ValueError(f"Client {i} labels are not numeric")

with open('client_data.pkl', 'wb') as f:
    pickle.dump(client_data, f)
with open('num_classes.pkl', 'wb') as f:
    pickle.dump(num_classes, f)


class IntrusionDetectionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(IntrusionDetectionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class NetworkDataset(Dataset):
    def __init__(self, X, y):
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float32)
        if not isinstance(y, np.ndarray):
            y = np.array(y, dtype=np.int64)
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


with open('client_data.pkl', 'rb') as f:
    client_data = pickle.load(f)
with open('num_classes.pkl', 'rb') as f:
    num_classes = pickle.load(f)

input_size = client_data[0][0].shape[1]


class FederatedClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = int(cid)
        X, y = client_data[self.cid]
        dataset = NetworkDataset(X, y)
        self.trainloader = DataLoader(dataset, batch_size=32, shuffle=True)
        self.testloader = DataLoader(dataset, batch_size=32, shuffle=False)
        self.model = IntrusionDetectionModel(input_size, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        privacy_engine = PrivacyEngine()
        self.model, optimizer, train_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=optimizer,
            data_loader=self.trainloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0
        )
        self.model.train()
        for epoch in range(2):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        loss /= total
        accuracy = correct / total
        return loss, total, {"accuracy": accuracy}


strategy = fl.server.strategy.FedAvg(
    min_available_clients=10,
    min_fit_clients=5,
    min_evaluate_clients=2,
    fraction_fit=0.5,
    fraction_evaluate=0.2,
)

client_resources = {
    "num_cpus": 1,
    "num_gpus": 0.05 if torch.cuda.is_available() else 0
}

fl.simulation.start_simulation(
    client_fn=lambda cid: FederatedClient(cid),
    num_clients=10,
    config=fl.server.ServerConfig(num_rounds=20),
    strategy=strategy,
    client_resources=client_resources
)
