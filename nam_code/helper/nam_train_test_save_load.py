import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os


############################## first version of NAM model ##############################
# Define Activation Layers
class ReLULayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(in_features))

    def forward(self, x):
        return F.relu((x - self.bias) @ self.weight)

class ExU(nn.Module):
    def __init__(self):
        super(ExU, self).__init__()
        # Just one scalar weight + bias per feature
        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Ensure x is 2D: (batch_size, 1)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        out = torch.exp(x * self.weight + self.bias) - 1
        return F.relu(out)


class ExuFeatureNN(nn.Module):
    def __init__(self, hidden_dim=1024):
        super(ExuFeatureNN, self).__init__()
        self.exu = ExU()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # final scalar output

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)   # (batch, 1)
        x = self.exu(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)       # (batch, 1)

# Define Feature-wise Neural Networks
class FeatureNN(nn.Module):
    def __init__(self, shallow_units, hidden_units=[]):
        super().__init__()
        layers = [ReLULayer(1, shallow_units)]
        for h in hidden_units:
            layers.append(ReLULayer(shallow_units, h))
        self.layers = nn.ModuleList(layers)
        self.linear = nn.Linear(shallow_units if not hidden_units else hidden_units[-1], 1, bias=False)

    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        return self.linear(x)

# Define the NAM Model
class NeuralAdditiveModel(nn.Module):
    def __init__(self, input_size, shallow_units=10, with_exu=True, hidden_units=[]):
        super().__init__()
        self.input_size = input_size
        self.with_exu = with_exu
        if with_exu:
            self.feature_nns = nn.ModuleList([
                ExuFeatureNN(1024) for _ in range(input_size)
            ])
        else:
            self.feature_nns = nn.ModuleList([
                FeatureNN(shallow_units, hidden_units) for _ in range(input_size)
            ])
        self.feature_weights = nn.Parameter(torch.ones(input_size))  # Trainable feature weights
        self.bias = nn.Parameter(torch.zeros(1))

    # def forward(self, x):
    #     feature_outputs = torch.stack([self.feature_nns[i](x[:, i]) for i in range(self.input_size)], dim=-1)
    #     weighted_sum = (feature_outputs * self.feature_weights).sum(dim=-1)
    #     return weighted_sum + self.bias
    
    def forward(self, x):
        # Ensure x is 2D: (batch_size, input_size)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        if self.with_exu:
            feature_outputs = torch.cat(
                [self.feature_nns[i](x[:, i]) for i in range(self.input_size)],
                dim=1   # shape: (batch, input_size)
            )
            weighted_sum = (feature_outputs * self.feature_weights).sum(dim=1)  # (batch,)
            return weighted_sum + self.bias
            # Compute outputs for each feature using the corresponding NN
        else:
            feature_outputs = torch.stack([self.feature_nns[i](x[:, i]) for i in range(self.input_size)], dim=-1)
            # Weighted sum over features + bias
            weighted_sum = (feature_outputs * self.feature_weights).sum(dim=-1)
            return weighted_sum + self.bias



# Define Feature-wise Neural Networks
class FeatureNNBigger(nn.Module):
    def __init__(self, hidden_units=[64, 64, 32]):
        super().__init__()
        layers = []
        in_features = 1
        for h in hidden_units:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            in_features = h
        self.network = nn.Sequential(*layers)
        self.output = nn.Linear(in_features, 1, bias=False)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.network(x)
        return self.output(x)
    

############################## second version of NAM model ##############################
# Define the NAM Model
class NeuralAdditiveModelBigger(nn.Module):
    def __init__(self, input_size, hidden_units=[64, 64, 32]):
        super().__init__()
        self.input_size = input_size
        self.feature_nns = nn.ModuleList([
            FeatureNNBigger(hidden_units) for _ in range(input_size)
        ])
        self.feature_weights = nn.Parameter(torch.ones(input_size))  # Trainable feature weights
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        feature_outputs = torch.stack([self.feature_nns[i](x[:, i]) for i in range(self.input_size)], dim=-1)
        weighted_sum = (feature_outputs * self.feature_weights).sum(dim=-1)
        return weighted_sum + self.bias


# Load and preprocess the dataset
def load_data(dataset="breast_cancer", batch_size=32):
    
    if dataset == "connectionist_bench":
        from ucimlrepo import fetch_ucirepo   
        # fetch dataset 
        connectionist_bench = fetch_ucirepo(id=151) 
        # data (as pandas dataframes) 
        X = connectionist_bench.data.features 
        y = connectionist_bench.data.targets 
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        y_train = np.array(y_train["class"].to_list())
        # replace R with 1, M with 0
        y_train = np.where(y_train == 'R', 1, np.where(y_train == 'M', 0, 0))
        y_test = np.array(y_test["class"].to_list())
        y_test = np.where(y_test == 'R', 1, np.where(y_test == 'M', 0, 0))
    elif dataset == "spambase":
        from ucimlrepo import fetch_ucirepo   
        
        # fetch dataset 
        spambase = fetch_ucirepo(id=94) 
        
        # data (as pandas dataframes) 
        X = spambase.data.features 
        y = spambase.data.targets 
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        y_train = np.array(y_train["Class"].to_list())
        y_test = np.array(y_test["Class"].to_list())
    elif dataset == "cal_housing":
        df = pd.read_csv('data/cal_housing.data', header=None)
        df.columns = [
            'longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income',
            'median_house_value'
        ]
        df['total_bedrooms'].fillna(df['total_bedrooms'].mean(), inplace=True)
        # df['total_bedrooms'].fillna(0, inplace=True)
        X = df.drop(['median_house_value'],axis=1)
        y = df['median_house_value']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        y_train = np.array(y_train.to_list())
        y_test = np.array(y_test.to_list())
    elif dataset == "credit":
        data = pd.read_csv('data/creditcard.csv')
        fraud = data[data['Class'] == 1]
        legit = data[data.Class == 0]  # 284315 samples
        fraud = data[data.Class == 1]  # only 492 samples
        legit_sample = legit.sample(n=492)
        new_dataset = pd.concat([legit_sample, fraud], axis=0)
        X = new_dataset.drop(columns='Class', axis=1)
        Y = new_dataset['Class']
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, stratify=Y, random_state=42
        )
        y_train = np.array(y_train.to_list())
        y_test = np.array(y_test.to_list())
    elif dataset == "heloc":
        heloc = pd.read_csv('data/heloc_dataset.csv')
        X = heloc.drop(columns = 'RiskPerformance')
        y = heloc.RiskPerformance.replace(to_replace=['Bad', 'Good'], value=[1, 0])
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
    # from aix360.datasets.heloc_dataset import HELOCDataset, nan_preprocessing
    # data = HELOCDataset(custom_preprocessing=nan_preprocessing).data()
    elif dataset == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, X_train.shape[1]

# Train function
def train_model(model, train_loader, device, epochs=10):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Test function
def test_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = torch.sigmoid(model(X_batch).squeeze())
            predictions = (y_pred > 0.5).float()
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

# Save the entire model
def save_full_model(model, path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Full model saved at {path}")

# Load the entire model
def load_full_model(path, input_size, device, is_bigger_nam=False):
    cls = NeuralAdditiveModelBigger if is_bigger_nam else NeuralAdditiveModel
    model = cls(input_size).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Full model loaded from {path}")
    return model

# Save each partial model
def save_partial_models(model, path):
    os.makedirs(path, exist_ok=True)
    for i, feature_nn in enumerate(model.feature_nns):
        torch.save(feature_nn.state_dict(), os.path.join(path, f"feature_{i}.pth"))
    torch.save(model.feature_weights, os.path.join(path, "feature_weights.pth"))
    torch.save(model.bias, os.path.join(path, "bias.pth"))
    print("Partial models and weights saved.")

# Load each partial model
def load_partial_models(model, path, device):
    for i, feature_nn in enumerate(model.feature_nns):
        feature_nn.load_state_dict(torch.load(os.path.join(path, f"feature_{i}.pth"), map_location=device))
    model.feature_weights.data = torch.load(os.path.join(path, "feature_weights.pth"), map_location=device)
    model.bias.data = torch.load(os.path.join(path, "bias.pth"), map_location=device)
    print("Partial models and weights loaded.")


# Main execution
if __name__ == "__main__":
    # seed everything
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_bigger_nam = True

    # Load dataset
    # dataset = "breast_cancer"
    # dataset = "heloc"
    # dataset = "credit"
    # dataset = "spambase"
    dataset = "connectionist_bench"
    
    train_loader, test_loader, input_size = load_data(dataset=dataset)

    # Initialize and train model
    cls = NeuralAdditiveModelBigger if is_bigger_nam else NeuralAdditiveModel
    model = cls(input_size).to(device)
    train_model(model, train_loader, device)

    # Test model
    test_model(model, test_loader, device)

    # Save and Load full model
    model_path = f"models/{dataset}/nam_full.pth"
    save_full_model(model, model_path)
    loaded_model = load_full_model(model_path, input_size, device, is_bigger_nam)
    test_model(loaded_model, test_loader, device)

    # # Save and Load partial models
    # partial_models_path = f"models/{dataset}/partial_models/"
    # save_partial_models(model, partial_models_path)
    # partial_loaded_model = NeuralAdditiveModel(input_size).to(device)
    # load_partial_models(partial_loaded_model, partial_models_path, device)
    # test_model(partial_loaded_model, test_loader, device)
