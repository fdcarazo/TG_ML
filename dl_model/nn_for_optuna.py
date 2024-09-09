import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib
import matplotlib.pyplot as plt

# Define the neural network model
class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_units, dropout_prob):
        super(RegressionModel, self).__init__()
        layers = []
        in_features = input_size
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            in_features = hidden_units
        layers.append(nn.Linear(in_features, 1))  # Output layer
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Define training and evaluation functions
def train(model, criterion, optimizer, dataloader):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, criterion, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameters to optimize
    input_size = 10  # Example input size
    hidden_layers = trial.suggest_int('hidden_layers', 1, 3)
    hidden_units = trial.suggest_int('hidden_units', 16, 128)
    dropout_prob = trial.suggest_float('dropout_prob', 0.0, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    batch_size = trial.suggest_int('batch_size', 16, 128)

    # Create synthetic regression data
    X, y = make_regression(n_samples=1000, n_features=input_size, noise=0.1)
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.float32).view(-1, 1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model, criterion, and optimizer
    model = RegressionModel(input_size, hidden_layers, hidden_units, dropout_prob)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train and evaluate
    train_losses = []
    val_losses = []

    for epoch in range(10):  # Fixed number of epochs
        train_loss = train(model, criterion, optimizer, train_loader)
        val_loss = evaluate(model, criterion, val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    # Save the best model and scaler
    trial.set_user_attr("train_losses", train_losses)
    trial.set_user_attr("val_losses", val_losses)
    
    return val_losses[-1]

# Optimize hyperparameters using Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Print best hyperparameters
print("Best hyperparameters:")
for key, value in study.best_params.items():
    print(f"{key}: {value}")

print(f"Best value: {study.best_value}")

# Plot training and validation loss
best_train_losses = study.best_trial.user_attrs["train_losses"]
best_val_losses = study.best_trial.user_attrs["val_losses"]

plt.figure(figsize=(12, 6))
plt.plot(best_train_losses, label='Train Loss')
plt.plot(best_val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Save the best model and scaler
best_trial_params = study.best_params
input_size = 10  # Example input size

best_model = RegressionModel(input_size, best_trial_params['hidden_layers'],
                             best_trial_params['hidden_units'],
                             best_trial_params['dropout_prob'])

# Load the best trial's parameters and save the model
best_model.load_state_dict(torch.load('best_model.pth'))
joblib.dump(scaler, 'best_scaler.pkl')

print("Best model and scaler saved.")
