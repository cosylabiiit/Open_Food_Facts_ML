import cudf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data = cudf.read_csv("/home/pavit21178/Nalin_OFF/Data/OFF_FNDDS_COLS_NOVA.csv", index_col=0)
columns_needed = ["Total Fat", "Carbohydrate", "Protein", "Sodium", "Sugars, total",
                   "Fatty acids, total saturated", "Energy", "Fiber, total dietary", "novaclass"]
data = data[columns_needed]


data = data[data['novaclass'].isin([1, 2, 3, 4])]


X = data.drop('novaclass', axis=1)
y = data['novaclass']


X = X.fillna(X.mean())


mean_X = X.mean()
std_X = X.std()

print("Mean:\n", mean_X)
print("\nStandard Deviation:\n", std_X)

X_train_scaled = (X - mean_X) / std_X

y_train = cudf.Series(y.values)

print("Training data shape:", X_train_scaled.shape)



# %%
import torch


X_train_tensor = torch.tensor(X_train_scaled.to_numpy(), dtype=torch.float32)


print("Training tensor shape:", X_train_tensor.shape)



# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna

# Define your device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, activations, dropout_prob):
        super(Autoencoder, self).__init__()
        self.encoder = self.build_layers(input_dim, hidden_dims + [latent_dim], activations, dropout_prob)
        self.decoder = self.build_layers(latent_dim, hidden_dims[::-1] + [input_dim], activations, dropout_prob)

    def build_layers(self, input_dim, dims, activations, dropout_prob):
        layers = []
        for i in range(len(dims)):
            layers.append(nn.Linear(input_dim, dims[i]))
            layers.append(self.get_activation(activations[i]))
            if i < len(dims) - 1:  
                layers.append(nn.Dropout(dropout_prob))
            input_dim = dims[i]
        return nn.Sequential(*layers[:-1])  

    def get_activation(self, name):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'leaky_relu':
            return nn.LeakyReLU()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'elu':
            return nn.ELU()
        elif name == 'selu':
            return nn.SELU()
        elif name == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation function: {name}")

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

# %%
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def objective(trial):
    input_dim = X.shape[1]
    latent_dim = trial.suggest_int('latent_dim', 40, 500)  
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 3)  
    hidden_dims = [trial.suggest_int(f'hidden_dim_{i}', 8, 1000) for i in range(num_hidden_layers)]
    activations = [trial.suggest_categorical(f'activation_{i}', ['relu', 'leaky_relu', 'tanh', 'sigmoid', 'elu', 'selu', 'gelu']) for i in range(num_hidden_layers + 1)]
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 256)  
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    dropout_prob = trial.suggest_float('dropout_prob', 0.1, 0.5)

    model = Autoencoder(input_dim, latent_dim, hidden_dims, activations, dropout_prob).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    epochs = 50
    epoch_losses = []  

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for data in train_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        
        epoch_loss /= len(train_loader.dataset)
        epoch_losses.append(epoch_loss)
        print(f"Trial {trial.number}, Epoch {epoch+1}, Loss: {epoch_loss}")


    model.eval()
    with torch.no_grad():
        X_train_tensor_device = X_train_tensor.to(device)
        reconstructed, _ = model(X_train_tensor_device)
        train_loss = criterion(reconstructed, X_train_tensor_device).item()

    print(f"Trial {trial.number} final train loss: {train_loss}")

    return train_loss


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=40)

best_params = study.best_params
print("Best hyperparameters:", best_params)


# %%
input_dim = X.shape[1]
latent_dim = best_params['latent_dim']
num_hidden_layers = best_params['num_hidden_layers']
hidden_dims = [best_params[f'hidden_dim_{i}'] for i in range(num_hidden_layers)]
activations = [best_params[f'activation_{i}'] for i in range(num_hidden_layers + 1)]
lr = best_params['lr']
batch_size = best_params['batch_size']
weight_decay = best_params['weight_decay']
dropout_prob = best_params['dropout_prob']



# %%
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, activations, dropout_prob):
        super(Autoencoder, self).__init__()
        self.encoder = self.build_layers(input_dim, hidden_dims + [latent_dim], activations, dropout_prob)
        self.decoder = self.build_layers(latent_dim, hidden_dims[::-1] + [input_dim], activations, dropout_prob)

    def build_layers(self, input_dim, dims, activations, dropout_prob):
        layers = []
        for i in range(len(dims)):
            layers.append(nn.Linear(input_dim, dims[i]))
            layers.append(self.get_activation(activations[i]))
            if i < len(dims) - 1:
                layers.append(nn.Dropout(dropout_prob))
            input_dim = dims[i]
        return nn.Sequential(*layers[:-1])

    def get_activation(self, name):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'leaky_relu':
            return nn.LeakyReLU()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'elu':
            return nn.ELU()
        elif name == 'selu':
            return nn.SELU()
        elif name == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation function: {name}")

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
val_dataset = TensorDataset(X_val_tensor, X_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


model = Autoencoder(input_dim, latent_dim, hidden_dims, activations, dropout_prob).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


train_losses = []
val_losses = []


epochs = 500  
for epoch in range(epochs):
    model.train()
    train_loss_epoch = 0
    for data in train_loader:
        inputs, _ = data
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item() * inputs.size(0)

    # Compute average training loss for the epoch
    train_loss_epoch /= len(train_loader.dataset)
    train_losses.append(train_loss_epoch)

    Validation
    model.eval()
    val_loss_epoch = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, inputs)
            val_loss_epoch += loss.item() * inputs.size(0)

    # Compute average validation loss for the epoch
    val_loss_epoch /= len(val_loader.dataset)
    val_losses.append(val_loss_epoch)

    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss_epoch}')

# Plot the losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('AEC_TRAIN_TEST_Loss_plot.png')
Save the trained model
torch.save(model, 'autoencoder_best_model.pth')


# %%
def impute_missing_values(model, data_with_missing):
    model.eval()
    with torch.no_grad():
        data_tensor = torch.tensor(data_with_missing, dtype=torch.float32).to(device)
        reconstructed, _ = model(data_tensor)
        return reconstructed.cpu().numpy()




# %%
X.shape

# %%
# Impute missing values
imputed_data_1 = impute_missing_values(model, X_train_scaled.to_numpy())
imputed_data_2 = impute_missing_values(model, X_val_scaled.to_numpy())


# %%


# %%
import pandas as pd

imputed_df_train = cudf.DataFrame(imputed_data_1, columns=X_train_scaled.columns)
imputed_df_val = cudf.DataFrame(imputed_data_2, columns=X_val_scaled.columns)

imputed_df_train_pandas = imputed_df_train.to_pandas()
imputed_df_val_pandas = imputed_df_val.to_pandas()

# %%

y_train_pandas = y_train.to_pandas()
y_val_pandas = y_val.to_pandas()


imputed_df_train_pandas['novaclass'] = y_train_pandas.values
imputed_df_val_pandas['novaclass'] = y_val_pandas.values


# %%

imputed_df_train_pandas.to_csv('imputed_train_data_with_labels.csv', index=False)
imputed_df_val_pandas.to_csv('imputed_val_data_with_labels.csv', index=False)


# # %%



