# Get the training logic all together but the thing will be trained in colab via git
import data_setup, loop_logic, model_nn, model_save_logic
import torch
import torch.optim as optim
import torch.nn as nn

BATCH_SIZE = 32
NUM_WORKERS = 4
HIDDEN_NURONS = 128
EPOCHS = 3
LR = 1e-3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CLASS_NAMES, train_dl, val_dl, test_dl, X = data_setup.setup_data(BATCH_SIZE, NUM_WORKERS, DEVICE)

model = model_nn.define_nn_arch(HIDDEN_NURONS, X).to(DEVICE)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

loop_logic.train(model, train_dl, loss_fn, optimizer, EPOCHS, DEVICE)

