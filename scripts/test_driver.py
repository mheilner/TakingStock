import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from os import cpu_count, makedirs, path
from torch.utils.data import DataLoader

from utils.load_data import get_train_test_datasets, get_data_tensor
from utils.TrainModels import TrainModels
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import MulticlassAccuracy
from torcheval.metrics import Perplexity

from utils.models.Perceptron import Perceptron
from utils.models.RNN import RNN
from utils.models.LSTM import LSTM
from utils.models.Transformer import Transformer


# IF YOU RUN A MAC, SET THIS TO TRUE
ON_MAC_COMPUTER = True

SEQ_LEN = 100
BATCH_SIZE = 32
DEVICE = device="cuda" if torch.cuda.is_available() else "cpu"
BIAS = True
LR = float(0.01)
STOP_LR = 0.0001

model_trainer = TrainModels(seq_len=SEQ_LEN)

# Mac Data Loader Processes
num_dataloader_processes = 0 if ON_MAC_COMPUTER else cpu_count()

# Train Perceptron
print("Now training Perceptron....")
trained_perc_model = model_trainer.train_perceptron(
                            num_dataloader_processes=num_dataloader_processes,
                            batch_size=BATCH_SIZE,
                            use_pretrained=True)

# Train RNN
print("Now training RNN....")
trained_rnn_model = model_trainer.train_RNN(
                            num_dataloader_processes=num_dataloader_processes,
                            batch_size=BATCH_SIZE,
                            use_pretrained=True)

# Train LSTM
print("Now training LSTM....")
trained_lstm_model = model_trainer.train_LSTM(
                            num_dataloader_processes=num_dataloader_processes,
                            batch_size=BATCH_SIZE,
                            use_pretrained=True)

# Train Transformer
print("Now training Transformer....")
trained_transformer_model = model_trainer.train_transformer(
                            num_dataloader_processes=num_dataloader_processes,
                            batch_size=BATCH_SIZE,
                            use_pretrained=True)

def evaluate_model(model: nn.Module,
                   train_dataloader: DataLoader,
                   test_dataloader: DataLoader):

    eval_frequency = 5
    makedirs("evals", exist_ok=True)
    metric1 = MulticlassAccuracy(DEVICE)
    metric2 = Perplexity(DEVICE)

    # Testing loop
    test_abs_err = 0
    test_loss = 0
    with torch.inference_mode():
        for batch_feats, batch_lbls in tqdm(test_dataloader):
            # Move the data to the device for testing, either GPU or CPU
            batch_feats = batch_feats.to(device=DEVICE, non_blocking=True)
            batch_lbls = batch_lbls.to(DEVICE, non_blocking=True)

            # Updates metric states with new data calculated every 5 epochs
            # Computes accuracy and perplexity and adds values to a scalar
            metric1.update(batch_feats, batch_lbls)
            metric2.update(batch_feats, batch_lbls)
            if epoch_num % eval_frequency == 0:
                accuracy = metric1.compute()
                perplexity = metric2.compute()

            # Get model predictions
            preds = model(batch_feats)

            # Calculate absolute error between preds and labels
            test_abs_err += torch.sum(torch.abs(preds - batch_lbls))

            # Calculate loss (multiply by 10 because losses are small)
            loss = self.loss_fn(preds, batch_lbls) * 10
            test_loss += loss

    test_mse = test_loss / len(self.test_dataset)
    test_mae = test_abs_err / len(self.test_dataset)
    print(f"Testing MSE Loss: {test_mse}")
    print(f"Testing Mean Absolute Error (MAE): {test_mae}")



master_tensor = get_data_tensor(relative_change=True)
train_dataset, test_dataset = get_train_test_datasets(
                data_tensor=master_tensor,
                seq_len=SEQ_LEN,
                train_split=0.8)
train_dataloader = DataLoader(dataset=train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=num_dataloader_processes)
test_dataloader = DataLoader(dataset=test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=num_dataloader_processes)


# TODO: Evaluate Perceptron
evaluate_model(
    model=trained_perc_model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader)

# TODO: Evaluate RNN

evaluate_model(
    model=trained_rnn_model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader)

# TODO: Evaluate LSTM

evaluate_model(
    model=trained_lstm_model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader)

# TODO: Evaluate Transformer

evaluate_model(
    model=trained_transformer_model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader)
