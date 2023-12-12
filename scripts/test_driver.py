import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from os import cpu_count, makedirs, path
from torch.utils.data import DataLoader

from TakingStock.scripts.utils.load_data import get_train_test_datasets, get_data_tensor
from utils.TrainModels import TrainModels
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import MulticlassAccuracy
from torcheval.metrics import Perplexity

from TakingStock.scripts.utils.models.Perceptron import Perceptron
from TakingStock.scripts.utils.models.RNN import RNN
from TakingStock.scripts.utils.models.LSTM import LSTM
from TakingStock.scripts.utils.models.Transformer import Transformer


# IF YOU RUN A MAC, SET THIS TO TRUE
ON_MAC_COMPUTER = False

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

def _evaluate_model(self, opt: torch.optim.Optimizer,
                     model: nn.Module,
                     model_name: str,
                     train_dataloader: DataLoader,
                     test_dataloader: DataLoader,
                     stopping_lr: float):

    eval_frequency = 5
    makedirs("evals", exist_ok=True)
    writer = SummaryWriter(path.join("evals", model_name))
    metric1 = MulticlassAccuracy(DEVICE)
    metric2 = Perplexity(DEVICE)

    # Create Learning Rate Scheduler
    lr_sched = ReduceLROnPlateau(opt, patience=5, verbose=True)

    # Train until lr_sched and the lack of improvement on the test dataset
    # breaks us out of the training loop
    epoch_num = 0
    while opt.param_groups[0]["lr"] > stopping_lr:
        epoch_num += 1
        print(f"\nEpoch #{epoch_num}:")

        # Training loop
        train_abs_err = 0
        train_loss = 0
        for batch_feats, batch_lbls in tqdm(train_dataloader):
            # Move the data to the device for training, either GPU or CPU
            batch_feats = batch_feats.to(device=self.device, non_blocking=True)
            batch_lbls = batch_lbls.to(device=self.device, non_blocking=True)

            # Reset gradients for optimizer
            opt.zero_grad()

            # Updates metric states with new data calculated every 5 epochs
            # Computes accuracy and perplexity and adds values to a scalar
            metric1.update(batch_feats, batch_lbls)
            metric2.update(batch_feats, batch_lbls)
            if epoch_num % eval_frequency == 0:
                accuracy = metric1.compute()
                perplexity = metric2.compute()

                writer.add_scalar(tag="train_accuracy",
                                  scalar_value=accuracy)
                writer.add_scalar(tag="train_perplexity",
                                  scalar_value=perplexity)

            # Get model predictions
            preds = model(batch_feats)

            # Calculate absolute error between preds and labels
            train_abs_err += torch.sum(torch.abs(preds - batch_lbls))

            # Calculate loss (multiply by 10 because losses are small)
            loss = self.loss_fn(preds, batch_lbls) * 10
            train_loss += loss

            # Backprop with gradient clipping to prevent exploding gradients
            loss.backward()
            opt.step()

        train_mse = train_loss / len(self.train_dataset)
        train_mae = train_abs_err / len(self.train_dataset)
        print(f"Evaluation Training MSE Loss: {train_mse}")
        print(f"Evaluation Training Mean Absolute Error (MAE): {train_mae}")
        writer.add_scalar(tag="train_mse",
                          scalar_value=train_mse)
        writer.add_scalar(tag="train_mae",
                          scalar_value=train_mae)
        # Resets metric states
        metric1.reset()
        metric2.reset()


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

                    writer.add_scalar(tag="test_accuracy",
                                      scalar_value=accuracy)
                    writer.add_scalar(tag="test_perplexity",
                                      scalar_value=perplexity)

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
        writer.add_scalar(tag="test_mse",
                          scalar_value=test_mse)
        writer.add_scalar(tag="test_mae",
                          scalar_value=test_mae)

    writer.close()


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
eval_perceptron_model = torch.compile(Perceptron(
    input_size=train_dataset[0][0].shape[-1],
    seq_len=SEQ_LEN,
    bias=BIAS)).to(DEVICE)
opt = torch.optim.Adam(params=eval_perceptron_model.parameters(), lr=LR)
_evaluate_model(opt=opt,
    model=eval_perceptron_model,
    model_name="Perceptron",
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    stopping_lr=STOP_LR)

# TODO: Evaluate RNN

eval_rnn_model = torch.compile(Perceptron(
    input_size=train_dataset[0][0].shape[-1],
    seq_len=SEQ_LEN,
    bias=BIAS)).to(DEVICE)
opt = torch.optim.Adam(params=eval_rnn_model.parameters(), lr=LR)
_evaluate_model(opt=opt,
    model=eval_rnn_model,
    model_name="RNN",
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    stopping_lr=STOP_LR)

# TODO: Evaluate LSTM

eval_lstm_model = torch.compile(LSTM(
    input_size=train_dataset[0][0].shape[-1],
    seq_len=SEQ_LEN,
    bias=BIAS)).to(DEVICE)
opt = torch.optim.Adam(params=eval_rnn_model.parameters(), lr=LR)
_evaluate_model(opt=opt,
    model=eval_lstm_model,
    model_name="LSTM",
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    stopping_lr=STOP_LR)

# TODO: Evaluate Transformer

eval_trans_model = torch.compile(Transformer(
    input_size=train_dataset[0][0].shape[-1],
    seq_len=SEQ_LEN,
    bias=BIAS)).to(DEVICE)
opt = torch.optim.Adam(params=eval_rnn_model.parameters(), lr=LR)
_evaluate_model(opt=opt,
    model=eval_trans_model,
    model_name="Transformer",
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    stopping_lr=STOP_LR)