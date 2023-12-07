from os import cpu_count
from utils.TrainModels import TrainModels

ON_MAC_COMPUTER = False
SEQ_LEN = 100

model_trainer = TrainModels(seq_len=SEQ_LEN)

# Train Perceptron
# print("Now training Perceptron....")
# trained_perc_model = model_trainer.train_perceptron()

# Train RNN
print("Now training RNN....")
num_dataloader_processes = 0 if ON_MAC_COMPUTER else cpu_count()
trained_rnn_model = model_trainer.train_RNN(batch_size=32,
                            num_dataloader_processes=num_dataloader_processes)

# Train LSTM
# print("Now training LSTM....")
# trained_lstm_model = model_trainer.train_LSTM(batch_size=32, input_size=100)

# TODO: Train Transformer

# TODO: Evaluate Perceptron

# TODO: Evaluate RNN

# TODO: Evaluate LSTM

# TODO: Evaluate Transformer
