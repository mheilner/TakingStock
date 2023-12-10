from os import cpu_count
from utils.TrainModels import TrainModels

# IF YOU RUN A MAC, SET THIS TO TRUE
ON_MAC_COMPUTER = True

SEQ_LEN = 100

model_trainer = TrainModels(seq_len=SEQ_LEN)

# Mac Data Loader Processes
num_dataloader_processes = 0 if ON_MAC_COMPUTER else cpu_count()

# Train Perceptron
print("Now training Perceptron....")
trained_perc_model = model_trainer.train_perceptron()

# Train RNN
print("Now training RNN....")
trained_rnn_model = model_trainer.train_RNN(batch_size=32,
                            num_dataloader_processes=num_dataloader_processes)

# Train LSTM
print("Now training LSTM....")
trained_lstm_model = model_trainer.train_LSTM(num_dataloader_processes=num_dataloader_processes,
                                              batch_size=32)

# TODO: Train Transformer
print("Now training Transformer....")
trained_transformer_model = model_trainer.train_transformer(
                            num_dataloader_processes=num_dataloader_processes)

# TODO: Evaluate Perceptron

# TODO: Evaluate RNN

# TODO: Evaluate LSTM

# TODO: Evaluate Transformer
