from utils.TrainModels import TrainModels

SEQ_LEN = 100

model_trainer = TrainModels(seq_len=SEQ_LEN)

# Train Perceptron
print("Now training Perceptron....")
trained_perc_model = model_trainer.train_perceptron()

# Train RNN
print("Now training RNN....")
trained_rnn_model = model_trainer.train_RNN(batch_size=32)

# TODO: Train LSTM

# TODO: Train Transformer

# TODO: Evaluate Perceptron

# TODO: Evaluate RNN

# TODO: Evaluate LSTM

# TODO: Evaluate Transformer
