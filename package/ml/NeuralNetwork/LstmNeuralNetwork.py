import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

class LstmNeuralNetwork():

    warnings.filterwarnings("ignore")

    def normalise_windows(self,window_data):
        normalised_data = []
        for window in window_data:
            normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
            normalised_data.append(normalised_window)
        return normalised_data

    def build_model(self,layers):
        model_train = Sequential()
        model_train.add(LSTM(
            input_dim=layers[0],
            output_dim=layers[1],
            return_sequences=True))
        model_train.add(Dropout(0.2))

        model_train.add(LSTM(
            layers[2],
            return_sequences=False))
        model_train.add(Dropout(0.2))

        model_train.add(Dense(
            output_dim=layers[3]))
        model_train.add(Activation("linear"))
        model_train.compile(loss="mse", optimizer="rmsprop")
        return model_train

    def predict_point_by_point(self,model, data):
        #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        predicted = model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequence_full(self,model, data, window_size):
        #Shift the window by 1 new prediction each time, re-run predictions on new window
        curr_frame = data[0]
        predicted = []
        for i in xrange(len(data)):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        return predicted

    def predict_sequences_multiple(self,model, data, window_size, prediction_len):
        #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        prediction_seqs = []
        for i in xrange(len(data)/prediction_len):
            curr_frame = data[i*prediction_len]
            predicted = []
            for j in xrange(prediction_len):
                predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def fit(self,feature,data):
        epochs  = 1
        seq_len = 50
        sequence_length = seq_len + 1
        result = []
        for index in range(len(data) - sequence_length):
            result.append(data[index: index + sequence_length])

        result = self.normalise_windows(result)

        result = np.array(result)

        row = round(0.9 * result.shape[0])
        train = result[:int(row), :]
        np.random.shuffle(train)
        x_train = train[:, :-1]
        y_train = train[:, -1]
        x_test = result[int(row):, :-1]
        y_test = result[int(row):, -1]

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        #X_train, y_train, X_test, y_test = self.load_data(data,seq_len, True)
        model_train = self.build_model([1, 50, 100, 1])
        model_train.fit(
            x_train,
            y_train,
            batch_size=512,
            nb_epoch=epochs,
            validation_split=0.05)
        self._model = model_train
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        return self

    def predict(self,data):
        #predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)
        #predictions = lstm.predict_sequence_full(model, X_test, seq_len)
        data = np.array(data)
        predictions = self.predict_point_by_point(self._model, data)
        return predictions.tolist()

    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
    



