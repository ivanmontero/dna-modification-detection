from tensorflow import keras
import tqdm

# Callback for training progress.
class train_progress(keras.callbacks.Callback):
    def __init__(self, length):
        self.accuracy = 0
        self.validation = 0
        self.length = length
        self.format = '{desc:11} {n_fmt:>8}/{total_fmt:8} {bar:100} - {rate_fmt}{postfix}'

    def on_train_begin(self, logs):
        self.length = self.length * self.params['epochs']
        self.progress = tqdm.tqdm(
            desc = 'Training',
            total = self.length,
            unit = ' batches',
            bar_format = self.format)

    def on_train_batch_end(self, batch, logs):
        self.progress.update()

        self.accuracy = logs['accuracy'] * 100
        if self.validation:
            metrics = f'Accuracy: {self.accuracy:.2f}, Validation: {self.validation:.2f}'
        else: 
            metrics = f'Accuracy: {self.accuracy:.2f}, Validation: N/A'
        self.progress.set_postfix_str(metrics)
        
    def on_test_batch_end(self, batch, logs):
        self.progress.update()

        self.validation = logs['accuracy'] * 100
        metrics = f'Accuracy: {self.accuracy:.2f}, Validation: {self.validation:.2f}'
        self.progress.set_postfix_str(metrics)

    def on_train_end(self, logs):
        self.progress.close()

# Callback for prediction progress. 
class predict_progress(keras.callbacks.Callback):
    def __init__(self, length):
        self.accuracy = 0
        self.length = length
        self.format = '{desc:11} {n_fmt:>8}/{total_fmt:8} {bar:100} - {rate_fmt}'

    def on_predict_begin(self, logs):
        self.progress = tqdm.tqdm(
            desc = 'Validation',
            total = self.length,
            unit = ' batches',
            bar_format = self.format)

    def on_predict_batch_end(self, batch, logs):
        self.progress.update()

    def on_predict_end(self, logs):
        self.progress.close()

