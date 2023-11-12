import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def dividir(time_step, time, series):
    """
    Divides the dataset into training and validation sets.

    Parameters:
    - time_step: List or array of time steps.
    - time: Array of time points corresponding to the series data.
    - series: Array of data values (e.g., temperature, humidity).

    Returns:
    - time_train: Time points for training set.
    - x_train: Data values for training set.
    - time_valid: Time points for validation set.
    - x_valid: Data values for validation set.
    - split_time: Index at which the dataset is split into training and validation.
    """
    split_time = int(len(time_step) * 0.75)
    print(split_time)
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]
    return time_train, x_train, time_valid, x_valid, split_time 

def modelo():
    """
    Creates and returns a TensorFlow Sequential model for time series forecasting.

    The model architecture includes:
    - 1D Convolutional layer for initial feature extraction.
    - Two LSTM layers for capturing temporal dependencies.
    - Dense layers for further processing and output.

    Returns:
    - A compiled TensorFlow Sequential model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=60, kernel_size=5, strides=1, padding="causal", activation="relu", input_shape=[None, 1]),
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 400)
    ])
    return model 

def plot_tt2(name, column_name, columna, time, series, format="-", start=0, end=None, dims=(18, 5)):
    """
    Plots and saves a time series graph.

    Parameters:
    - name: Name of the dataset.
    - column_name: Name of the variable being plotted.
    - columna: Column identifier used in saving the file.
    - time: Array of time points.
    - series: Array of data values.
    - format: (Optional) Matplotlib format string for line style.
    - start, end: (Optional) Indices to slice the time series for plotting.

    The function saves the plot as an image file.
    """
    plt.figure(figsize=dims)
    plt.plot(time[start:end], series[start:end], format)
    plt.title("Dataset: {}\nVariable: {}".format(name, column_name))
    plt.xlabel("n dias")
    plt.ylabel("Valores")
    plt.grid(True)
    plt.savefig("{}_{}_valores.png".format(name.replace(".csv", ""), columna))
    plt.show()

def plot_tt(name, column_name, columna, time, series, format="-", start=0, end=None, dims=(18, 5)):
    """
    Plots a time series graph.

    Parameters are the same as plot_tt2 but this function does not save the plot as an image.
    """
    plt.figure(figsize=dims)
    plt.plot(time[start:end], series[start:end], format)
    plt.title("Dataset: {}\nVariable: {}".format(name, column_name))
    plt.xlabel("n dias")
    plt.ylabel("Valores")
    plt.grid(True)
    plt.show()

def tensorial_preprocessing(x_train, window_size, shuffle_buffer_size, batch_size):
    """
    Prepares the training data for input into the TensorFlow model.

    Parameters:
    - x_train: Training data series.
    - window_size: Size of the window for segmenting the data.
    - shuffle_buffer_size: Size of the buffer for shuffling data.
    - batch_size: Batch size for the model training.

    Returns:
    - TensorFlow dataset ready for model training.
    """
    series_expand = tf.expand_dims(x_train, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series_expand)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    ds = ds.batch(batch_size).prefetch(1)
    return ds
    
def to_forecast(model, series, window_size, shuffle_buffer_size, split_time):
    """
    Generates forecasts using the trained model.

    Parameters:
    - model: Trained TensorFlow model.
    - series: Complete dataset (including both training and validation).
    - window_size: Size of the window for segmenting the data.
    - shuffle_buffer_size: Size of the buffer for shuffling data (unused in this function).
    - split_time: Index used to split the data into training and validation.

    Returns:
    - dw: The dataset window used for predictions.
    - forecast: Predicted values.
    - rnn_forecast: Extracted forecast values post-processing.
    """
    series_newaxis = series[:, np.newaxis]
    dw = tf.data.Dataset.from_tensor_slices(series_newaxis)
    dw = dw.window(window_size, shift=1, drop_remainder=True)
    dw = dw.flat_map(lambda w: w.batch(window_size))
    dw = dw.batch(32).prefetch(1)
    forecast = model.predict(dw)
    rnn_forecast = forecast[split_time - window_size:-1, -1, 0]
    return dw, forecast, rnn_forecast
