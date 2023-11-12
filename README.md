# WeatherForecast-LSTM

This project demonstrates the use of TensorFlow and Python for analyzing and forecasting weather data. It consists of a Jupyter Notebook that performs data analysis, model training, and prediction, along with a Python module that provides essential functions for data processing and model building.

## Project Structure

- `weather_model.ipynb`: A Jupyter Notebook that contains the main analysis, including data preprocessing, model training, and visualization.
- `utils.py`: A Python module that provides functions used in the notebook for data division, model creation, plotting, preprocessing, and forecasting.

## `weather_model.ipynb`

This Jupyter Notebook contains the core analysis workflow:
1. Importing necessary libraries.
2. Data loading and preprocessing.
3. Data visualization.
4. Building and training the neural network model.
5. Forecasting and visualizing the results.

## `utils.py`

This Python module contains essential functions:
- `dividir`: Splits the dataset into training and validation sets.
- `modelo`: Creates and compiles a TensorFlow Sequential model for time series forecasting.
- `plot_tt` and `plot_tt2`: Functions for plotting time series data. `plot_tt2` also saves the plot as an image.
- `tensorial_preprocessing`: Prepares the data for training in TensorFlow format.
- `to_forecast`: Generates forecasts using the trained TensorFlow model.

## Usage

To use this project, clone the repository and run the Jupyter Notebook `weather_model.ipynb`. Ensure that `utils.py` is in the same directory as the notebook, as it imports functions from this module.

## Requirements

This project requires the following libraries:
- TensorFlow
- NumPy
- Pandas
- Matplotlib

Install these libraries using pip:

```bash
pip install tensorflow numpy pandas matplotlib
```

## Contributing
Contributions to this project are welcome. Please fork the repository and open a pull request with your changes or suggestions.

## License
This project is open-sourced under the MIT License. See the LICENSE file for more details. 
