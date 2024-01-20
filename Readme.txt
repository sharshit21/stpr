# Stock Index Prediction

This Python code uses LSTM (Long Short-Term Memory) and EMA (Exponential Moving Average) models to predict stock prices. It combines the predictions from both models to improve accuracy.

## Dependencies

The following libraries are required to run the code:
numpy (imported as np)
pandas (imported as pd)
tensorflow
arch (from arch_model)
sklearn.preprocessing (imported as MinMaxScaler)
tensorflow.keras.models (imported as Sequential)
tensorflow.keras.layers (imported as LSTM, Dense)
sklearn.linear_model (imported as LinearRegression)
matplotlib.pyplot (imported as plt)



## Instructions for Running the Code
1- Make sure you have Python installed on your system.
2- Install the required dependencies using pip or any package manager of your choice.
3- Place the input files (sample_input.csv and sample_close.txt) in the same directory as the script.
4- Open a terminal or command prompt.
5- Navigate to the directory where the script is located.
6- Run the script using the command python script_name.py.
7- The script will execute and display the results in the console.

## Usage 
1- Ensure that you have the necessary dependencies installed.
2- Place the input files (sample_input.csv and sample_close.txt) in the same directory as the script.
3- Run the script using the command python script_name.py.
4- The script will read the CSV file and the text file containing actual closing prices.
5- The predict_func() function will make predictions using a trained LSTM model and EMA.
6- The script will calculate the mean square error and directional accuracy of the predicted closing prices.
7- The results will be displayed in the console.
