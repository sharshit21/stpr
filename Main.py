


import numpy as np
import pandas as pd
import tensorflow as tf
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# In[6]:
# Get the path of the script file
script_path = __file__

# Get the directory path by removing the script filename from the path
script_dir = script_path.rsplit(r'\211110_Tejaswa.py', 1)[0]

# Specify the file names relative to the script location
file1_name = 'sample_input.csv'
file2_name = 'sample_close.txt'
file3_name = 'model_final.m1'

# Construct the paths to the files using the script directory
file1_path = f'{script_dir}\{file1_name}'
file2_path = f'{script_dir}\{file2_name}'
file3_path = f'{script_dir}\{file3_name}'

def evaluate():
    # Input the csv file
    """
    Sample evaluation function
    Don't modify this function
    """
    df = pd.read_csv(file1_path)
    
    actual_close = np.loadtxt(file2_path)
    
    pred_close = predict_func(df)
    
    # Calculation of squared_error
    actual_close = np.array(actual_close)
    pred_close = np.array(pred_close)
    mean_square_error = np.mean(np.square(actual_close-pred_close))


    pred_prev = [df['Close'].iloc[-1]]
    pred_prev.append(pred_close[0])
    pred_curr = pred_close
    
    actual_prev = [df['Close'].iloc[-1]]
    actual_prev.append(actual_close[0])
    actual_curr = actual_close

    # Calculation of directional_accuracy
    pred_dir = np.array(pred_curr)-np.array(pred_prev)
    actual_dir = np.array(actual_curr)-np.array(actual_prev)
    dir_accuracy = np.mean((pred_dir*actual_dir)>0)*100

    print(f'Mean Square Error: {mean_square_error:.6f}\nDirectional Accuracy: {dir_accuracy:.1f}')


# In[7]:


def predict_func(test_df):
    
    lstm_model = tf.keras.models.load_model(file3_path)
    test_df.ffill(inplace=True)
    # Prepare the DataFrame with log returns
    test_df['log_returns'] = np.log(test_df['Close']).diff().dropna()
    # Normalize the test data using the scaler from training
    scaler = MinMaxScaler()
    scaler.fit(test_df['Close'].values.reshape(-1, 1))
    scaled_test_data = scaler.transform(test_df['Close'].values.reshape(-1, 1))
    lookback_lstm = 10
    # Prepare the test data for LSTM
    inputs_lstm = scaled_test_data[-lookback_lstm:]
    inputs_lstm = inputs_lstm.reshape(1, -1, 1)

    # Make predictions using LSTM model
    predicted_prices_lstm = lstm_model.predict(inputs_lstm)
    predicted_prices_lstm = scaler.inverse_transform(predicted_prices_lstm)

    # Prepare the test data for EMA
    inputs_ema = scaled_test_data[-lookback_lstm:]
    inputs_ema = inputs_ema.reshape(-1, 1)

    # Calculate EMA using alpha 0.75
    ema = [inputs_ema[0]]  # Initialize the first EMA value as the first data point
    alpha = 0.85
    for i in range(1, len(inputs_ema)):
        ema_value = alpha * inputs_ema[i] + (1 - alpha) * ema[i - 1]
        ema.append(ema_value)

    ema = np.array(ema)
    ema = scaler.inverse_transform(ema)

    # Adjust the shapes of predicted_prices_lstm and ema arrays
    predicted_prices_lstm = predicted_prices_lstm[-len(ema):]

    # Combine the predictions
    combined_predictions = 0.05 * predicted_prices_lstm[-10:, 0] + 0.95 * ema[-10:, 0]
    # Return the combined predictions for the next two days
    next_two_days = combined_predictions[-2:]
    return next_two_days


# In[8]:


if __name__ == '__main__':
    evaluate()


# In[ ]:





# In[ ]:




