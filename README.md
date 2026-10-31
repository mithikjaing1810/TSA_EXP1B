# Ex.No: 1B                     CONVERSION OF NON STATIONARY TO STATIONARY DATA
# Date: 31-10-2026
# Reg no : 212224240087
# Dataset : Nvidia stock price

### AIM:
To perform regular differncing,seasonal adjustment and log transformatio on international airline passenger data
### ALGORITHM:
1. Import the required packages like pandas and numpy
2. Read the data using the pandas
3. Perform the data preprocessing if needed and apply regular differncing,seasonal adjustment,log transformation.
4. Plot the data according to need, before and after regular differncing,seasonal adjustment,log transformation.
5. Display the overall results.
### PROGRAM:
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
data = pd.read_csv('/content/NVIDIA_STOCK.csv')

data = data.iloc[2:].copy()

data['Price'] = pd.to_datetime(data['Price'])

data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

data.set_index('Price', inplace=True)


data['stock_diff'] = data['Close'].diff()


result = seasonal_decompose(
    data['Close'],
    model='additive',
    period=252
)

data['stock_sea_diff'] = result.resid

data['stock_log'] = np.log(data['Close'])

data['stock_log_diff'] = data['stock_log'].diff()

data['stock_log_seasonal_diff'] = (
    data['stock_log'].diff().diff(252)
)

plt.figure(figsize=(16, 18))

plt.subplot(6, 1, 1)
plt.plot(data['Close'], label='Original')
plt.legend(loc='best')
plt.title('Original NVIDIA Stock Data')
plt.xlabel('Year')
plt.ylabel('Stock Price')


plt.subplot(6, 1, 2)
plt.plot(data['stock_diff'], label='Regular Differencing')
plt.legend(loc='best')
plt.title('Regular Differencing')
plt.xlabel('Year')
plt.ylabel('Differenced Stock Price')


plt.subplot(6, 1, 3)
plt.plot(data['stock_sea_diff'], label='Seasonal Adjustment')
plt.legend(loc='best')
plt.title('Seasonal Adjustment')
plt.xlabel('Year')
plt.ylabel('Adjusted Stock Price')


plt.subplot(6, 1, 4)
plt.plot(data['stock_log'], label='Log Transformation')
plt.legend(loc='best')
plt.title('Log Transformation')
plt.xlabel('Year')
plt.ylabel('Log Stock Price')


plt.subplot(6, 1, 5)
plt.plot(data['stock_log_diff'], label='Log + Regular Differencing')
plt.legend(loc='best')
plt.title('Log Transformation and Regular Differencing')
plt.xlabel('Year')
plt.ylabel('Log Difference')


plt.subplot(6, 1, 6)
plt.plot(
    data['stock_log_seasonal_diff'],
    label='Log + Regular + Seasonal Differencing'
)
plt.legend(loc='best')
plt.title('Log Transformation, Regular and Seasonal Differencing')
plt.xlabel('Year')
plt.ylabel('Seasonally Differenced Log Price')

plt.tight_layout()
plt.show()

data.plot(kind = 'line')

```

### OUTPUT:

<img width="1588" height="1789" alt="image" src="https://github.com/user-attachments/assets/d59f3a8c-1671-41f7-ab09-1021a1423d94" />
<img width="554" height="405" alt="image" src="https://github.com/user-attachments/assets/51bb01eb-3416-4328-8cf7-4bdfcb873e28" />



### RESULT:
Thus we have created the python code for the conversion of non stationary to stationary data on international airline passenger
data.
