import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Upload the data
train_path = "sales_data_training.csv"
train_scaled_path = "sales_data_training_scaled.csv"
test_path = "sales_data_test.csv"
test_scaled_path = "sales_data_test_scaled.csv"
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Scaler the data
scaler_x = StandardScaler()
scaler_y = StandardScaler()
scaled_training = scaler_x.fit_transform(train_data)
scaled_testing = scaler_y.fit_transform(test_data)
scaled_training_df = pd.DataFrame(scaled_training, columns=train_data.columns.values)
scaled_testing_df = pd.DataFrame(scaled_testing, columns=test_data.columns.values)

# Save the scaled data
scaled_training_df.to_csv(train_scaled_path, index=False)
scaled_testing_df.to_csv(test_scaled_path, index=False)

# Upload the scaled data
data_train = pd.read_csv(train_scaled_path)
data_test = pd.read_csv(test_scaled_path)

# Split into feature map and the output value
X_train = data_train.drop('total_earnings', axis=1).values
Y_train = data_train[['total_earnings']].values
X_test = data_test.drop('total_earnings', axis=1).values
Y_test = data_test[['total_earnings']].values

# Keras Model
model = Sequential()
model.add(Dense(50, input_dim = 9, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse'])

# History
history = model.fit(X_train, Y_train, epochs = 50)
plt.plot(history.history['mean_squared_error'], color = 'blue')

# Compute the error ('mse')
Y_pred = model.predict(X_test)
Y_pred = scaler_y.inverse_transform(Y_pred)
Y_test = scaler_y.inverse_transform(Y_test)
error = mean_squared_error(Y_test, Y_pred)
print("Error" + str(error))




