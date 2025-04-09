import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pickle

prices1 = pd.read_csv(r'./round-1-island-data-bottle/round-1-island-data-bottle/prices_round_1_day_-2.csv',sep=';')
prices2 = pd.read_csv(r'./round-1-island-data-bottle/round-1-island-data-bottle/prices_round_1_day_-1.csv',sep=';')
prices3 = pd.read_csv(r'./round-1-island-data-bottle/round-1-island-data-bottle/prices_round_1_day_0.csv',sep=';')

import matplotlib.pyplot as plt
# Concatenate all dataframes
all_prices = pd.concat([prices1, prices2, prices3])
all_prices['timestamp'] = all_prices['timestamp']+(1000000*(all_prices['day']+2))

kelp_data = all_prices[all_prices['product'] == 'KELP']
kelp_x = kelp_data['timestamp'].to_numpy()
kelp_y = kelp_data['mid_price'].to_numpy()
kelp_z1 = kelp_data['bid_price_1'].to_numpy()
kelp_z2 = kelp_data['bid_price_2'].to_numpy()
kelp_z3 = kelp_data['bid_price_3'].to_numpy()
kelp_w1 = kelp_data['ask_price_1'].to_numpy()
kelp_w2 = kelp_data['ask_price_2'].to_numpy()
kelp_w3 = kelp_data['ask_price_3'].to_numpy()
print(kelp_x)
print(kelp_y)

resin_data = all_prices[all_prices['product'] == 'RAINFOREST_RESIN']
resin_x = resin_data['timestamp'].to_numpy()
resin_y = resin_data['mid_price'].to_numpy()
resin_z1 = resin_data['bid_price_1'].to_numpy()
resin_z2 = resin_data['bid_price_2'].to_numpy()
resin_z3 = resin_data['bid_price_3'].to_numpy()
resin_w1 = resin_data['ask_price_1'].to_numpy()
resin_w2 = resin_data['ask_price_2'].to_numpy()
resin_w3 = resin_data['ask_price_3'].to_numpy()

squid_ink_data = all_prices[all_prices['product'] == 'SQUID_INK']
squid_ink_x = squid_ink_data['timestamp'].to_numpy()
squid_ink_y = squid_ink_data['mid_price'].to_numpy()
squid_ink_z1 = squid_ink_data['bid_price_1'].to_numpy()
squid_ink_z2 = squid_ink_data['bid_price_2'].to_numpy()
squid_ink_z3 = squid_ink_data['bid_price_3'].to_numpy()

squid_ink_w1 = squid_ink_data['ask_price_1'].to_numpy()
squid_ink_w2 = squid_ink_data['ask_price_2'].to_numpy()
squid_ink_w3 = squid_ink_data['ask_price_3'].to_numpy()
squid_ink_a1 = squid_ink_data['ask_volume_1'].to_numpy()
squid_ink_a2 = squid_ink_data['ask_volume_2'].to_numpy()
squid_ink_a3 = squid_ink_data['ask_volume_3'].to_numpy()
squid_ink_b1 = squid_ink_data['bid_volume_1'].to_numpy()
squid_ink_b2 = squid_ink_data['bid_volume_2'].to_numpy()
squid_ink_b3 = squid_ink_data['bid_volume_3'].to_numpy()
# Calculate weighted average price for squid ink using both bid and ask prices
# Handle any potential division by zero with np.nan_to_num
squid_ink_z1 = np.nan_to_num(squid_ink_z1, nan=0.0)
squid_ink_z2 = np.nan_to_num(squid_ink_z2, nan=0.0)
squid_ink_z3 = np.nan_to_num(squid_ink_z3, nan=0.0)
squid_ink_b1 = np.nan_to_num(squid_ink_b1, nan=0.0)
squid_ink_b2 = np.nan_to_num(squid_ink_b2, nan=0.0)
squid_ink_b3 = np.nan_to_num(squid_ink_b3, nan=0.0)
squid_ink_a1 = np.nan_to_num(squid_ink_a1, nan=0.0)
squid_ink_a2 = np.nan_to_num(squid_ink_a2, nan=0.0)
squid_ink_a3 = np.nan_to_num(squid_ink_a3, nan=0.0)
squid_ink_w1 = np.nan_to_num(squid_ink_w1, nan=0.0)
squid_ink_w2 = np.nan_to_num(squid_ink_w2, nan=0.0)
squid_ink_w3 = np.nan_to_num(squid_ink_w3, nan=0.0)


squid_ink_weighted_mean = (
    (squid_ink_z1 * squid_ink_b1 + squid_ink_z2 * squid_ink_b2 + squid_ink_z3 * squid_ink_b3 + squid_ink_a1*squid_ink_w1 + squid_ink_a2*squid_ink_w2 + squid_ink_a3*squid_ink_w3) /
    (squid_ink_b1 + squid_ink_b2 + squid_ink_b3 + squid_ink_a1 + squid_ink_a2 + squid_ink_a3)
)

print(squid_ink_weighted_mean)


# plt.figure(figsize=(10, 6))
# # plt.plot(kelp_x, kelp_y, label='Kelp', color='green')
# # plt.plot(resin_x, resin_y, label='Rainforest Resin', color='blue')
# # plt.plot(squid_ink_x, squid_ink_y, label='Squid Ink', color='purple')
# plt.plot(squid_ink_x, squid_ink_weighted_mean, label='Weighted squid ink price', color='green')
# plt.xlabel('Timestamp')
# plt.ylabel('Mid Price')
# plt.title('Mid Price of Products Over Time')
# plt.legend()
# plt.show()




X, y = [], []

n = 7                                 ################################################## MAY CHANGE THIS

degree = 3                            ################################################## MAY CHANGE THIS


data = squid_ink_y
for i in range(n,len(data)):
    X.append(data[i-n:i])
    y.append(data[i])

X = np.array(X)
y = np.array(y)

def make_dataset(y, window=5):
    X = np.column_stack([y[i:-(window - i)] for i in range(window)])
    y_target = y[window:]
    return X, y_target

X, y_target = make_dataset(squid_ink_y, window=10)
split_index = int(0.25 * len(X))
X_train, y_train = X[:split_index], y_target[:split_index]
X_test, y_test = X[split_index:], y_target[split_index:]
from pyESN import ESN

esn = ESN(n_inputs=10, n_outputs=1, n_reservoir=200, spectral_radius=1.2, sparsity=0.3, random_state=42)

# Train the ESN and discard the predictions returned by fit
esn.fit(X_train, y_train)

with open('esn_model.pkl', 'rb') as f:
    loaded_esn = pickle.load(f)

# Then make predictions separately
y_pred_esn = esn.predict(X_test)
mse = mean_squared_error(y_test, y_pred_esn)
r2 = r2_score(y_test, y_pred_esn)
print(f"R^2 Score: {r2}")
print(f"Mean Squared Error: {mse}")

# Plotting
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.plot(y_test, label='True')
plt.plot(y_pred_esn, label='ESN Predicted', alpha=0.7)
plt.legend()
plt.title("Echo State Network")
plt.show()
