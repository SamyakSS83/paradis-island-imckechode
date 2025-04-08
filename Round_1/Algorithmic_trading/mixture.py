import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

prices1 = pd.read_csv(r'./round-1-island-data-bottle/round-1-island-data-bottle/prices_round_1_day_-2.csv',sep=';')
prices2 = pd.read_csv(r'./round-1-island-data-bottle/round-1-island-data-bottle/prices_round_1_day_-1.csv',sep=';')
prices3 = pd.read_csv(r'./round-1-island-data-bottle/round-1-island-data-bottle/prices_round_1_day_0.csv',sep=';')

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
# plt.plot(kelp_x, kelp_y, label='Kelp', color='green')
# plt.plot(resin_x, resin_y, label='Rainforest Resin', color='blue')
# plt.plot(squid_ink_x, squid_ink_y, label='Squid Ink', color='purple')
# plt.plot(squid_ink_x, squid_ink_weighted_mean, label='Weighted squid ink price', color='green')
# plt.xlabel('Timestamp')
# plt.ylabel('Mid Price')
# plt.title('Mid Price of Products Over Time')
# plt.legend()
# plt.show()

mse = np.mean((squid_ink_weighted_mean - squid_ink_y) ** 2)
print(f'Mean Squared Error: {mse}')

# print(all_prices.head(6))

# 🧠 Hardcoded Learned Parameters
w1 = np.array([-1.1997, 1.9629, 0.0335, -0.7015, 0.2122])
b1 = -0.1123
w2 = np.array([-0.1914, 0.6438, 0.7871, -0.5009, -0.5462])
b2 = -0.275
w_poly = np.array([-2.0193, -1.2397, 1.6306, 1.9031, 0.01])
w_lin = np.array([0.4267, 0.1615, 2.9293, 0.9701, -0.2774])
a1 = -0.9555
a2 = -1.6064
a3 = 0.0
a4 = 0.2368
b3 = -1.4222 + 7.3

# Use the full squid_ink_y array
y = squid_ink_y
window = 5
predictions = []

# Loop through data starting from `window` index
for i in range(window, len(y)):
    x = y[i - window:i]  # past 5 values

    ar1 = np.dot(w1, x) + b1
    ar2 = np.dot(w2, x) + b2
    poly = np.sum(w_poly * (x ** 2))
    linear = np.dot(w_lin, x)

    pred = a1 * np.sin(ar1) + a2 * np.cos(ar2) + a3 * poly + a4 * linear + b3
    predictions.append(pred)

# Align true values with predictions
true_values = y[window:]
predictions = np.array(predictions)

# 📉 Evaluation
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(true_values, predictions)
r2 = r2_score(true_values, predictions)

# 📊 Plotting
plt.figure(figsize=(14, 4))
plt.plot(true_values, label="True", color='black')
plt.plot(predictions, label="Predicted", color='orange', alpha=0.7)
plt.title(f"Backtest on Full Data\nMSE: {mse:.2f}, R²: {r2:.4f}")
plt.xlabel("Time"); plt.ylabel("Price")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()
