import pandas as pd
import numpy as np

prices1 = pd.read_csv(r'Round_1\Algorithmic_trading\round-1-island-data-bottle\round-1-island-data-bottle\prices_round_1_day_-2.csv',sep=';')
prices2 = pd.read_csv(r'Round_1\Algorithmic_trading\round-1-island-data-bottle\round-1-island-data-bottle\prices_round_1_day_-1.csv',sep=';')
prices3 = pd.read_csv(r'Round_1\Algorithmic_trading\round-1-island-data-bottle\round-1-island-data-bottle\prices_round_1_day_0.csv',sep=';')

import matplotlib.pyplot as plt
# Concatenate all dataframes
all_prices = pd.concat([prices1, prices2, prices3])
all_prices['timestamp'] = all_prices['timestamp']+(999900*(all_prices['day']+2))

kelp_data = all_prices[all_prices['product'] == 'KELP']
kelp_x = kelp_data['timestamp'].to_numpy()
kelp_y = kelp_data['mid_price'].to_numpy()
kelp_z1 = kelp_data['bid_price_1'].to_numpy()
kelp_z2 = kelp_data['bid_price_2'].to_numpy()
kelp_z3 = kelp_data['bid_price_3'].to_numpy()
kelp_w1 = kelp_data['ask_price_1'].to_numpy()
kelp_w2 = kelp_data['ask_price_2'].to_numpy()
kelp_w3 = kelp_data['ask_price_3'].to_numpy()
# print(kelp_x)
# print(kelp_y)

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



plt.figure(figsize=(10, 6))
# plt.plot(kelp_x, kelp_z1, label='Kelp Bid Price 1', color='green')
# plt.plot(kelp_x, kelp_z2, label='Kelp Bid Price 2', color='green')
# plt.plot(kelp_x, kelp_z3, label='Kelp Bid Price 3', color='green')
# plt.plot(kelp_x, kelp_w1, label='Kelp Ask Price 1', color='red')
# plt.plot(kelp_x, kelp_w2, label='Kelp Ask Price 2', color='red')
# plt.plot(kelp_x, kelp_w3, label='Kelp Ask Price 3', color='red')
plt.plot(kelp_x, kelp_y, label='Kelp', color='blue')
# plt.plot(resin_x, resin_y, label='Rainforest Resin', color='blue')
# plt.plot(squid_ink_x, squid_ink_z1, label='squid_ink Bid Price 1', color='lime')
# plt.plot(squid_ink_x, squid_ink_z2, label='squid_ink Bid Price 2', color='lime')
# plt.plot(squid_ink_x, squid_ink_z3, label='squid_ink Bid Price 3', color='lime')
# plt.plot(squid_ink_x, squid_ink_w1, label='squid_ink Ask Price 1', color='crimson')
# plt.plot(squid_ink_x, squid_ink_w2, label='squid_ink Ask Price 2', color='crimson')
# plt.plot(squid_ink_x, squid_ink_w3, label='squid_ink Ask Price 3', color='crimson')
plt.plot(squid_ink_x, squid_ink_y, label='Squid Ink', color='purple')
plt.xlabel('Timestamp')
plt.ylabel('Mid Price')
plt.title('Mid Price of Products Over Time')
plt.legend()
plt.show()

# print(all_prices.head(6))
