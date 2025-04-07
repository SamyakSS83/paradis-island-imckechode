import pandas as pd
import numpy as np
# Load the CSV file
file_path = r'D:\Study-Work\Study\Prosperity_IMC\paradis-island-imckechode\Round_1\Algorithmic_trading\round-1-island-data-bottle\round-1-island-data-bottle\prices_round_1_day_0.csv'
data = pd.read_csv(file_path)
import matplotlib.pyplot as plt
data = data['day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss'].str.split(';', expand=True)
data.columns = ['day', 'timestamp', 'product', 'bid_price_1', 'bid_volume_1', 'bid_price_2', 'bid_volume_2', 'bid_price_3', 'bid_volume_3', 
                'ask_price_1', 'ask_volume_1', 'ask_price_2', 'ask_volume_2', 'ask_price_3', 'ask_volume_3', 'mid_price', 'profit_and_loss']
squid_ink_data = data[data['product'] == 'SQUID_INK']
squid_ink_data['timestamp'] = pd.to_numeric(squid_ink_data['timestamp'])
squid_ink_data['mid_price'] = pd.to_numeric(squid_ink_data['mid_price'])
squid_ink_data['rolling_avg_5'] = squid_ink_data['mid_price'].rolling(window=5).mean()
valid_data = squid_ink_data.dropna(subset=['rolling_avg_5'])
squared_diff = (valid_data['mid_price'] - valid_data['rolling_avg_5']) ** 2
mse_rolling_avg = squared_diff.mean()
print(f"Mean Squared Error (MSE) for 5-period Moving Average model: {mse_rolling_avg:.4f}")
rmse_rolling_avg = np.sqrt(mse_rolling_avg)
print(f"Root Mean Squared Error (RMSE) for 5-period Moving Average model: {rmse_rolling_avg:.4f}")
prices = squid_ink_data['mid_price'].to_numpy()
time_diff = squid_ink_data['timestamp'].diff().mean()
sampling_rate = 1 / time_diff if time_diff > 0 else 1
print(f"Effective sampling rate: 1 sample per {time_diff} time units")
window = np.hanning(len(prices))
windowed_prices = prices * window

fft_result = np.fft.fft(windowed_prices) / len(prices)
N = len(prices)
freq = np.fft.fftfreq(N, d=time_diff)  

# plt.figure(figsize=(12, 8))
# plt.subplot(2, 1, 1)
magnitude = np.abs(fft_result[:N//2])
# plt.semilogy(freq[:N//2], magnitude)
# plt.title('FFT Magnitude Spectrum of SQUID_INK Prices (Log Scale)')
# plt.xlabel('Frequency (1/time unit)')
# plt.ylabel('Magnitude (log scale)')
# plt.grid(True)

# dominant_indices = np.argsort(magnitude)[-5:]  
# for idx in dominant_indices:
#     if idx > 0:  
#         period = 1 / freq[idx]
#         plt.axvline(x=freq[idx], color='r', linestyle='--', alpha=0.3)
#         plt.text(freq[idx], magnitude[idx], f" Period: {period:.1f}", 
#                  verticalalignment='bottom')



















tempp = pd.read_csv(r"D:\Study-Work\Study\Prosperity_IMC\paradis-island-imckechode\Tutorial_round\so_far_best_output.csv")
pricesnew = tempp['price'].to_numpy()

N = len(pricesnew)








dtft_freq = np.linspace(-200,200, 100000)  
dtft_result = np.zeros(len(dtft_freq), dtype=complex)

n_array = np.arange(N)

# plt.figure(figsize=(12, 4))
plt.plot(pricesnew, label='Input Prices')
plt.title('Input Price Data')
plt.xlabel('Time Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# for i, f in enumerate(dtft_freq):
#     dtft_result[i] = np.sum(pricesnew * np.exp(-2j * np.pi * f * n_array))

# dtft_result /= N

# # plt.subplot(2, 1, 2)
# # plt.semilogy(dtft_freq, np.abs(dtft_result))
# # plt.title('DTFT Magnitude Spectrum of SQUID_INK Prices (Low Frequencies)')
# # plt.xlabel('Normalized Frequency')
# # plt.ylabel('Magnitude (log scale)')
# # plt.grid(True)

# # plt.figtext(0.5, 0.01, 
# #             "Peaks in the spectrum indicate cyclical patterns in price data.\n" + 
# #             "Frequency = 0.1 corresponds to a cycle every 10 time units.", 
# #             ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
# # plt.tight_layout()
# # plt.subplots_adjust(bottom=0.15)
# # plt.show()

# plt.figure(figsize=(12, 6))
# # plt.semilogy(dtft_freq, np.abs(dtft_result))
# plt.plot(dtft_freq, np.abs(dtft_result))
# plt.title('DTFT Magnitude Spectrum of Prices')
# plt.xlabel('Normalized Frequency')
# plt.ylabel('Magnitude (log scale)')
# plt.grid(True)
# plt.show()


# N_fft = 512

# # FFT using numpy
# X_fft = np.fft.fft(pricesnew, N_fft)
# freq_fft = np.fft.fftfreq(N_fft, d=1)

# plt.plot(freq_fft, np.abs(X_fft))
# plt.title("FFT Magnitude")
# plt.xlabel("Normalized Frequency")
# plt.ylabel("|X[k]|")
# plt.grid()
# plt.show()

# plt.plot(freq_fft, np.angle(X_fft))
# plt.title("FFT Phase")
# plt.xlabel("Normalized Frequency")
# plt.ylabel("∠X[k]")
# plt.grid()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Sample discrete-time signal
x = pricesnew.tolist()

# Length for FFT (can be length of x or next power of 2 for speed)
N_fft = 512

# FFT using numpy
X_fft = np.fft.fft(x, N_fft)
freq_fft = np.fft.fftfreq(N_fft, d=1)

# DTFT computation (high-resolution)
omega = np.linspace(-np.pi, np.pi, N_fft)
X_dtft = np.array([sum(x[n] * np.exp(-1j * w * n) for n in range(len(x))) for w in omega])

# Plotting
plt.figure(figsize=(12, 6))

# Magnitude of DTFT
plt.subplot(2, 2, 1)
plt.plot(omega, np.abs(X_dtft))
plt.title("DTFT Magnitude")
plt.xlabel("ω (radians/sample)")
plt.ylabel("|X(e^{jω})|")
plt.grid()

# Phase of DTFT
plt.subplot(2, 2, 2)
plt.plot(omega, np.angle(X_dtft))
plt.title("DTFT Phase")
plt.xlabel("ω (radians/sample)")
plt.ylabel("∠X(e^{jω})")
plt.grid()

# Magnitude of FFT
plt.subplot(2, 2, 3)
plt.plot(freq_fft, np.abs(X_fft))
plt.title("FFT Magnitude")
plt.xlabel("Normalized Frequency")
plt.ylabel("|X[k]|")
plt.grid()

# Phase of FFT
plt.subplot(2, 2, 4)
plt.plot(freq_fft, np.angle(X_fft))
plt.title("FFT Phase")
plt.xlabel("Normalized Frequency")
plt.ylabel("∠X[k]")
plt.grid()

plt.tight_layout()
plt.show()

