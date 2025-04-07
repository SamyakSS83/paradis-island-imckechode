import pandas as pd
import numpy as np
# Load the CSV file
file_path = 'prices_round_1_day_0.csv'
data = pd.read_csv(file_path)

import matplotlib.pyplot as plt

# Split the single column into multiple columns based on the delimiter
data = data['day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss'].str.split(';', expand=True)

# Rename the columns for easier access
data.columns = ['day', 'timestamp', 'product', 'bid_price_1', 'bid_volume_1', 'bid_price_2', 'bid_volume_2', 'bid_price_3', 'bid_volume_3', 
                'ask_price_1', 'ask_volume_1', 'ask_price_2', 'ask_volume_2', 'ask_price_3', 'ask_volume_3', 'mid_price', 'profit_and_loss']

# Filter the data for the product SQUID_INK
squid_ink_data = data[data['product'] == 'SQUID_INK']

# Convert timestamp and mid_price to numeric for plotting
squid_ink_data['timestamp'] = pd.to_numeric(squid_ink_data['timestamp'])
squid_ink_data['mid_price'] = pd.to_numeric(squid_ink_data['mid_price'])

# Plot the price of SQUID_INK
# plt.figure(figsize=(10, 6))
# plt.plot(squid_ink_data['timestamp'], squid_ink_data['mid_price'], label='SQUID_INK Price', color='blue')
# plt.xlabel('Timestamp')
# plt.ylabel('Price')
# plt.title('Price of SQUID_INK Over Time')
# plt.legend()
# plt.grid()
# plt.show()

# Calculate the rolling average of past 5 values
squid_ink_data['rolling_avg_5'] = squid_ink_data['mid_price'].rolling(window=5).mean()

# Plot the original price and the rolling average
# plt.figure(figsize=(12, 6))
# plt.plot(squid_ink_data['timestamp'], squid_ink_data['mid_price'], label='SQUID_INK Mid Price', alpha=0.6)
# plt.plot(squid_ink_data['timestamp'], squid_ink_data['rolling_avg_5'], label='5-Period Moving Average', linewidth=2)
# plt.xlabel('Timestamp')
# plt.ylabel('Price')
# plt.title('SQUID_INK Price and 5-Period Moving Average')
# plt.legend()
# plt.grid(True)
# plt.show()

# Drop rows with NaN values in rolling_avg_5
valid_data = squid_ink_data.dropna(subset=['rolling_avg_5'])

# Calculate the squared differences
squared_diff = (valid_data['mid_price'] - valid_data['rolling_avg_5']) ** 2

# Calculate the MSE
mse_rolling_avg = squared_diff.mean()

print(f"Mean Squared Error (MSE) for 5-period Moving Average model: {mse_rolling_avg:.4f}")

# Optionally calculate RMSE
rmse_rolling_avg = np.sqrt(mse_rolling_avg)
print(f"Root Mean Squared Error (RMSE) for 5-period Moving Average model: {rmse_rolling_avg:.4f}")

# Convert mid_price to a numpy array for FFT
prices = squid_ink_data['mid_price'].to_numpy()

# Calculate the actual sampling rate based on timestamps
time_diff = squid_ink_data['timestamp'].diff().mean()
sampling_rate = 1 / time_diff if time_diff > 0 else 1
print(f"Effective sampling rate: 1 sample per {time_diff} time units")

# Apply Hanning window to reduce spectral leakage
window = np.hanning(len(prices))
windowed_prices = prices * window

# Compute the FFT with proper normalization
fft_result = np.fft.fft(windowed_prices) / len(prices)
# Get the frequencies corresponding to the FFT result
N = len(prices)
freq = np.fft.fftfreq(N, d=time_diff)  # Using actual sampling interval

# Plot the FFT magnitude on a log scale
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
# Only plot positive frequencies on a normal scale
magnitude = np.abs(fft_result[:N//2])
plt.semilogy(freq[:N//2], magnitude)
plt.title('FFT Magnitude Spectrum of SQUID_INK Prices (Log Scale)')
plt.xlabel('Frequency (1/time unit)')
plt.ylabel('Magnitude (log scale)')
plt.grid(True)

# Identify dominant frequencies
dominant_indices = np.argsort(magnitude)[-5:]  # Top 5 frequencies
for idx in dominant_indices:
    if idx > 0:  # Skip DC component
        period = 1 / freq[idx]
        plt.axvline(x=freq[idx], color='r', linestyle='--', alpha=0.3)
        plt.text(freq[idx], magnitude[idx], f" Period: {period:.1f}", 
                 verticalalignment='bottom')

# For DTFT, we'll use a more efficient method focusing on lower frequencies
dtft_freq = np.linspace(-200,200, 100000)  # Focus on lower frequencies (0 to 0.2)
dtft_result = np.zeros(len(dtft_freq), dtype=complex)

# More efficient DTFT computation
n_array = np.arange(N)
for i, f in enumerate(dtft_freq):
    dtft_result[i] = np.sum(prices * np.exp(-2j * np.pi * f * n_array))

# Normalize DTFT
dtft_result /= N

# Plot the DTFT magnitude
plt.subplot(2, 1, 2)
plt.semilogy(dtft_freq, np.abs(dtft_result))
plt.title('DTFT Magnitude Spectrum of SQUID_INK Prices (Low Frequencies)')
plt.xlabel('Normalized Frequency')
plt.ylabel('Magnitude (log scale)')
plt.grid(True)

# Add interpretation text
plt.figtext(0.5, 0.01, 
            "Peaks in the spectrum indicate cyclical patterns in price data.\n" + 
            "Frequency = 0.1 corresponds to a cycle every 10 time units.", 
            ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()

# Optional: Add autocorrelation analysis which can be easier to interpret
# plt.figure(figsize=(10, 5))
# acf = np.correlate(prices - np.mean(prices), prices - np.mean(prices), mode='full')
# acf = acf[N-1:] / acf[N-1]  # Normalize
# lags = np.arange(len(acf))
# plt.stem(lags, acf, markerfmt='ro', basefmt='b-')
# plt.title('Autocorrelation of SQUID_INK Prices')
# plt.xlabel('Lag')
# plt.ylabel('Correlation')
# plt.grid(True)
# plt.show()