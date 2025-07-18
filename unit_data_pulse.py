'''
Test for XOR data generation
'''
import matplotlib.pyplot as plt
from data import generate_square_pulses

signal_length = 128
pulse_width = 8
amp = 1.0
batch_size = 32
plot_size = batch_size//8

# Generate a batch of pulses
pulse_batch = generate_square_pulses(
    length=signal_length,
    pulse_width=pulse_width,
    amplitude=amp,
    batch_size=batch_size
)

print(f"Shape of the generated batch: {pulse_batch.shape}")

# --- Plotting the results for visualization ---
# Create a figure to display the first few examples from the batch
fig, axes = plt.subplots(
    plot_size, 1, 
    figsize=(12, 8), 
    sharex=True, 
    sharey=True
)
fig.suptitle('1D Square Pulse Batch Generator with Random Shifts', fontsize=16)

for i, ax in enumerate(axes):
    ax.plot(pulse_batch[i])
    ax.set_title(f"Pulse Example {i+1}")
    ax.grid(True, linestyle='--', alpha=0.6)

# Set common labels and limits
axes[-1].set_xlabel("Sample Index")
fig.text(0.06, 0.5, 'Amplitude', va='center', rotation='vertical')
plt.ylim(-amp * 0.1, amp * 1.1)

plt.tight_layout(rect=[0.07, 0.03, 1, 0.95])
plt.show()
