'''
Misc data pipelines for training
'''
import numpy as np

def generate_square_pulses(
    length: int = 128,
    pulse_width: int = 8,
    amplitude: float = 1.0,
    batch_size: int = 1
) -> np.ndarray:
    """
    Generates a batch of 1D square pulses, each with a random shift.

    Args:
        length (int): The total length of each 1D signal array.
        pulse_width (int): The width of the square pulse in data points.
        amplitude (float): The amplitude (height) of the pulse.
        batch_size (int): The number of signals to generate in the batch.

    Returns:
        np.ndarray: A 2D NumPy array of shape (batch_size, length).
    
    Raises:
        ValueError: If pulse_width is greater than the signal length.

    @author Gemini 2.5Pro
    """
    if pulse_width > length:
        raise ValueError("Pulse width cannot be greater than the total signal length.")

    # 1. Create a silent batch of signals (a 2D array of zeros)
    signals = np.zeros((batch_size, length))

    # 2. For each signal in the batch, create a pulse at a random location
    for i in range(batch_size):
        # Determine a random starting point for the pulse in this specific signal
        # The maximum possible start index is length - pulse_width to ensure
        # the pulse fits entirely within the signal array.
        max_start_index = length - pulse_width
        start_index = np.random.randint(0, max_start_index + 1)
        
        # "Turn on" the pulse for the calculated duration
        end_index = start_index + pulse_width
        signals[i, start_index:end_index] = amplitude
    
    return signals
