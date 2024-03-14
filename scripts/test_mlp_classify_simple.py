'''
Simply classify experiment using MLP network
'''
import jax
import numpy as np
import jax.numpy as jnp

def trigSignalGenerator(freqs, amps, num_samples, srate):
    '''
    Generates a random sine signal as per args
    '''
    time = num_samples/sample_rate # second

    #create each wave for each member of lists freqs and amplitudes
    t = np.linspace(0, time, num_samples)
    waves = np.array([amps * np.sin(freqs * i * 2*np.pi) for i in t]) #2D array of the waves
