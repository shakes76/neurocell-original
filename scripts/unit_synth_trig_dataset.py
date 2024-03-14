'''
Unit test to generate simple trignometry dataset for classification.
'''
import jax
import jax.numpy as jnp

def trigSignalGenerator(batch_size, freqs, amps, num_samples, srate, add_noise=True):
    '''
    Generates a random sine signal as per args:
    a freq and amp is randomly chosen from provided freqs and amps at the sample rate 
    with total number of samples
    '''
    # indices = jnp.arange(len(freqs))
    categories = len(freqs)
    time = num_samples/sample_rate # in seconds
    # time = jnp.full(batch_size, num_samples/sample_rate) # in seconds
    # num_samples = jnp.full(batch_size, num_samples) # second

    # choices = jax.random.choice(jax.random.key(1), categories, )
    choices = jax.random.randint(jax.random.key(1), (batch_size,), 0, categories)
    print(choices)

    #create each wave for each member of lists freqs and amplitudes
    t = jnp.linspace(0, time, num_samples)
    t = jnp.tile(t, (batch_size,1))
    # print(i.shape, i)
    
    print(amps[choices])
    waves = jnp.array([amps[choices] * jnp.sin(freqs[choices] * i * 2*jnp.pi) for i in t[choices]]) #2D array of the waves
    # waves = amps[choices] * jnp.sin(freqs[choices] * t[choices] * 2*jnp.pi) #2D array of the waves

    return waves

#three waves
batch_size = 8
freqs = jnp.array([0.5, 1, 2.0]) #Hz - cycles per second
amplitudes = jnp.array([1.0, 0.5, 1.5])
time_to_plot = 2 # second
sample_rate = 10 # samples per second
num_samples = sample_rate * time_to_plot

#create each wave for each member of lists freqs and amplitudes
waves = trigSignalGenerator(batch_size, freqs, amplitudes, num_samples, sample_rate) #2D array of the waves
print(waves.shape)

import matplotlib.pyplot as plt

no_signals = len(freqs)
fig, ax = plt.subplots(nrows=no_signals, ncols=1, figsize=(16, 10))

for number, wave in enumerate(waves[0].T): #transpose to allow traversing columns instead of rows
    ax[number].plot(t, wave, '-')
    ax[number].axis('off')
    ax[number].set_title('Wave '+str(number))

plt.show()
