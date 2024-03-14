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
    time = num_samples/srate # in seconds

    # choices = jax.random.choice(jax.random.key(1), categories, )
    choices = jax.random.randint(jax.random.key(1), (batch_size,), 0, categories)
    # print("choices:",choices)

    #create each wave for each member of lists freqs and amplitudes
    t = jnp.linspace(0, time, num_samples)
    t = jnp.tile(t, (batch_size,1))
    # print(i.shape, i)
    freqs = freqs.reshape((1,categories))
    amps = amps.reshape((1,categories))
    print(freqs)
    freqs = jnp.tile(freqs.T, (1,num_samples))
    amps = jnp.tile(amps.T, (1,num_samples))
    # print("new freqs",freqs)

    #noise
    noise = 0.02 * jax.random.normal(jax.random.key(1), (batch_size, num_samples))
    # noise = jax.random.normal(jax.random.key(1))
    print("noise:",noise.shape)
    
    # print("choosen:",amps[choices])
    waves = amps[choices] * jnp.sin(freqs[choices] * t[choices] * 2*jnp.pi) + noise #2D array of the waves

    return waves, choices

#three waves
batch_size = 4
freqs = jnp.array([0.5, 1, 2.0]) #Hz - cycles per second
amplitudes = jnp.array([1.0, 0.5, 1.5])
time_to_plot = 1 # second
sample_rate = 100 # samples per second
num_samples = sample_rate * time_to_plot

#create each wave for each member of lists freqs and amplitudes
waves, labels = trigSignalGenerator(batch_size, freqs, amplitudes, num_samples, sample_rate) #2D array of the waves
print("batch size:", batch_size)
print("data size:",waves.shape)
print("labels:",labels)
print("freqs chosen:",freqs[labels])

import matplotlib.pyplot as plt

no_signals = len(freqs)
fig, ax = plt.subplots(nrows=batch_size, ncols=1, figsize=(16, 10))

t = jnp.linspace(0, time_to_plot, num_samples)
for number, wave in enumerate(waves): #transpose to allow traversing columns instead of rows
    ax[number].plot(t, wave, '-')
    ax[number].axis('off')
    ax[number].set_title('Wave '+str(labels[number]))

plt.show()
