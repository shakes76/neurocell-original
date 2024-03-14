'''
Trignometry sythn data pipeline for simple NN training
'''
import jax
import jax.numpy as jnp

def trigSignalGenerator(batch_size, freqs, amps, num_samples, srate, key, add_noise=True):
    '''
    Generates a random sine signal as per args:
    a freq and amp is randomly chosen from provided freqs and amps at the sample rate 
    with total number of samples
    '''
    # indices = jnp.arange(len(freqs))
    categories = len(freqs)
    time = num_samples/srate # in seconds

    new_key, subkey = jax.random.split(key)
    # choices = jax.random.choice(jax.random.key(1), categories, )
    choices = jax.random.randint(subkey, (batch_size,), 0, categories)
    # print("choices:",choices)

    #create each wave for each member of lists freqs and amplitudes
    t = jnp.linspace(0, time, num_samples)
    t = jnp.tile(t, (batch_size,1))
    # print(i.shape, i)
    freqs = freqs.reshape((1,categories))
    amps = amps.reshape((1,categories))
    # print(freqs)
    freqs = jnp.tile(freqs.T, (1,num_samples))
    amps = jnp.tile(amps.T, (1,num_samples))
    # print("new freqs",freqs)

    #noise
    noise = 0
    if add_noise:
        new_key2, subkey2 = jax.random.split(new_key)
        noise = 0.001 * jax.random.normal(subkey2, (batch_size, num_samples))
    # print("noise:",noise.shape)
    
    # print("choosen:",amps[choices])
    waves = amps[choices] * jnp.sin(freqs[choices] * t[choices] * 2*jnp.pi) + noise #2D array of the waves

    return waves, choices

class TrigData:
    def __init__(self, batch_size, freqs, amps, num_samples, srate, key, add_noise=True):
        self.key = key
        # self.new_key, self.subkey = jax.random.split(key)
        self.batch_size = batch_size
        self.freqs = freqs
        self.amps = amps
        self.num_samples = num_samples
        self.srate = srate

    def next(self):
        self.key, subkey = jax.random.split(self.key) #ensure use current key and replace old one
        return trigSignalGenerator(self.batch_size, self.freqs, self.amps, self.num_samples, self.srate, subkey)
