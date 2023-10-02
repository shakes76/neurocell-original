'''
Utility functions for library
'''
import jax
import jax.numpy as jnp

def jax_status():
    print("Total Devices:",jax.devices())
    print("Used Devices:",jax.local_devices())
    print("Using", jnp.ones(3).device_buffer.device())

