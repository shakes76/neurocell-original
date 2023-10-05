# -*- coding: utf-8 -*-
"""
Discrete filters module module

Notes:
Integer values are used for the kernels and they are not normalised in general
"""
import jax.numpy as jnp

def kernelSmooth(dtype=jnp.int32):
    '''
    Returns the window or smoothing kernel
    '''
    kernel =   [[1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]]
    kernel = jnp.array(kernel, dtype=dtype)
    # print(kernel.shape)
    return kernel
    
def kernelGaussian(dtype=jnp.int32):
    '''
    Returns the Gaussian smoothing kernel
    '''
    kernel =   [[1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]]
    kernel = jnp.array(kernel, dtype=dtype)
    # print(kernel.shape)
    return kernel
    
def kernelSobelVertical(dtype=jnp.int32):
    '''
    Returns the Sobel vertical kernel, usually used for edge detection
    '''
    kernel =   [[1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]]
    kernel = jnp.array(kernel, dtype=dtype)
    # print(kernel.shape)
    return kernel
    
def kernelSobelHorizontal(dtype=jnp.int32):
    '''
    Returns the Sobel horizontal kernel, usually used for edge detection
    '''
    kernel =   [[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]]
    kernel = jnp.array(kernel, dtype=dtype)
    # print(kernel.shape)
    return kernel
    
def kernelEdge1(dtype=jnp.int32):
    '''
    Returns a diagonal edge kernel, usually used for edge detection
    [[1, 0, -1],
    [0, 0, 0],
    [-1, 0, 1]]
    '''
    kernel =   [[1, 0, -1],
                [0, 0, 0],
                [-1, 0, 1]]
    kernel = jnp.array(kernel, dtype=dtype)
    # print(kernel.shape)
    return kernel
    
def kernelEdge2(dtype=jnp.int32):
    '''
    Returns a center diagonal edge kernel, usually used for edge detection
    [[1, 0, -1],
    [0, -4, 0],
    [-1, 0, 1]]
    '''
    kernel =   [[1, 0, 1],
                [0, -4, 0],
                [1, 0, 1]]
    kernel = jnp.array(kernel, dtype=dtype)
    # print(kernel.shape)
    return kernel
    
def kernelEdge3(dtype=jnp.int32):
    '''
    Returns an edge kernel, usually used for edge detection
    [[-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]]
    '''
    kernel =   [[-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]]
    kernel = jnp.array(kernel, dtype=dtype)
    # print(kernel.shape)
    return kernel
    
def kernelHaar(dtype=jnp.int32):
    '''
    Returns the Haar kernel, usually used for edge detection
    [[1, 1],
    [1, -1]]
    '''
    kernel =   [[1, 1],
                [1, -1]]
    kernel = jnp.array(kernel, dtype=dtype)
    # print(kernel.shape)
    return kernel
    
def kernelSharpen(dtype=jnp.int32):
    '''
    Returns the sharpen kernel, usually used for edge enhancement
    [[0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]]
    '''
    kernel =   [[0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]]
    kernel = jnp.array(kernel, dtype=dtype)
    # print(kernel.shape)
    return kernel

def filter(N, filterType='smooth', x0 = 0, y0 = 0):
    '''
    Returns an NxN image of the kernel requested by filterType translated by (x0,y0)
    filterType
    - smooth or window (default)
    - gauss
    - sobelV or sobelH
    - edge1
    - edge2
    - edge3
    - haar
    - sharpen
    '''
    #create kernel
    kernelImage = jnp.zeros((N,N), dtype=np.int64)

    if filterType == 'sobelV':
        kernel = kernelSobelVertical()
    elif filterType == 'sobelH':
        kernel = kernelSobelHorizontal()
    elif filterType == 'edge1':
        kernel = kernelEdge1()
    elif filterType == 'edge2':
        kernel = kernelEdge2()
    elif filterType == 'edge3':
        kernel = kernelEdge3()
    elif filterType == 'haar':
        kernel = kernelHaar()
    elif filterType == 'sharpen':
        kernel = kernelSharpen()
    elif filterType == 'gauss':
        kernel = kernelGaussian()
    else:
        kernel = kernelSmooth()
    
    kernelImage[x0:x0+kernel.shape[0], y0:y0+kernel.shape[1]] = kernel
    # print(kernelImage)
    return kernelImage
