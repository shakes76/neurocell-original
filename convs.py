'''
Conv module for all convolution algorithms
'''
import jax
import jax.numpy as jnp

#JIT compiled members for performance
from functools import partial

@partial(jax.jit, static_argnums=(0,1))
def _fft_1D(N, length, filters):
    """
    2D conv done in 1D convs via the CCP and Fourier space.

    Cells is assumed to be the relevant sub-region to process
    """
    #pad to length of layers
    filters = jnp.pad(filters, ((0,(N-length)), (0,0)), mode='constant')
    filters = jnp.roll(filters, -1, axis=0) #make sure centre pixel is first in filter by periodic shift
    # print("filters:", filters.shape, filters)
    # FFTs of filter
    return jnp.fft.fft(filters, axis=0) #ffts of rows

@partial(jax.jit, static_argnums=(2,3,4,5))
def _filter(cells, nethood, dn, window_strides, lhs_dilation=None, rhs_dilation=None):
    """
    Filter the array provided by filter provided.
    """
    #get weighted sum of neighbors using convolution
    cells = jax.lax.conv_general_dilated(cells[jnp.newaxis,...,jnp.newaxis], # lhs = NHWC image tensor
                                        nethood,  # rhs = OHWI conv kernel tensor
                                        window_strides,  # window strides
                                        'SAME', # padding mode
                                        lhs_dilation,  # lhs/image dilation
                                        rhs_dilation,  # rhs/kernel dilation
                                        dn)     # dimension_numbers = lhs, rhs, out dimension permutation
    return cells[0,...,0] # NHWC

# conv operators
@partial(jax.jit, static_argnums=(2,3,4,5,6))
def _fast_conv(sub_cells, filters, dn, window_strides, padding_mode="SAME", lhs_dilation=None, rhs_dilation=None):
    """
    nD conv done with filters with JAX builtin general conv func. Supports multiple filters in channel dim.
    Assumes sub_cells has matching channel dim to filters
    The dims are defined by dn and strides

    Cells is assumed to be the relevant sub-region to process
    """
    #Conv via 2D brute force
    #get weighted sum of neighbors using convolution
    sub_cells_conv = jax.lax.conv_general_dilated(sub_cells[jnp.newaxis,...], # lhs = NHWC image tensor
                                        filters,  # rhs = OHWI conv kernel tensor
                                        window_strides,  # window strides
                                        padding_mode, # padding mode
                                        lhs_dilation,  # lhs/image dilation
                                        rhs_dilation,  # rhs/kernel dilation
                                        dn)     # dimension_numbers = lhs, rhs, out dimension permutation
    return sub_cells_conv[0] # NHWC, remove extra batch dim, return multi-filters result

@partial(jax.jit, static_argnums=())
def _fast_conv_1D(sub_cells, filter_ffts):
    """
    2D conv done in 1D convs via the CCP and Fourier space.

    Cells is assumed to be the relevant sub-region to process
    """
    #Conv via CCP
    # FFTs of universe
    cells_conv1D = jnp.fft.fft(sub_cells, axis=0) #ffts of rows
    #convolve in Fourier space
    result_conv1D = filter_ffts * cells_conv1D
    # print("i:", i, sub_universe.shape, sub_universe)
    result_conv1D = jnp.sum(result_conv1D, axis=-1).reshape(-1,1) #reshape promotes to column vec
    # print(result_conv1D.shape, result_conv1D)
    #inverse FFT
    result_block = jnp.fft.ifft(result_conv1D, axis=0) #ffts of rows

    return jnp.real(result_block)

@partial(jax.jit, static_argnums=(2,3,4,5))
def _fast_conv_2D(image, kernel, epsilon=1e-7, apply_antialias=False, aa_cutoff=0.2, aa_order=3):
    """
    Performs a 2D convolution using FFT with proper phase alignment,
    numerical noise suppression, and optional anti-aliasing.

    Args:
        image: 2D JAX array representing the image.
        kernel: 2D JAX array representing the filter.
        epsilon: Threshold to zero-out microscopic floating-point noise.
        apply_antialias: If True, applies a Butterworth low-pass filter in the
                         frequency domain to prevent frequency aliasing.
        aa_cutoff: Cutoff frequency for the Butterworth filter (0.0 to 0.5).
        aa_order: Order of the Butterworth filter. Higher means steeper cutoff
                  but more potential for spatial ringing.

    Returns:
        2D JAX array of the convolved image, same shape as input image.

    @author Gemini Pro 3.1
    """
    img_shape = image.shape
    ker_shape = kernel.shape

    # 1. Pad for linear convolution to prevent spatial aliasing (circular wraparound)
    fft_shape = (img_shape[0] + ker_shape[0] - 1, img_shape[1] + ker_shape[1] - 1)

    padded_image = jnp.pad(
        image,
        ((0, fft_shape[0] - img_shape[0]), (0, fft_shape[1] - img_shape[1])),
        mode='constant'
    )

    # 2. Pad the kernel to match the fft_shape, placing it dead center.
    pad_h_before = (fft_shape[0] - ker_shape[0]) // 2
    pad_h_after = fft_shape[0] - ker_shape[0] - pad_h_before

    pad_w_before = (fft_shape[1] - ker_shape[1]) // 2
    pad_w_after = fft_shape[1] - ker_shape[1] - pad_w_before

    padded_kernel = jnp.pad(
        kernel,
        ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after)),
        mode='constant'
    )

    # 3. Apply ifftshift to the padded spatial kernel to zero the phase.
    shifted_kernel = jnp.fft.ifftshift(padded_kernel)

    # 4. Transform to frequency domain using Real FFT
    image_freq = jnp.fft.rfft2(padded_image)
    kernel_freq = jnp.fft.rfft2(shifted_kernel)

    # 5. Point-wise multiplication in the frequency domain
    result_freq = image_freq * kernel_freq

    # 6. Apply optional frequency-domain anti-aliasing (Butterworth low-pass)
    # This prevents high-frequency data from folding over if subsampling later
    if apply_antialias:
        fy = jnp.fft.fftfreq(fft_shape[0])
        fx = jnp.fft.rfftfreq(fft_shape[1])
        FY, FX = jnp.meshgrid(fy, fx, indexing='ij')

        # Squared normalized frequency
        F_sq = FY**2 + FX**2

        # Butterworth low-pass filter
        # Maximally flat passband with controllable roll-off
        aa_filter = 1.0 / (1.0 + (F_sq / (aa_cutoff**2))**aa_order)
        result_freq = result_freq * aa_filter

    # 7. Inverse Real FFT to return to the spatial domain
    conv_result = jnp.fft.irfft2(result_freq, s=fft_shape)

    # 8. Crop back to the original image shape (handles the 'same' mode cropping safely)
    cropped_result = conv_result[:img_shape[0], :img_shape[1]]

    # 9. Handle Epsilon: Suppress microscopic floating-point numerical noise
    cleaned_result = jnp.where(jnp.abs(cropped_result) < epsilon, 0.0, cropped_result)

    return cleaned_result

@partial(jax.jit, static_argnums=(2,3,4,5,6))
def _fast_conv_3D(volume, kernel, epsilon=1e-7, suppress_epsilons=False, apply_antialias=False, aa_cutoff=0.4, aa_order=3):
    """
    Performs a 3D convolution using FFT with proper phase alignment,
    numerical noise suppression, and optional anti-aliasing.

    Args:
        volume: 3D JAX array representing the volume (Z, Y, X).
        kernel: 3D or 4D JAX array. If 4D, expects (Z, Y, X, C) for multi-channel CNN-like output.
        epsilon: Threshold to zero-out microscopic floating-point noise.
        suppress_epsilons: If True, applies the epsilon threshold to zero-out noise.
        apply_antialias: If True, applies a Butterworth low-pass filter in the
                         frequency domain to prevent frequency aliasing.
        aa_cutoff: Cutoff frequency for the Butterworth filter (0.0 to 0.5).
        aa_order: Order of the Butterworth filter.

    Returns:
        JAX array of the convolved volume. Shape matches volume if kernel is 3D,
        or (Z, Y, X, C) if kernel is 4D.
    """
    # Track original dimensions for backward compatibility
    # was_3d = kernel.ndim == 3
    # if was_3d:
    #     kernel = kernel[..., None] # Convert to (Z, Y, X, 1)
    fft_shape = volume.shape

    # Transform to frequency domain using N-Dimensional Real FFT
    vol_freq = jnp.fft.rfftn(volume)
    ker_freq = jnp.fft.rfftn(kernel, s=fft_shape, axes=(0, 1, 2)) #handles pad to fft_shape

    # Point-wise multiplication in the frequency domain
    # vol_freq is (Z, Y, X_freq). ker_freq is (Z, Y, X_freq, C).
    # We expand vol_freq's dimensions to broadcast across the channels.
    result_freq = vol_freq[..., None] * ker_freq

    # Apply optional 3D frequency-domain anti-aliasing (Butterworth low-pass)
    if apply_antialias:
        fz = jnp.fft.fftfreq(fft_shape[0])
        fy = jnp.fft.fftfreq(fft_shape[1])
        fx = jnp.fft.rfftfreq(fft_shape[2])  # Real FFT only goes up to Nyquist on the last axis
        FZ, FY, FX = jnp.meshgrid(fz, fy, fx, indexing='ij')

        # 3D Squared normalized frequency
        F_sq = FZ**2 + FY**2 + FX**2

        # 3D Butterworth low-pass filter (expanded to broadcast across channels)
        aa_filter = 1.0 / (1.0 + (F_sq / (aa_cutoff**2))**aa_order)
        result_freq = result_freq * aa_filter[..., None]

    # Inverse N-Dimensional Real FFT to return to the spatial domain
    conv_result = jnp.fft.irfftn(result_freq, s=fft_shape, axes=(0, 1, 2))

    # Handle Epsilon: Optionally suppress microscopic floating-point numerical noise
    if suppress_epsilons:
        cleaned_result = jnp.where(jnp.abs(conv_result) < epsilon, 0.0, conv_result)
    else:
        cleaned_result = conv_result

    # if was_3d:
    #     return cleaned_result[..., 0]

    return cleaned_result
