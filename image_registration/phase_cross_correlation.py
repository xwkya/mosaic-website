from scipy.fft import fftn, ifftn, fftfreq
import numpy as np

def _compute_error(cross_correlation_max, src_amp, target_amp):
    """
    Compute RMS error metric between ``src_image`` and ``target_image``.
    Parameters
    ----------
    cross_correlation_max : complex
        The complex value of the cross correlation at its maximum point.
    src_amp : float
        The normalized average image intensity of the source image
    target_amp : float
        The normalized average image intensity of the target image
    """
    error = 1.0 - cross_correlation_max * cross_correlation_max.conj() /\
        (src_amp * target_amp)
    return np.sqrt(np.abs(error))

def phase_cross_corr(image, moving_image, normalization='phase', return_error=True):
    src_freq = fftn(image)
    target_freq = fftn(moving_image)

    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    if normalization == "phase":
        eps = np.finfo(image_product.real.dtype).eps
        image_product /= np.maximum(np.abs(image_product), 100 * eps)
    
    cross_correlation = ifftn(image_product)
    # Locate maximum
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])
    corr_values = np.abs(cross_correlation)
    corr_values[:,:int(midpoints[1]+1),:] = -1

    maxima = np.unravel_index(np.argmax(corr_values),
                              cross_correlation.shape)

    float_dtype = image_product.real.dtype

    shifts = np.stack(maxima).astype(float_dtype, copy=False)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if return_error:
        src_amp = np.sum(np.real(src_freq * src_freq.conj()))
        src_amp /= src_freq.size
        target_amp = np.sum(np.real(target_freq * target_freq.conj()))
        target_amp /= target_freq.size
        CCmax = cross_correlation[maxima]
    
        return shifts, _compute_error(CCmax, src_amp, target_amp)
    
    else: return shifts
