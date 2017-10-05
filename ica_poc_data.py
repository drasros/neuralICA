# Generate sythetic signals for ICA demo

# Two base waveforms (sine wave + sawtooth)
# 
# * Choose phases
# * Mix them with a random matrix
# * Normalize EACH channel (contrary to usual ICA where
# PCA is performed) and feed to neural net
# * train to MAXIMIZE non-gaussianity

# * See if we get independent components
# * See if this trained model works on other waveforms

# See if if works on training set of larger diversity
# (more frequencies and shapes)

# in_size=512

# TODO: TEST normalizing data by /(2*std) rathen than /std
# Too high input variance may bring instability! (nans for 'normal')
# learning rates. This could allow higher LRs... TO TRY

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# TODO: vectorize on BATCH
def sine_wave(k, period, phase):
    # k: list of ints
    # period: (batch,), in number of samples
    # phase: (batch,), in number of samples
    signal_len = len(k)
    batch_size = len(period)

    k = np.reshape(k, (1, signal_len))
    period = np.reshape(period, (batch_size, 1))
    phase = np.reshape(phase, (batch_size, 1))
    if np.any(phase > period):
        raise ValueError('phases and periods must ' + \
                         'verify: 0<=phase<period')
    y = np.sin(2*np.pi*(k+phase)/period)
    m = np.mean(y, axis=1, keepdims=True)
    v = np.var(y, axis=1, keepdims=True)
    y  = (y - m) / np.sqrt(1e-8 + v)
    return y


def sawtooth_wave(k, period, phase):
    # k: list of ints
    # period: (batch,), in number of samples
    # phase: (batch,), in number of samples
    signal_len = len(k)
    batch_size = len(period)

    k = np.reshape(k, (1, signal_len))
    period = np.reshape(period, (batch_size, 1))
    phase = np.reshape(phase, (batch_size, 1))
    if np.any(phase > period):
        raise ValueError('phases and periods must ' + \
                         'verify: 0<=phase<period')
    max_v = 1. # we don't really care because we will normalize
    y = max_v * (k+phase)/period
    y = y - np.floor(y / max_v) * max_v
    m = np.mean(y, axis=1, keepdims=True)
    v = np.var(y, axis=1, keepdims=True)
    y  = (y - m) / np.sqrt(1e-8 + v)
    return y

def get_random_M(batch_size, min_mix_coeff, min_Mlines_angle, loops_done=0):
    # sample a mixing matrix that will be used for mixing 
    # the sources
    #
    # To make sure that we are in mixing conditions that are
    # ok for ICA, it is possible to use a min_mix_coeff (in 
    # the interval [0., 1.]). Coeffs of the mixing matrix will be drawn from
    # the interval [-1, -min_mix_coeff]U[min_mix_coeff, 1], instead of
    # the interval [-1, 1]. 
    #
    # It is also possible to specify a min_Mlines_angle. If min_Mlines_angle
    # is not 0, the vectors constituted by the lines of M will 
    # have at least this angle between them

    # CAREFUL to not use a too big angle otherwise the
    # condition will never be satisfied for 1.2 batch_size
    # and the function will loop forever!!! (alert added for this.)

    # This function is made for 2 dimensions only, so far. 
    if loops_done >= 3:
        raise ValueError("Warning: already 3 loops done. "
                         " The chosen min_Mlines_angle is probably too high...")

    batch_size_temp = (batch_size if min_Mlines_angle==0. \
        else 2*batch_size)
    M_ = np.random.uniform(min_mix_coeff, 1., size=(batch_size_temp, 1, 2, 2))
    M_vals_list = np.array([-1, 1])
    M_sign = np.random.randint(0, 2, size=(batch_size_temp, 1, 2, 2))
    # low=0, high=2 because high is exclusive
    M_sign = M_vals_list[M_sign]
    M = np.multiply(M_, M_sign)

    # now make sure that there is at least min_Mlines_angle between 
    # the two lines
    # norms of lines of M
    norms_Mlines = np.linalg.norm(M, axis=-1) # (batch_size_temp, 1, 2)
    # dot products
    dotprod_Mlines = np.sum(
        np.prod(M, axis=2),
        axis=-1) # (batch_size_temp, 1)
    cos_angles = np.divide(
        dotprod_Mlines, 
        np.prod(norms_Mlines, axis=-1))
    valid_idxs = np.where(np.abs(cos_angles) < np.cos(min_Mlines_angle))
    valid_idxs = valid_idxs[0]

    M = M[valid_idxs, :, :, :]
    if len(M) >= batch_size:
        return M[:batch_size]
    else: return get_random_M(
        batch_size, min_mix_coeff, min_Mlines_angle, loops_done+1)



def sine_sawtooth_iterator_fixedperiods(batch_size,
                                        in_size,
                                        sine_period,
                                        sawtooth_period,
                                        min_mix_coeff, 
                                        min_Mlines_angle):
    # this iterator is for the experiment where the mixing is
    # variable but component keep the same period over examples

    # To avoid mixing very little of one input component in 
    # output components, one can use the parameter 'min_mix_coeff' (in 
    # the interval [0., 1.])
    # in which case coeffs of the mixing matrix will be drawn from
    # the interval [-1, -min_mix_coeff]U[min_mix_coeff, 1], instead of
    # the interval [-1, 1]

    k = np.arange(in_size)
    sine_period = [sine_period] * batch_size
    sawtooth_period = [sawtooth_period] * batch_size
    while True:
        sine_phase = np.random.randint(
            sine_period[0], size=(batch_size,))
        sawtooth_phase = np.random.randint(
            sawtooth_period[0], size=(batch_size,))
        sine_y = sine_wave(k, sine_period, sine_phase)
        sawtooth_y = sawtooth_wave(k, sawtooth_period, sawtooth_phase)
        in_chan = np.stack([sine_y, sawtooth_y], axis=-1) # shape (batch, in_size, 2)
        in_chan = np.reshape(in_chan, (batch_size, in_size, 2, 1))
        # Mixing matrix
        M = get_random_M(
            batch_size, min_mix_coeff, min_Mlines_angle)
        # Mix
        out_chan = np.matmul(M, in_chan)
        # center and scale to 0, PER CHANNEL 
        # (contrary to normal ICA where a PCA is done)
        m = np.mean(out_chan, axis=1, keepdims=True)
        v = np.var(out_chan, axis=1, keepdims=True)
        out_chan = (out_chan - m) / np.sqrt(1e-8 + v)
        out_chan = np.reshape(out_chan, (batch_size, in_size, 2))
        yield out_chan

def plot_example_2ch(expl, save_name=None):
        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].plot(expl[:, 0])
        axarr[1].plot(expl[:, 1])
        if save_name is not None:
            plt.savefig(save_name)
            plt.close()

if __name__ == "__main__":
    # ### TEST 
    import os
    if not os.path.exists('ica_demo_results'):
        os.makedirs('ica_demo_results')
    k = np.arange(512)
    s = sine_wave(k, [32]*64, [10]*64)[0]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot(s)
    plt.savefig('ica_demo_results/test.png')

    ### TEST ITERATOR
    batch_size = 128
    in_size = 512
    sine_period = 30
    sawtooth_period = 50
    b_it = sine_sawtooth_iterator_fixedperiods(
        batch_size,
        in_size,
        sine_period,
        sawtooth_period)
    b = next(b_it)
    print(b.shape)

    plot_example_2ch(b[0], "ica_demo_results/mix_0.png")
    plot_example_2ch(b[1], "ica_demo_results/mix_1.png")
    # do loop to estimate speed
    print('testing speed...')
    import time
    t = time.time()
    for _ in range(1000):
        b = next(b_it)
    t = time.time() - t
    print('Iterated 1000 batches in %d seconds' %t)










