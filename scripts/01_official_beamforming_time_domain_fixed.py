"""
This is a longer example that applies time domain beamforming towards a source
of interest in the presence of a strong interfering source.

Modified for pyroomacoustics 0.10.1:
- ShoeBox no longer accepts absorption=...
- Use materials=pra.Material(energy_absorption=...)
"""

from __future__ import division, print_function

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

import pyroomacoustics as pra
from pyroomacoustics.transform import stft


# =========================
# Spectrogram figure properties
# =========================
figsize = (15, 7)
fft_size = 512
fft_hop = 8
fft_zp = 512
analysis_window = pra.hann(fft_size)
t_cut = 0.83


# =========================
# Simulation parameters
# =========================
Fs = 8000
absorption = 0.1
max_order_sim = 2
sigma2_n = 5e-7


# =========================
# Microphone array design parameters
# =========================
mic1 = np.array([2, 1.5])
M = 8
d = 0.08
phi = 0.0
max_order_design = 1
shape = "Linear"
Lg_t = 0.100
Lg = int(np.ceil(Lg_t * Fs))
delay = 0.050


# =========================
# FFT length
# =========================
N = 1024


# =========================
# Create a microphone array
# =========================
if shape == "Circular":
    R = pra.circular_2D_array(mic1, M, phi, d * M / (2 * np.pi))
else:
    R = pra.linear_2D_array(mic1, M, phi, d)


# =========================
# Paths
# =========================
path = os.path.dirname(__file__)
input_dir = os.path.join(path, "input_samples")
output_dir = os.path.join(path, "output_samples")
figure_dir = os.path.join(path, "figures")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)


# =========================
# Load source of interest
# =========================
rate1, signal1 = wavfile.read(
    os.path.join(input_dir, "singing_" + str(Fs) + ".wav")
)

if rate1 != Fs:
    raise ValueError(f"Expected sampling rate {Fs}, but got {rate1} for singing signal.")

signal1 = np.array(signal1, dtype=float)
signal1 = pra.normalize(signal1)
signal1 = pra.highpass(signal1, Fs)
delay1 = 0.0


# =========================
# Load interfering source
# =========================
rate2, signal2 = wavfile.read(
    os.path.join(input_dir, "german_speech_" + str(Fs) + ".wav")
)

if rate2 != Fs:
    raise ValueError(f"Expected sampling rate {Fs}, but got {rate2} for speech signal.")

signal2 = np.array(signal2, dtype=float)
signal2 = pra.normalize(signal2)
signal2 = pra.highpass(signal2, Fs)
delay2 = 1.0


# =========================
# Create the room
# =========================
room_dim = [4, 6]

# pyroomacoustics 0.10.1 does not accept absorption=...
# Use Material instead.
material = pra.Material(energy_absorption=absorption)

room1 = pra.ShoeBox(
    room_dim,
    materials=material,
    fs=Fs,
    max_order=max_order_sim,
    sigma2_awgn=sigma2_n,
)


# =========================
# Add sources to room
# =========================
good_source = np.array([1, 4.5])
normal_interferer = np.array([2.8, 4.3])

room1.add_source(good_source, signal=signal1, delay=delay1)
room1.add_source(normal_interferer, signal=signal2, delay=delay2)


"""
MVDR direct path only simulation
"""

# compute beamforming filters
mics = pra.Beamformer(R, Fs, N=N, Lg=Lg)
room1.add_microphone_array(mics)
room1.compute_rir()

# Save RIR computed by the official example
rir_array = np.array(room1.rir, dtype=object)
np.save(os.path.join(output_dir, "official_rir.npy"), rir_array, allow_pickle=True)

room1.simulate()

mics.rake_mvdr_filters(
    room1.sources[0][0:1],
    room1.sources[1][0:1],
    sigma2_n * np.eye(mics.Lg * mics.M),
    delay=delay,
)

# process the signal
output = mics.process()

# save to output file
input_mic = pra.normalize(pra.highpass(mics.signals[mics.M // 2], Fs))
wavfile.write(os.path.join(output_dir, "input.wav"), Fs, input_mic)

out_DirectMVDR = pra.normalize(pra.highpass(output, Fs))
wavfile.write(os.path.join(output_dir, "output_DirectMVDR.wav"), Fs, out_DirectMVDR)


"""
Rake MVDR simulation
"""

mics = pra.Beamformer(R, Fs, N, Lg=Lg)
room1.mic_array = mics
room1.compute_rir()
room1.simulate()

good_sources = room1.sources[0][: max_order_design + 1]
bad_sources = room1.sources[1][: max_order_design + 1]

mics.rake_mvdr_filters(
    good_sources,
    bad_sources,
    sigma2_n * np.eye(mics.Lg * mics.M),
    delay=delay,
)

output = mics.process()

out_RakeMVDR = pra.normalize(pra.highpass(output, Fs))
wavfile.write(os.path.join(output_dir, "output_RakeMVDR.wav"), Fs, out_RakeMVDR)


"""
Perceptual direct path only simulation
"""

mics = pra.Beamformer(R, Fs, N, Lg=Lg)
room1.mic_array = mics
room1.compute_rir()
room1.simulate()

mics.rake_perceptual_filters(
    room1.sources[0][0:1],
    room1.sources[1][0:1],
    sigma2_n * np.eye(mics.Lg * mics.M),
    delay=delay,
)

output = mics.process()

out_DirectPerceptual = pra.normalize(pra.highpass(output, Fs))
wavfile.write(
    os.path.join(output_dir, "output_DirectPerceptual.wav"),
    Fs,
    out_DirectPerceptual,
)


"""
Rake Perceptual simulation
"""

mics = pra.Beamformer(R, Fs, N, Lg=Lg)
room1.mic_array = mics
room1.compute_rir()
room1.simulate()

mics.rake_perceptual_filters(
    good_sources,
    bad_sources,
    sigma2_n * np.eye(mics.Lg * mics.M),
    delay=delay,
)

output = mics.process()

out_RakePerceptual = pra.normalize(pra.highpass(output, Fs))
wavfile.write(
    os.path.join(output_dir, "output_RakePerceptual.wav"),
    Fs,
    out_RakePerceptual,
)


"""
Plot all the spectrogram
"""

dSNR = pra.dB(room1.direct_snr(mics.center[:, 0], source=0), power=True)
print("The direct SNR for good source is " + str(dSNR))

# remove a bit of signal at the end
n_lim = int(np.ceil(len(input_mic) - t_cut * Fs))

input_clean = signal1[:n_lim]
input_mic = input_mic[:n_lim]
out_DirectMVDR = out_DirectMVDR[:n_lim]
out_RakeMVDR = out_RakeMVDR[:n_lim]
out_DirectPerceptual = out_DirectPerceptual[:n_lim]
out_RakePerceptual = out_RakePerceptual[:n_lim]


# compute time-frequency planes
F0 = stft.analysis(input_clean, fft_size, fft_hop, win=analysis_window, zp_back=fft_zp)
F1 = stft.analysis(input_mic, fft_size, fft_hop, win=analysis_window, zp_back=fft_zp)
F2 = stft.analysis(
    out_DirectMVDR,
    fft_size,
    fft_hop,
    win=analysis_window,
    zp_back=fft_zp,
)
F3 = stft.analysis(
    out_RakeMVDR,
    fft_size,
    fft_hop,
    win=analysis_window,
    zp_back=fft_zp,
)
F4 = stft.analysis(
    out_DirectPerceptual,
    fft_size,
    fft_hop,
    win=analysis_window,
    zp_back=fft_zp,
)
F5 = stft.analysis(
    out_RakePerceptual,
    fft_size,
    fft_hop,
    win=analysis_window,
    zp_back=fft_zp,
)


# scale setting
p_min = 7
p_max = 100

all_vals = np.concatenate(
    (
        pra.dB(F1 + pra.eps),
        pra.dB(F2 + pra.eps),
        pra.dB(F3 + pra.eps),
        pra.dB(F0 + pra.eps),
        pra.dB(F4 + pra.eps),
        pra.dB(F5 + pra.eps),
    )
).flatten()

vmin, vmax = np.percentile(all_vals, [p_min, p_max])

cmap = "afmhot"
interpolation = "none"

fig, axes = plt.subplots(figsize=figsize, nrows=2, ncols=3)


def plot_spectrogram(F, title):
    plt.imshow(
        pra.dB(F.T),
        extent=[0, 1, 0, Fs / 2],
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        cmap=plt.get_cmap(cmap),
        interpolation=interpolation,
    )
    plt.title(title)
    plt.ylabel("")
    plt.xlabel("")
    plt.gca().set_aspect("auto")
    plt.axis("off")


plt.subplot(2, 3, 1)
plot_spectrogram(F0, "Desired Signal")

plt.subplot(2, 3, 4)
plot_spectrogram(F1, "Microphone Input")

plt.subplot(2, 3, 2)
plot_spectrogram(F2, "Direct MVDR")

plt.subplot(2, 3, 5)
plot_spectrogram(F3, "Rake MVDR")

plt.subplot(2, 3, 3)
plot_spectrogram(F4, "Direct Perceptual")

plt.subplot(2, 3, 6)
plot_spectrogram(F5, "Rake Perceptual")

fig.savefig(os.path.join(figure_dir, "spectrograms.png"), dpi=150)

print("Official example finished.")
print("Generated WAV files:")
print(os.path.join(output_dir, "input.wav"))
print(os.path.join(output_dir, "output_DirectMVDR.wav"))
print(os.path.join(output_dir, "output_RakeMVDR.wav"))
print(os.path.join(output_dir, "output_DirectPerceptual.wav"))
print(os.path.join(output_dir, "output_RakePerceptual.wav"))
print("Generated figure:")
print(os.path.join(figure_dir, "spectrograms.png"))

plt.show()