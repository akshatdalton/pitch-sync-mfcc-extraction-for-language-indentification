import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt
import librosa


def zero_frequency_resonator(audio):
    result = np.zeros(audio.shape)
    for i in range(2, len(audio)):
        result[i] = -(-2 * result[i-1] + 1 * result[i-2]) + audio[i]
    return result


def mean_substraction(audio, window_length):
    mean_filter = np.ones(window_length) / window_length
    N = (window_length - 1) // 2
    result = audio[N: -N] - np.convolve(audio, mean_filter, mode="valid")
    return result


def extract_epochs(audio, mean_window_length):
    mean_window_length = mean_window_length + (mean_window_length % 2 == 0)
    x = audio.copy()
    x[1:] = np.diff(audio)
    y1 = zero_frequency_resonator(x)
    y2 = zero_frequency_resonator(y1)
    y = y2.copy()
    for i in range(2): y = mean_substraction(y, mean_window_length)
    GCI = np.nonzero(np.diff(np.sign(y)) > 0)[0]
    return y, GCI

def framing(data, window_length, hop_length):
    total_samples = len(data)
    num_frames = 1 + (total_samples - window_length) // hop_length + ((total_samples - window_length) % hop_length != 0)
    frames = np.zeros((num_frames, window_length))
    frame_num = 0
    i = 0
    for i in range(0, total_samples - window_length + 1, hop_length):
        frames[frame_num] = data[i: i+window_length]
        frame_num += 1
    if(frame_num != num_frames):
        i += hop_length
        frames[frame_num][:total_samples-i] = data[i:]
        frame_num += 1
    return frames

def getVoiced(audio, sr):
    np.random.seed(43)
    SNR = 30
    rms_audio = np.sqrt(np.mean(audio**2))
    std_noise = np.sqrt(rms_audio**2 / np.power(10, SNR / 10))
    mean_window_length = 10 * sr // 1000
    mean_window_length = mean_window_length + (mean_window_length % 2 == 0)
    noise = np.random.normal(0, std_noise, audio.shape)
    noised_1 = audio + noise
    _, GCI1 = extract_epochs(noised_1, mean_window_length)
    noise = np.random.normal(0, std_noise, audio.shape)
    noised_2 = audio + noise
    _, GCI2 = extract_epochs(noised_2, mean_window_length)
    win_len = 512
    frames = framing(audio[mean_window_length-1: -mean_window_length+1], win_len, win_len)
    voiced_audio = []
    count=0
    for i, frame in enumerate(frames):
        start = i * win_len
        end = start + win_len
        gci1 = GCI1[(GCI1 >= start) & (GCI1 <= end)]
        gci2 = GCI2[(GCI2 >= start) & (GCI2 <= end)]
        if gci1.shape[0] <= 1 or gci2.shape[0] <= 1 or \
            gci1.shape != gci2.shape or np.mean(np.abs(gci1 - gci2)) > 7:
            continue
        voiced_audio.extend(frame)
        count=count+1
    return np.array(voiced_audio)


def extract_pitch_sync_frames(audio, GCI):
    frames=[]
    for i,epoch in enumerate(GCI[:-1]):
        frames.append(audio[epoch:GCI[i+1]])
    return frames


def get_pitch_sync_frames(audio, sr):
    voiced_audio=getVoiced(audio, sr)
    mean_window_length = 10 * sr // 1000
    y, GCI = extract_epochs(voiced_audio, mean_window_length)
    frames=extract_pitch_sync_frames(voiced_audio, GCI)
    return frames
