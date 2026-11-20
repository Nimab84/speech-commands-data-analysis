"""Deterministic waveform preparation and MFCC/log-mel features for Phase 2."""
from fractions import Fraction
from io import BytesIO
import hashlib
import wave

import numpy as np
from scipy.fft import dct
from scipy.signal import resample_poly


def read_audio(path, expected_hash=None):
    payload = path.read_bytes()
    if expected_hash and hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError(f'Raw file changed since Phase 1: {path.name}')
    with wave.open(BytesIO(payload), 'rb') as stream:
        if stream.getsampwidth() != 2 or stream.getcomptype() != 'NONE':
            raise ValueError('Expected uncompressed PCM16')
        rate, channels, n = stream.getframerate(), stream.getnchannels(), stream.getnframes()
        raw = stream.readframes(n)
    if rate <= 0 or channels < 1 or len(raw) != n*channels*2 or n == 0:
        raise ValueError('Invalid or truncated audio')
    x = np.frombuffer(raw, dtype='<i2').reshape(-1, channels).mean(axis=1)/32768.0
    return x, rate


def prepare_signal(x, rate, cfg):
    """Downmix is handled by read_audio; center real samples before zero padding."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError('A finite nonempty mono waveform is required')
    original_n = len(x)
    if rate != cfg['sample_rate_hz']:
        ratio = Fraction(cfg['sample_rate_hz'], rate)
        x = resample_poly(x, ratio.numerator, ratio.denominator)
    if cfg['remove_dc']:
        x = x-x.mean()
    target = cfg['target_samples']
    before_length = len(x)
    if before_length > target:
        start = (before_length-target)//2
        x = x[start:start+target]
    else:
        x = np.pad(x, (0,target-before_length))
    return x, {'original_samples':original_n, 'original_rate':rate,
               'target_samples':len(x), 'padded_samples':max(0,target-before_length),
               'cropped_samples':max(0,before_length-target),
               'resampled':rate != cfg['sample_rate_hz']}


def mel_filterbank(cfg):
    hz_to_mel = lambda hz:2595*np.log10(1+np.asarray(hz)/700)
    mel_to_hz = lambda mel:700*(10**(np.asarray(mel)/2595)-1)
    edges = mel_to_hz(np.linspace(hz_to_mel(cfg['f_min_hz']), hz_to_mel(cfg['f_max_hz']),cfg['n_mels']+2))
    frequencies = np.fft.rfftfreq(cfg['n_fft'],1/cfg['sample_rate_hz'])
    filters = np.maximum(0,np.minimum(
        (frequencies[None,:]-edges[:-2,None])/(edges[1:-1,None]-edges[:-2,None]),
        (edges[2:,None]-frequencies[None,:])/(edges[2:,None]-edges[1:-1,None])))
    if not np.all(filters.sum(axis=1)>0):
        raise ValueError('Empty mel band; increase FFT resolution')
    return filters, frequencies


def feature_names(cfg):
    mfcc = [f'mfcc_{i:02}_{stat}' for stat in ['mean','std','delta_mean','delta_std'] for i in range(cfg['n_mfcc'])]
    mel = [f'logmel_{i:02}_{stat}' for stat in ['mean','std'] for i in range(cfg['n_mels'])]
    domain = ['duration_s','rms_dbfs','peak_abs','quiet_fraction','zero_crossing_rate',
              'centroid_hz','bandwidth_hz','rolloff85_hz','spectral_flatness','spectral_flux']
    return {'mfcc':mfcc,'logmel':mel,'domain':domain}


def extract_features(x, original_duration, cfg, bank=None):
    """25 ms/10 ms, no frame centering; population standard deviations."""
    filters, frequencies = bank if bank is not None else mel_filterbank(cfg)
    w,hop = cfg['window_samples'],cfg['hop_samples']
    plain_frames = np.lib.stride_tricks.sliding_window_view(x,w)[::hop]
    emphasized = np.r_[x[0],x[1:]-cfg['pre_emphasis']*x[:-1]]
    frames = np.lib.stride_tricks.sliding_window_view(emphasized,w)[::hop]
    window = np.hanning(w)
    power = np.abs(np.fft.rfft(frames*window,n=cfg['n_fft'],axis=1))**2/window.sum()**2
    mel_power = power@filters.T
    logmel = 10*np.log10(np.maximum(mel_power,cfg['power_floor']))
    mfcc = dct(logmel,type=2,norm='ortho',axis=1)[:,:cfg['n_mfcc']]
    # Five-frame regression derivative, with replicated edge frames.
    padded = np.pad(mfcc,((2,2),(0,0)),mode='edge')
    delta = (padded[3:-1]-padded[1:-3]+2*(padded[4:]-padded[:-4]))/10
    spectral_sum = power.sum(axis=1)
    weights = power/np.maximum(spectral_sum[:,None],1e-20)
    centroid = weights@frequencies
    bandwidth = np.sqrt(np.sum(weights*(frequencies[None,:]-centroid[:,None])**2,axis=1))
    rolloff = frequencies[(np.cumsum(weights,axis=1)>=.85).argmax(axis=1)]
    flatness = np.exp(np.log(np.maximum(power,1e-20)).mean(axis=1))/np.maximum(power.mean(axis=1),1e-20)
    flatness[spectral_sum<1e-15] = 0
    flux = np.sqrt(np.mean(np.diff(weights,axis=0)**2,axis=1)).mean()
    rms = np.sqrt(np.mean(x*x))
    frame_db = 20*np.log10(np.maximum(np.sqrt(np.mean(plain_frames**2,axis=1)),1e-12))
    domain = [original_duration,20*np.log10(max(rms,1e-12)),np.abs(x).max(),
              np.mean(frame_db<cfg['quiet_frame_dbfs']), np.mean(np.signbit(x[1:])!=np.signbit(x[:-1])),
              centroid.mean(),bandwidth.mean(),rolloff.mean(),flatness.mean(),flux]
    values = np.r_[mfcc.mean(axis=0),mfcc.std(axis=0),delta.mean(axis=0),delta.std(axis=0),
                   logmel.mean(axis=0),logmel.std(axis=0),domain]
    if not np.isfinite(values).all():
        raise ValueError('Non-finite feature; inspect the clip rather than silently impute it')
    return values, {'logmel':logmel, 'mfcc':mfcc, 'frames':len(frames)}


def augment_train(x, split, rng, cfg):
    """Training-only zero-filled shift; retain >=99% energy, cap gain at headroom."""
    if split != 'train':
        raise ValueError('Augmentation is restricted to training recordings')
    max_shift = round(cfg['sample_rate_hz']*cfg['augmentation_shift_ms']/1000)
    energy = float(x@x)
    shift = 0
    y = x.copy()
    for _ in range(12):
        proposal = int(rng.integers(-max_shift,max_shift+1))
        trial = np.zeros_like(x)
        if proposal > 0:
            trial[proposal:] = x[:-proposal]
        elif proposal < 0:
            trial[:proposal] = x[-proposal:]
        else:
            trial[:] = x
        retained = float(trial@trial)/max(energy,1e-20)
        if retained >= cfg['augmentation_min_energy_retained']:
            shift,y = proposal,trial
            break
    requested_db = float(rng.uniform(-cfg['augmentation_gain_db'],cfg['augmentation_gain_db']))
    gain = min(10**(requested_db/20), .99/max(np.abs(y).max(),1e-12))
    retained = float(y@y)/max(energy,1e-20)
    return y*gain, {'shift_samples':shift,'requested_gain_db':requested_db,
                    'effective_gain_db':20*np.log10(gain),'retained_energy_fraction':retained}
