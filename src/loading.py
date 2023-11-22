from torchaudio.functional import resample
from torchaudio import load
from torch import zeros
from time import sleep

from torch import Tensor
from typing import Union


def enforce_length(
    waveform, sample_rate: int, enforce_length_s: Union[int, float]
) -> Tensor:
    """Ensures that waveform is (enforce_length_s) seconds long

    Parameters
    ----------
    waveform : torch.Tensor
        1d list, containing waveform

    sample_rate : int
        Sample rate of waveform

    enforce_length_s: [int, float]
        Length (in seconds) to which trim or pad loaded waveform

    Returns
    -------
    waveform : torch.Tensor
        Loaded waveform
    """
    if not isinstance(waveform, Tensor):
        raise Exception("Input waveform is not a torch.Tensor instance")
    if len(waveform) <= 0:
        raise Exception("Waveform is empty")
    if len(enforce_length_s) <= 0:
        raise Exception("Cannot enforce length less than 0")
    if sample_rate <= 0:
        raise Exception("Sample rate is less or equal 0")

    length = sample_rate * enforce_length_s

    if len(waveform) == length:
        return waveform

    elif len(waveform) > length:
        return waveform[0:length]

    elif len(waveform) < length:
        tmp = zeros(length)
        tmp[0 : len(waveform)] = waveform
        return tmp


def load_audio(
    source,
    mono: bool = True,
    sample_rate: Union[int, None] = None,
    enforce_length_s: Union[int, float, None] = None,
    **kwargs
) -> Tensor:
    """Loads waveform, then resamples it and converts to mono, if needed

    Parameters
    ----------
    source : str
        A source to audio file

    mono : bool, optional
        To convert audio to mono
        By default equal True

    sample_rate : Union[int, None], optional
        Sample rate to which resample audio, if it has different sample rate
        If None, does not resample audio
        By default equal None

    enforce_length_s: [int, None], optional
        Length (in seconds) to which trim or pad loaded waveform
        If None, then does not enforce
        By default equals None

    kwargs
        Parameters for torchaudio.load

    Returns
    -------
    waveform : torch.Tensor
        Loaded waveform
    """

    # try to load waveform 3 times with interval of 0.5 seconds
    for i in range(3):
        try:
            waveform, rate = load(source, **kwargs)
            break
        except:
            sleep(0.5)

    # throw an error if data is abscent at given location
    if waveform is None:
        raise Exception("File at source does not contain any data or does not exist")

    # check if resampling is needed
    # if needed, then resample to sample_rate, if waveform has different sample rate
    if sample_rate is not None and rate != sample_rate:
        waveform = resample(waveform, orig_freq=rate, new_freq=sample_rate)
        rate = sample_rate

    # convert waveform to mono, if it is in stereo
    if mono is not None and len(waveform.shape) > 1:
        waveform = waveform.mean(dim=0)

    if enforce_length_s is not None:
        waveform = enforce_length_s(waveform, rate)

    return waveform
