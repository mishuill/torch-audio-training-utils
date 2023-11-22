from torch.utils.data import Dataset
from torchaudio.functional import resample
from torchaudio import load
from torch import mean, randperm
from time import sleep

from .loading import load_audio

from typing import Union


class AudioDataset(Dataset):
    """Audio dataset
    ...

    Attributes
    ----------
    _sources : list
        Paths to audio files

    _labels : list, optional
        Labels which correspond to provided audio files

    _only_waveform: bool, optional
        To return only waveform (for batch loading)
        By default equals False

    _sample_rate : [int, None], optional
        Sample rate to which resample all loaded audios
        If None, then does not resample audio to different sample rate
        By default equals None

    _enforce_length_s: [int, float, None], optional
        Length (in seconds) to which trim or pad loaded waveform
        If None, then does not enforce
        By default equals None

    Methods
    -------
    __len__()
        Returns length of dataset

    __getitem__(idx)
        Attempts to load audio at paths[idx].
        If attempt is unsuccessful, attepmts to load another audio from given paths, until valid one is found.
        If no valid audios were found, returns Error.
        If any attempt was successful, returns dictionary {paths[valid_idx]: waveform},
        if labels were provided, returns ({paths[valid_idx]: waveform}, labels[valid_idx]).
        *valid_idx - index at which audio was successfully loaded, could be initial idx, if inital attempt was successful.

    """

    def __init__(
        self,
        paths,
        labels=None,
        only_waveform: bool = False,
        mono: bool = True,
        sample_rate: Union[int, None] = None,
        enforce_length_s: Union[int, float, None] = None,
    ):
        self._sources = paths
        self._labels = labels
        self._to_mono = mono
        self._enforce_length_s = enforce_length_s
        self._only_waveform = only_waveform
        self.sr = sample_rate

        # check if paths and labels have the same dimensions
        if self._labels is not None and len(self._sources) != len(self._labels):
            raise TypeError(
                f"Expected paths and labels to have save dimensions, but got len(sources):({len(self._sources)}) and len(labels):({len(self._labels)})"
            )

    def __len__(self):
        """Returns length of dataset

        Returns
        -------
        length : int
            Dataset length

        """

        length = len(self._sources)
        return length

    def __getitem__(self, idx):
        """
        Attempts to load audio at paths[idx].
        If attempt is unsuccessful, attepmts to load another audio from given paths, until valid one is found.
        If no valid audios were found, raises Error.
        If any attempt was successful, returns (paths[valid_idx], waveform).
        If labels were provided, returns ((paths[valid_idx], waveform), labels[valid_idx]).
        If (only_waveform==True), returns (waveform).
        If (only_waveform==True) and labels were provided, returns (waveform, labels[valid_idx]).
        *valid_idx - index at which audio was successfully loaded. It could be initial idx, if inital attempt was successful.

        Parameters
        ----------
        idx : int
            Index of the element to return

        Returns
        -------
        track_path: str
            Path to loaded track

        waveform: list
            Waveform loaded at track_path

        label: list, optional
            Label corresponding to loaded track, if labels were provided

        """

        # Try to load song at index idx, if unsuccessful try to load another random song
        try:
            waveform = load_audio(
                self._sources[idx],
                mono=self._to_mono,
                sample_rate=self.sr,
                enforce_length_s=self._enforce_length_s,
            )
        except Exception as e:
            print(f"Failed to load data at: {self._sources[idx]}")
            print(f"Reason: {e}")
            for i in randperm(self.__len__()).tolist():
                try:
                    waveform = self.load_audio(self._sources[i], self.sr)
                    idx = i  # replace idx with i (index at which audio was successfully loaded)
                    print(f"Loaded instead: {self._sources[i]}")
                    break
                except Exception as e:
                    print(f"Failed to load audio at: {self._sources[i]}")
                    print(f"Reason: {e}")

        if waveform is None:
            raise Exception("All paths are invalid")

        if self._only_waveform is True:
            if self._labels is None:
                return waveform
            else:
                return (waveform, self._labels[idx])
        else:
            if self._labels is None:
                return (self._sources[idx], waveform)
            else:
                return ((self._sources[idx], waveform), self._labels[idx])
