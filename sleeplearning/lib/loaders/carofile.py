import numpy as np

import scipy.io
from scipy import signal
from typing import Tuple

from sleeplearning.lib.loaders.baseloader import BaseLoader


class Carofile(BaseLoader):
    def __init__(self, path: str, epoch_length: int = 20, verbose: bool = False):
        super().__init__(path, epoch_length)
        psg_dict = {'EEG_raw': 'FpzA2', 'EEG': 'FpzA2_filt', 'EOGR': 'EOGR_filt',
                    'EOGL': 'EOGL_filt', 'EMG': 'EMG_filt'}
        self.label = self.path.split('/')[-1][5:-12]
        self.psgs = {}
        mat = scipy.io.loadmat(self.path)
        self.sampling_rate_ = int(mat['sampling_rate'][0][0])

        for expert in ["E1","E2","E3","E4","E5","CL"]:
            # TODO use labelling more intelligently
            if f'sleepStage_score_{expert}' in mat:
                self.artefact_data = {'artefacts': mat[f'artfact_per4s_{expert}'][0],
                                      'epoch_size': 4}

                epoch_scoring_length = int(mat['epoch_size_scoring_sec'][0][0])
                if epoch_scoring_length % self.epoch_length != 0:
                    raise ValueError(
                        "epoch length ({0}s) must divide scoring length ({1}s)".format(
                            str(self.epoch_length), str(epoch_scoring_length)))
                self.hypnogram = mat[f'sleepStage_score_{expert}'][0]
                break

        if self.hypnogram is None:
            print(mat.keys())
        if 4 in self.hypnogram:
            print(path)
            raise Exception("Invalid value in sleep labels")

        else:
            epoch_scoring_length = self.epoch_length

        for k, v in psg_dict.items():
            num_samples = mat[v].shape[1]
            samples_wo_label = num_samples % (
                        epoch_scoring_length * self.sampling_rate_)
            if verbose: print(k + ": cutting ",
                              samples_wo_label / self.sampling_rate_,
                              "seconds at the end")
            psg_cut = mat[v][0][:-samples_wo_label]
            self.psgs[k] = psg_cut

    def get_psds(self, channel: str, window: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the power spectral densities for a specific channel and for
        every epoch excluding artefacts.
        :param channel: channel key as stored in self.psgs
        :return: frequencies [fs/2+1], psds [numEpochs, fs/2+1]
        """
        pxxs = []
        artefacts = np.repeat(self.artefact_data['artefacts'],
                              self.artefact_data[
                                  'epoch_size'] * self.sampling_rate_)
        # reshape to [num epochs, samples per epoch]
        psgs = self.psgs[channel].reshape(
            (-1, self.sampling_rate_ * self.epoch_length))
        artefacts = artefacts.reshape(
            (-1, self.sampling_rate_ * self.epoch_length))

        padding = window // 2 - stride // 2
        psgs = np.pad(psgs, pad_width=((0, 0), (padding, padding)), mode='edge')
        artefacts = np.pad(artefacts, pad_width=((0, 0), (padding, padding)),
                           mode='edge')
        f = 0
        for psg, artefact in zip(psgs, artefacts):
            psg_clean = psg[artefact == 0]
            f, pxx = signal.welch(psg_clean, fs=self.sampling_rate_,
                                  nperseg=window,
                                  noverlap=window - stride,
                                  scaling='density')
            pxxs.append(pxx)
        return f, np.array(pxxs)