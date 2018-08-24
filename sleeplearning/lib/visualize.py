from typing import List
from sleeplearning.lib.loaders.baseloader import BaseLoader
import matplotlib.pyplot as plt
import numpy as np
import itertools


def compute_transition_matrix_(data):
    M = np.zeros((7, 7))
    for sl in data:
        for i, j in zip(sl.hypnogram, sl.hypnogram[1:]):
            M[i, j] += 1
    return M


class Visualize(object):
    "Class to visualize psg data of a list of sleeplearning objects"

    def __init__(self, sleeplearnings: List[BaseLoader]):
        self.data = sleeplearnings

    def class_distribution(self):
        sleep_stage_dist = []
        subject_labels = []
        total_labels = 0
        for sl in self.data:
            sleep_stage_dist.append(sl.hypnogram)
            total_labels += len(sl.hypnogram)
            subject_labels.append(sl.label)
        plt.figure(figsize=(10, 5))

        plt.hist(sleep_stage_dist, label=subject_labels,
                 bins=np.arange(8) - 0.5)
        plt.legend()
        plt.ylabel('count')
        plt.title(
            "Sleep Phases Distribution ({0} labels)".format(str(total_labels)))
        _ = plt.xticks(np.arange(7),
                       list(BaseLoader.sleep_stages_labels.values()))

    def transition_distribution(self):
        num_sleep_phases = len(BaseLoader.sleep_stages_labels.keys())
        M = compute_transition_matrix_(self.data).astype(int)
        cmap = plt.cm.Blues
        M_norm = M / (0.0001 + np.sum(M, axis=1)[:, np.newaxis])
        plt.figure(figsize=(10,10))
        plt.imshow(M_norm, interpolation='nearest', cmap=cmap)
        tick_marks = np.arange(num_sleep_phases)
        plt.yticks(tick_marks, BaseLoader.sleep_stages_labels.values())
        plt.xticks(tick_marks, BaseLoader.sleep_stages_labels.values())
        plt.ylabel('FROM')
        plt.xlabel('TO')
        plt.title("Transition probabilities")
        for i, j in itertools.product(range(M.shape[0]), range(M.shape[1])):
            plt.text(j, i, '{}\n({:.2f})'.format(M[i, j], M_norm[i, j]),
                    horizontalalignment="center", fontsize=12,
                    verticalalignment='center', color="black")

    def psd(self, channel: str, sleep_phases: List[int]):
        plt.figure(figsize=(10, 5))
        ax = plt.subplot('111')
        for i in sleep_phases:
            y = np.array([])
            for sl in self.data:
                window = sl.sampling_rate_ * 2
                stride = sl.sampling_rate_
                f, periodograms = sl.get_psds(channel, window, stride)
                ind = np.where(sl.hypnogram == i)[0]
                y_add = periodograms[ind]
                # remove none types (epochs with only artefacts)
                y_add = y_add[np.array([y.size > 0 for y in y_add])]
                y = np.vstack([y, y_add]) if y.size else y_add
            y_mean = np.mean(y, axis=0)
            error = np.std(y, axis=0)

            ax.plot(f, y_mean, label=BaseLoader.sleep_stages_labels[i])
            # ax.fill_between(f, y_mean - error/2, y_mean + error/2,
            #                 alpha=0.3)

        ax.set_xlim(xmin=0.5, xmax=100)
        ax.set_ylim(ymin=0.0099)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD [$\mu$V**2/Hz]")
        ax.set_title("Mean Power Spectral Density ")
        ax.set_yscale("log")
        ax.legend()

    def epoch(self, epoch_index: int, channel: str):
        for sl in self.data:
            if sl.hypnogram is not None:
                sleep_phase = sl.hypnogram[epoch_index]
                sleep_phase = BaseLoader.sleep_stages_labels[sleep_phase]
            else:
                sleep_phase = "UNK"
            f, axarr = plt.subplots(1, 3, sharex=False, figsize=(20, 5))
            f.suptitle('[' + sl.label + '] - epoch ' + str(
                epoch_index) + ' - sleep phase: [' + sleep_phase + ']',
                       fontsize=16)

            samples_per_epoch = sl.sampling_rate_ * sl.epoch_length
            t = sl.epoch_length * np.arange(samples_per_epoch, dtype=float) / samples_per_epoch
            # t+=offset/30
            psg = sl.psgs[channel][epoch_index * samples_per_epoch:
                                    (epoch_index + 1) * samples_per_epoch]
            axarr[0].plot(t, psg)
            axarr[0].set_xlabel("time [s]")
            axarr[0].set_ylabel("mV")

            window = sl.sampling_rate_ * 2
            stride = sl.sampling_rate_
            f, t, Sxx_list = sl.get_spectrograms(channel, window, stride)
            Sxx = Sxx_list[epoch_index]
            Sxx = Sxx ** 2  # psd from magnitude spectrum
            db = 10 * np.log10(Sxx)  # convert to dB
            axarr[1].pcolormesh(t, f, db, cmap='jet', vmin=-15)
            axarr[1].invert_yaxis()
            axarr[1].set_xlabel("time [s]")
            axarr[1].set_ylabel('frequency [Hz]')
            axarr[1].set_ylim(ymin=30)
            window = sl.sampling_rate_*2
            stride = sl.sampling_rate_
            f, Pxx_list = sl.get_psds(channel, window, stride)
            pxx = Pxx_list[epoch_index]
            axarr[2].plot(f, pxx)
            axarr[2].set_ylabel("$\mu$V**2/Hz")
            axarr[2].set_xlabel("frequency (Hz)")
            axarr[2].set_xlim(xmax=50)
            axarr[2].set_yscale('log')

    def qualitative_analysis(self, channel, from_epoch=0, to_epoch=None):
        for sl in self.data:
            if to_epoch is None:
                # set to last epoch of sample if no value given
                to_epoch = int(len(
                    sl.psgs[channel]) // sl.epoch_length // sl.sampling_rate_)

            fig, axarr = plt.subplots(3, sharex=True, sharey=False,
                                      figsize=(20, 20))
            fig.suptitle(
                '[' + sl.label + '] Qualitative Analysis (epoch length ' + str(
                    sl.epoch_length) + 's)', fontsize=16)
            samples_per_epoch = sl.sampling_rate_ * sl.epoch_length
            axarr[0].set_xlim(xmin=from_epoch, xmax=to_epoch)
            axarr[0].plot(np.linspace(from_epoch, to_epoch, (
                        to_epoch - from_epoch) * samples_per_epoch),
                          sl.psgs[channel][from_epoch * samples_per_epoch:
                                            to_epoch * samples_per_epoch])
            axarr[0].set_ylim(ymin=-300, ymax=300)
            axarr[0].set_ylabel('mV')
            window = sl.sampling_rate_ * 2
            stride = sl.sampling_rate_
            f, t, Sxx_list = sl.get_spectrograms(channel, window, stride)
            Sxx = Sxx_list[from_epoch:to_epoch]
            # compute psd from magnitude
            Sxx = Sxx ** 2
            dBS = 10 * np.log10(Sxx)  # convert to dB
            dBS = dBS.swapaxes(0, 1)
            dBS = dBS.reshape((dBS.shape[0], -1))
            pc = axarr[1].pcolormesh(np.linspace(from_epoch, to_epoch,
                len(t) * (to_epoch - from_epoch)),f, dBS, cmap='jet', vmin=-15)
            axarr[1].set_ylim(ymax=40)
            axarr[1].invert_yaxis()
            axarr[1].set_ylabel("spectogram [Hz]")
            h = range(from_epoch, to_epoch)
            axarr[2].plot(h, sl.hypnogram[from_epoch:to_epoch])
            axarr[2].set_yticks(
                range(0, len(BaseLoader.sleep_stages_labels.keys())))
            axarr[2].set_yticklabels(BaseLoader.sleep_stages_labels.values())
            axarr[2].set_xlim(xmin=from_epoch, xmax=to_epoch)
            axarr[2].set_xlabel("epoch")
            position = fig.add_axes([0.74, 0.633, 0.15, 0.01])
            cbar = fig.colorbar(pc, cax=position, orientation='horizontal')
            cbar.ax.set_title('Power (dB)')
            # cbar.set_ticks([zmin, (zmax+zmin)/2, zmax])
            # cbar.set_ticklabels([zmin, (zmax+zmin)/2, zmax])
