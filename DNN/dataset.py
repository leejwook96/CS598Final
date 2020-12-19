from __future__ import print_function, division
from scipy.io import wavfile
from scipy import signal
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class FSDD(Dataset):
    """ Free Spoken Digit Dataset """

    def __init__(self, audio_dir, transform=None):
        """
        Args:
            audio_dir: Path to the audio files
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.audio_dir = audio_dir
        self.audio_files = [f for f in os.listdir(
            audio_dir) if os.path.isfile(os.path.join(audio_dir, f))]
        self.transform = transform

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        f, t, spec = self._spec_analysis(
            os.path.join(self.audio_dir, self.audio_files[idx]))
        spec = Image.fromarray(spec.astype(np.uint8))

        if self.transform is not None:
            spec = self.transform(spec)

        return spec, int(self.audio_files[idx][0])

    def _spec_analysis(self, file_path):
        sample_rate, data = wavfile.read(file_path)
        f, t, spec = signal.spectrogram(
            data, fs=8000, nperseg=50, window='hamming')
        return f, t, np.log(abs(spec))


if __name__ == '__main__':
    fsdd = FSDD('./free-spoken-digit-dataset/recordings',
                transform=transforms.Compose([
                    transforms.Resize((500, 10)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
    for i in range(5):
        print(fsdd[i][0])
