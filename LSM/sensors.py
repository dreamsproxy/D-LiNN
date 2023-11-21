import numpy as np
from sklearn.preprocessing import minmax_scale
import cv2

class Utilities:
    def __init__(self) -> None:
        pass
    def ConvertToWave(self, image):
        """this function converts images into k-space wave form"""
        ft = np.fft.fftshift(image)
        ft = np.fft.fft2(ft)
        ft = np.fft.ifftshift(ft)
class Vision:
    def __init__(self, n_channels: int = 1, resolution: int = 256, kernel_size: int = 3) -> None:
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.n_kernels = int(pow(resolution, 2) / pow(kernel_size, 2))
        #print(self.n_kernels)
        #raise
        self.resolution = (resolution, resolution)
        pass

    def Load(self, path):
        if self.n_channels == 1:
            grayscale_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            grayscale_image = cv2.resize(grayscale_image, (self.resolution))
            grayscale_image = np.float64(grayscale_image)
            return grayscale_image
        elif self.n_channels == 3:
            rgb_image = cv2.imread(path, cv2.IMREAD_COLOR)
            rgb_image = cv2.resize(rgb_image, self.resolution)
            return rgb_image

    def KernelHandler(self, img):
        k_size = self.kernel_size
        tiles = [img[x:x+k_size,y:y+k_size] for x in range(0,img.shape[0],k_size) for y in range(0,img.shape[1],k_size)]
        self.n_sensors = len(tiles)
        return tiles

    def ConvertToCurrent(self, image):
        image *= 255 / np.max(image)
        if self.n_channels == 1:
            image_as_current = minmax_scale(image, feature_range=(-75,-55))
        elif self.n_channels == 3:
            image_as_current = minmax_scale(image, feature_range=(-75,-55))
        return image_as_current
    
    def ToVector(self, image: np.ndarray):
        return image.flatten()


import pyaudio
import struct
class Audio:
    def __init__(self, n_bands) -> None:
        print("INIT AUDIO")
        self.n_bands = n_bands
        self.CHUNK = 2**10
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
        self.stream=p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK,
            input_device_index=1)

    def ExtractFrequencyBands(self):
        data = struct.unpack(
            str(self.CHUNK*self.CHANNELS) + 'h',
            self.stream.read(self.CHUNK))
        fft_data = np.fft.rfft(data)
        fft_data = np.abs(fft_data[:self.CHUNK]) * 2 / (256 * self.CHUNK)
        n = fft_data.size // self.n_bands
        bands = np.array([np.mean(fft_data[i:(i + n)]) for i in range(0, fft_data.size, n)])
        bands /= np.max(np.abs(bands),axis=0)
        bands_scaled = minmax_scale(bands, feature_range=(-75,-55))

        return bands_scaled

    def shutdown(self):
        self.stream.close()