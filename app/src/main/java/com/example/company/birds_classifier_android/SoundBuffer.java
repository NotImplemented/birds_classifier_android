package com.example.company.birds_classifier_android;

import android.media.AudioFormat;
import android.provider.Settings;

import static com.example.company.birds_classifier_android.SoundParameters.SampleSize;
import static com.example.company.birds_classifier_android.SoundParameters.SpectrogramLength;
import static com.example.company.birds_classifier_android.SoundParameters.TimeShift;

/**
 * Created by Mikhail_Kaspiarovich on 2/20/2017.
 */


public class SoundBuffer {

    final int n;

    short buffer[];
    long index;
    long spectrogram_index;

    FastFourierTransform fft = new FastFourierTransform(SampleSize);
    SpectrogramBuffer spectrogramBuffer;

    final double[] fourier_buffer_x = new double[SampleSize];
    final double[] fourier_buffer_y = new double[SampleSize];
    final double[] spectrogram_buffer =  new double[SampleSize];

    public SoundBuffer(Classifier imageClassifier, int SAMPLE_RATE, int CHANNELS, int ENCODING_BITS) {

        n = SAMPLE_RATE;
        buffer = new short[n];
        index = 0;
        spectrogram_index = 0;

        for(int i = 0; i < SampleSize; ++i)
            fourier_buffer_x[i] = fourier_buffer_y[i] = 0;

        spectrogramBuffer = new SpectrogramBuffer(imageClassifier);
    }

    public void append(short[] data, int size) {

        if (index % n + size <= n) {

            System.arraycopy(data, 0, buffer, (int)(index % n), size);
        }
        else {

            int amount = n - (int)(index % n);
            System.arraycopy(data, 0, buffer, (int)(index % n), amount);
            System.arraycopy(data, 0, buffer, 0, size - amount);
        }
        index += size;

        while (spectrogram_index + SampleSize < index) {

            for (int i = 0, j = (int)(spectrogram_index % n); i < SampleSize; ++i) {

                fourier_buffer_y[i] = buffer[j++];
                fourier_buffer_x[i] = 0;

                if (j == n)
                    j = 0;
            }
            fft.fft(fourier_buffer_x, fourier_buffer_y);

            for (int i = 0; i < SampleSize; ++i) {

                double x = fourier_buffer_x[i];
                double y = fourier_buffer_y[i];

                spectrogram_buffer[i] = Math.sqrt(x * x + y * y);
            }

            spectrogram_index += TimeShift;

            spectrogramBuffer.appendSpectrogram(spectrogram_buffer);
        }
    }
}

