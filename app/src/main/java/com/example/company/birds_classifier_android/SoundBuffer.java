package com.example.company.birds_classifier_android;

import android.media.AudioFormat;
import android.provider.Settings;

/**
 * Created by Mikhail_Kaspiarovich on 2/20/2017.
 */

public class SoundBuffer {

    short buffer[];
    int length;
    int index;

    private static final int WINDOW_SIZE = 512;
    private static final int TIME_SHIFT = (int)(WINDOW_SIZE * 0.25);

    private static final int FRAME_RATE = 22050;
    private static final int SAMPLE_LENGTH = 10; // Seconds.

    FastFourierTransform fft = new FastFourierTransform(WINDOW_SIZE);

    double[] fft_buffer_x = new double[WINDOW_SIZE];
    double[] fft_buffer_y = new double[WINDOW_SIZE];
    double[] spectrogram = new double[WINDOW_SIZE];

    int spectrogram_buffer_length = (FRAME_RATE * SAMPLE_LENGTH - WINDOW_SIZE) / TIME_SHIFT + 1;
    int spectrogram_shift = spectrogram_buffer_length / 4;
    double[][] spectrogram_buffer = new double[spectrogram_buffer_length][];
    int spectrogram_buffer_index = 0;

    int spectrogram_index = 0;
    int spectrogram_classify_index = 0;

    public SoundBuffer(int SAMPLE_RATE, int CHANNELS, int ENCODING_BITS) {

        length = SAMPLE_RATE;
        buffer = new short[length];
        index = 0;

        for(int i = 0; i < WINDOW_SIZE; ++i)
            fft_buffer_x[i] = fft_buffer_y[i] = 0;

        for(int i = 0; i < spectrogram_buffer_length; ++i)
            spectrogram_buffer[i] = new double[WINDOW_SIZE / 2];
    }

    public void append(short[] data, int size) {

        if (index % length + size <= length) {

            System.arraycopy(data, 0, buffer, index, size);
        }
        else {

            int amount = length - index % length;
            System.arraycopy(data, 0, buffer, index % length, amount);
            System.arraycopy(data, 0, buffer, 0, size - amount);
        }
        index += size;

        while (spectrogram_index + WINDOW_SIZE < index) {

            for (int i = 0, j = spectrogram_index % length; i < WINDOW_SIZE; ++i) {

                fft_buffer_y[i] = buffer[j++];

                if (j == this.length)
                    j = 0;
            }
            fft.fft(fft_buffer_x, fft_buffer_y);
            for (int i = 0; i < WINDOW_SIZE / 2; ++i) {

                double amplitude = Math.sqrt(fft_buffer_x[i] * fft_buffer_x[i] + fft_buffer_y[i] * fft_buffer_y[i]);
                spectrogram[i] = 20 * Math.log10(amplitude);
            }

            spectrogram_index += TIME_SHIFT;
            append(spectrogram);
        }
    }

    private void append(double[] spectrogram) {

        System.arraycopy(spectrogram, 0, spectrogram_buffer[spectrogram_buffer_index % spectrogram_buffer_length], 0, WINDOW_SIZE);
        ++spectrogram_buffer_index;

        while (spectrogram_classify_index + spectrogram_buffer_length < spectrogram_buffer_index) {

            // prepare image and send to nn
            // [spectrogram_classify_index, spectrogram_classify_index + max_spectrogram_index]

            spectrogram_classify_index += spectrogram_shift;
        }
    }

}
