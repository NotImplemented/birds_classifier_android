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
    int spectrogram_start = 0;
    int sample_size = 512;
    int time_shift  = sample_size / 2;
    int max_frame_rate = 44100;
    int sample_length = 5;
    int max_spectrogram_length = (max_frame_rate * sample_length - sample_size) / time_shift + 1;
    int spectrogram_classify_index = 0;

    FastFourierTransform fft = new FastFourierTransform(sample_size);

    double[] fft_buffer_x = new double[sample_size];
    double[] fft_buffer_y = new double[sample_size];
    double[] spectrogram = new double[sample_size];

    double[][] spectrogram_buffer = new double[max_spectrogram_length][];
    int spectrogram_buffer_index = 0;


    public SoundBuffer(int SAMPLE_RATE, int CHANNELS, int ENCODING_BITS) {

        length = SAMPLE_RATE;
        buffer = new short[length];
        index = 0;

        for(int i = 0; i < sample_size; ++i)
            fft_buffer_x[i] = fft_buffer_y[i] = 0;

        for(int i = 0; i < max_spectrogram_length; ++i)
            spectrogram_buffer[i] = new double[sample_size];
    }

    public void append(short[] data, int size) {

        if (index + size <= this.length) {

            System.arraycopy(data, 0, buffer, index, size);
            index += size;
        }
        else {

            int amount = this.length - index;
            System.arraycopy(data, 0, buffer, 0, amount);

            System.arraycopy(data, 0, buffer, 0, size - amount);
            index = size - amount;
        }

        int spectrogram_end = spectrogram_start + sample_size;
        if (spectrogram_end >= this.length)
            spectrogram_end -= this.length;

        while (spectrogram_end < index) {

            for (int i = 0, j = spectrogram_start; i < sample_size; ++i) {

                fft_buffer_y[i] = buffer[j++];

                if (j == this.length)
                    j = 0;
            }
            fft.fft(fft_buffer_x, fft_buffer_y);
            for (int i = 0; i < sample_size; ++i) {

                spectrogram[i] = Math.sqrt(1.0 + Math.sqrt(fft_buffer_x[i] * fft_buffer_x[i] + fft_buffer_y[i] * fft_buffer_y[i]));
            }

            spectrogram_start += time_shift;
            spectrogram_end = spectrogram_start + sample_size;
            if (spectrogram_end >= this.length)
                spectrogram_end -= this.length;

            appendSpectrogram();
        }
    }

    private void appendSpectrogram() {

        System.arraycopy(spectrogram, 0, spectrogram_buffer[spectrogram_buffer_index], 0, sample_size);
        ++spectrogram_buffer_index;
        if (spectrogram_buffer_index == max_spectrogram_length)
            spectrogram_buffer_index = 0;

        int spectrogram_shift = max_spectrogram_length / 2;

        if (spectrogram_buffer_index >= spectrogram_classify_index) {

            // prepare image and send to nn
            // [spectrogram_classify_index, spectrogram_classify_index + max_spectrogram_index]

            spectrogram_classify_index += spectrogram_shift;
            if (spectrogram_classify_index >= max_spectrogram_length)
                spectrogram_classify_index -= max_spectrogram_length;
        }
    }

}
