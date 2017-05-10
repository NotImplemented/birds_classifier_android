package com.example.company.birds_classifier_android;

import android.media.AudioFormat;
import android.provider.Settings;

/**
 * Created by Mikhail_Kaspiarovich on 2/20/2017.
 */

public class SoundBuffer {

    /* Raw sound data buffer. */
    short buffer[];
    int length;
    int index;
    int spectrogram_index = 0;

    public static final int WINDOW_SIZE = 512;
    public static final int TIME_SHIFT = (int)(WINDOW_SIZE * 0.25);

    public static final int FRAME_RATE = 22050;
    public static final int SAMPLE_LENGTH = 10; // Seconds.

    private FastFourierTransform fft = new FastFourierTransform(WINDOW_SIZE);
    private Classifier classifier;

    double[] hann = new double[WINDOW_SIZE];
    double[] fft_buffer_x = new double[WINDOW_SIZE];
    double[] fft_buffer_y = new double[WINDOW_SIZE];
    double[] spectrogram = new double[WINDOW_SIZE];

    public static final int SPECTROGRAM_LENGTH = (FRAME_RATE * SAMPLE_LENGTH - WINDOW_SIZE) / TIME_SHIFT + 1;
    public static final int SPECTROGRAM_SHIFT = SPECTROGRAM_LENGTH / 4;

    /* Spectrogram buffer. */
    double[][] spectrogram_buffer = new double[SPECTROGRAM_LENGTH][];
    int spectrogram_buffer_index = 0;
    int spectrogram_classify_index = 0;

    double[] image = null;

    public SoundBuffer(Classifier imageClassifier, int SAMPLE_RATE, int CHANNELS, int ENCODING_BITS) {

        classifier = imageClassifier;

        length = SAMPLE_RATE;
        buffer = new short[length];
        index = 0;

        for(int i = 0; i < WINDOW_SIZE; ++i)
            fft_buffer_x[i] = fft_buffer_y[i] = 0;

        for(int i = 0; i < SPECTROGRAM_LENGTH; ++i)
            spectrogram_buffer[i] = new double[WINDOW_SIZE / 2];

        for(int i = 0; i < WINDOW_SIZE; ++i)
            hann[i] =  0.5 * (1.0 - Math.cos(2 * Math.PI * i / (WINDOW_SIZE - 1)) );
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

                fft_buffer_y[i] = buffer[j++] * hann[i];

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

    private double[] normalize(double[] data) {

        double mn = Double.MAX_VALUE;
        double mx = Double.MIN_VALUE;
        for(int i = 0; i < data.length; ++i) {

            mn = Math.min(data[i], mn);
            mx = Math.max(data[i], mx);
        }

        if (mn != mx) {

            for (int i = 0; i < data.length; ++i) {
                data[i] = (data[i] - mn) / (mx - mn);
            }
        }

        return data;
    }

    private void append(double[] spectrogram) {

        System.arraycopy(spectrogram, 0, spectrogram_buffer[spectrogram_buffer_index % SPECTROGRAM_LENGTH], 0, WINDOW_SIZE);
        ++spectrogram_buffer_index;

        while (spectrogram_classify_index + SPECTROGRAM_LENGTH < spectrogram_buffer_index) {

            // prepare image and send to nn
            // [spectrogram_classify_index, spectrogram_classify_index + max_spectrogram_index]

            if (image == null) {

                image = new double[SPECTROGRAM_LENGTH * WINDOW_SIZE / 2];
            }

            int image_index = 0;
            for (int i = 0, j = spectrogram_classify_index % length; i < SPECTROGRAM_LENGTH; ++i) {

                System.arraycopy(spectrogram_buffer[j], 0, image, image_index, WINDOW_SIZE / 2);

                ++j;
                image_index += WINDOW_SIZE / 2;

                if (j == SPECTROGRAM_LENGTH)
                    j = 0;
            }

            image = normalize(image);
            classifier.recognizeImage(image);

            spectrogram_classify_index += SPECTROGRAM_SHIFT;
        }
    }

}