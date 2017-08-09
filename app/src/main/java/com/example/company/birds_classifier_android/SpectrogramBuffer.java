package com.example.company.birds_classifier_android;

import android.provider.Settings;

import static com.example.company.birds_classifier_android.SoundParameters.SampleSize;
import static com.example.company.birds_classifier_android.SoundParameters.SpectrogramLength;
import static com.example.company.birds_classifier_android.SoundParameters.SpectrogramShift;

/**
 * Created by Mikhail_Kaspiarovich on 7/28/2017.
 */

public class SpectrogramBuffer {

    final int n;
    long index;
    float[][] spectrogram_buffer;
    long classified_index;

    float[] image = new float[SpectrogramLength * (SampleSize / 2)];
    Classifier imageClassifier;

    public SpectrogramBuffer(Classifier imageClassifier) {

        index = 0;
        classified_index = 0;
        n = SpectrogramLength;
        spectrogram_buffer = new float[n][];
        for(int i = 0; i < n; ++i)
            spectrogram_buffer[i] = new float[SampleSize / 2];

        this.imageClassifier = imageClassifier;
    }

    public void appendSpectrogram(float[] spectrogram) {

        System.arraycopy(spectrogram, 0, spectrogram_buffer[(int)(index % n)], 0, SampleSize / 2);
        ++index;

        while (classified_index + SpectrogramLength <= index) {

            // Prepare image and send to neural network
            // [spectrogram_classify_index, spectrogram_classify_index + spectrogram_length]

            for(int i = 0; i < SpectrogramLength; ++i)
                System.arraycopy(spectrogram_buffer[(int)((classified_index + i) % n)], 0, image, i * (SampleSize) / 2, SampleSize / 2);

            float mn = Float.MAX_VALUE, mx = Float.MIN_VALUE;
            for(int i = 0; i < SpectrogramLength * (SampleSize / 2); ++i)
            {
                mn = Math.min(mn, image[i]);
                mx = Math.max(mx, image[i]);
            }

            float ratio = mx - mn;
            for(int i = 0; i < SpectrogramLength * (SampleSize / 2); ++i)
            {
                image[i] = (float)Math.sqrt((double) (image[i] - mn) / ratio );
            }

            imageClassifier.recognizeImage(image);

            classified_index += SpectrogramShift;
        }
    }

}
