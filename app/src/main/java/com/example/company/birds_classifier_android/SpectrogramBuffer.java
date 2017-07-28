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
    double[][] spectrogram_buffer;
    long classified_index;

    double[] image = new double[SpectrogramLength * SampleSize];
    Classifier imageClassifier;

    public SpectrogramBuffer(Classifier imageClassifier) {

        index = 0;
        classified_index = 0;
        n = SpectrogramLength;
        spectrogram_buffer = new double[n][];

        this.imageClassifier = imageClassifier;
    }

    public void appendSpectrogram(double[] spectrogram) {

        System.arraycopy(spectrogram, 0, spectrogram_buffer[(int)(index % n)], 0, SampleSize);
        ++index;

        while (classified_index + SpectrogramLength <= index) {

            // Prepare image and send to neural network
            // [spectrogram_classify_index, spectrogram_classify_index + spectrogram_length]

            for(int i = 0; i < SampleSize; ++i)
                System.arraycopy(spectrogram_buffer[(int)((classified_index + i) % n)], 0, image, i*SampleSize, SampleSize);

            imageClassifier.recognizeImage(image);

            classified_index += SpectrogramShift;
        }
    }

}
