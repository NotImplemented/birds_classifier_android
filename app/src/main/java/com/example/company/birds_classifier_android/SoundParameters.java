package com.example.company.birds_classifier_android;

/**
 * Created by Mikhail_Kaspiarovich on 7/28/2017.
 */

public class SoundParameters {

    // Amount of sound values to calculate fourier transform
    public final static int SampleSize = 512;

    // Time shift between consecutive frames
    public final static int TimeShift = SampleSize / 4;

    // Sound frame rate
    public final static int MaxFrameRate = 16000;

    // Sample length in seconds to put to neural network
    public final static int SampleLength = 10;

    // Width of the spectrograms buffer
    public final static int SpectrogramLength = (MaxFrameRate * SampleLength - SampleSize) / TimeShift + 1;

    // Amount of spectrograms to shift (specifies period of running inception)
    public final static int SpectrogramShift = SpectrogramLength / 2;
}
