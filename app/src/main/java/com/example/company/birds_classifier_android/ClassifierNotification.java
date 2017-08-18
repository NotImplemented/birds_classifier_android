package com.example.company.birds_classifier_android;

/**
 * Created by Mikhail_Kaspiarovich on 8/14/2017.
 */

interface ClassifierNotification {

    void showNotification(String text);
    void displayRecognitionData(float[] pixels, int height, int width);
}