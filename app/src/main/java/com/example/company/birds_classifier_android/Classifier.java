package com.example.company.birds_classifier_android;

import android.graphics.RectF;

import java.util.List;

/**
 * Generic interface for interacting with different recognition engines.
 */
public interface Classifier {

    /**
     * An immutable result returned by a Classifier describing what was recognized.
     */
    public class Recognition {
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        private final String id;

        /**
         * Display name for the recognition.
         */
        private final String title;

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        private final Float confidence;

        /**
         * Optional location within the source image for the location of the recognized object.
         */
        private Object location;

        /**
         * Difference with basic level.
         */
        private float difference;

        public Recognition(final int id, final String title, final double confidence)
        {
            this(Integer.toString(id), title, confidence, null);
        }

        public Recognition(final String id, final String title, final double confidence)
        {
            this(id, title, confidence, null);
        }

        public Recognition(final String id, final String title, final double confidence, final Object location) {

            this.id = id;
            this.title = title;
            this.confidence = (float)confidence;
            this.location = location;
        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }

        public Float getDifference() {

            return difference;
        }

        public void setDifference(float d) {

            difference = d;
        }

        public Object getLocation() {
            return location;
        }

        public void setLocation(Object location) {
            this.location = location;
        }

        @Override
        public String toString() {
            String resultString = "";
            if (id != null) {
                resultString += "[" + id + "] ";
            }

            if (title != null) {
                resultString += title + " ";
            }

            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f);
            }

            if (location != null) {
                resultString += location + " ";
            }

            return resultString.trim();
        }
    }

    List<Recognition> recognizeImage(float[] pixels);

    String getStatisticsString();

    void enableStatisticsLogging(final boolean debug);

    void close();


}

