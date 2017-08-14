package com.example.company.birds_classifier_android;

import android.content.res.AssetManager;
import android.os.Trace;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;


public class TensorFlowImageClassifier implements Classifier {

    private static final String TAG = "TensorFlowImageClassifier";

    // Config values.
    private String inputName;
    private String outputName;
    private int inputWidth;
    private int inputHeight;
    private int outputSize;

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private float[] outputs;
    private String[] outputNames;
    private static MainActivity activity;

    private TensorFlowInferenceInterface inferenceInterface;

    private TensorFlowImageClassifier() {
    }

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager  The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     * @param inputHeight   The input size. A square image of inputHeight x inputWidth is assumed.
     * @param inputWidth    The input size. A square image of inputHeight x inputWidth is assumed.
     * @param inputName     The label of the image input node.
     * @param outputName    The label of the output node.
     * @throws IOException
     */
    public static Classifier create(MainActivity mainActivity, AssetManager assetManager, String modelFilename, String labelFilename,
                                    int inputHeight, int inputWidth, String inputName, String outputName)
            throws IOException {

        activity = mainActivity;
        TensorFlowImageClassifier c = new TensorFlowImageClassifier();
        c.inputName = inputName;
        c.outputName = outputName;

        // Read the label names into memory.
        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        log("Reading labels from: " + actualFilename + ".\n");
        BufferedReader br = null;
        br = new BufferedReader(new InputStreamReader(assetManager.open(actualFilename)));
        String line;
        while ((line = br.readLine()) != null) {
            c.labels.add(line);
        }
        br.close();

        c.inferenceInterface = new TensorFlowInferenceInterface();
        if (c.inferenceInterface.initializeTensorFlow(assetManager, modelFilename) != 0) {
            throw new RuntimeException("TensorFlow initialization failed");
        }
        log("Read " + c.labels.size() + " labels.\n");

        // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
        String output_shape = c.inferenceInterface.graph().operation(outputName).output(0).shape().toString();
        c.outputSize = (int)c.inferenceInterface.graph().operation(outputName).output(0).shape().size(3);
        log("Output layer size is " + output_shape + ".\n");

        //String input = c.inferenceInterface.graph().operation(inputName).type();
        //log("Input layer is " + input + ".\n");

        log("Model file name: " + modelFilename + ".\n");


        // Ideally, inputSize could have been retrieved from the shape of the input operation.  Alas,
        // the placeholder node for input in the graphdef typically used does not specify a shape, so it
        // must be passed in as a parameter.
        c.inputWidth = inputWidth;
        c.inputHeight = inputHeight;

        // Pre-allocate buffers.
        c.outputNames = new String[]{outputName};
        c.outputs = new float[c.outputSize];

        return c;
    }

    @Override
    public List<Recognition> recognizeImage(final float[] pixels) {

        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        // Copy the input data into TensorFlow.
        Trace.beginSection("fillNodeFloat");
        inferenceInterface.fillNodeFloat(inputName, new int[] {inputHeight * inputWidth}, pixels);
        Trace.endSection();

        // Display timestamp.
        SimpleDateFormat sdf = new SimpleDateFormat("HH:mm:ss.SSS");
        String time = sdf.format(new Date());
        log(String.format("[%s] Running inference.\n", time));

        // Run the inference call.
        Trace.beginSection("runInference");
        inferenceInterface.runInference(outputNames);
        Trace.endSection();

        // Copy the output Tensor back into the output array.
        Trace.beginSection("readNodeFloat");
        inferenceInterface.readNodeFloat(outputName, outputs);
        Trace.endSection();

        // Find the best classifications.
        PriorityQueue<Recognition> pq =
                new PriorityQueue<Recognition>(
                        outputSize,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (int i = 0; i < outputs.length; ++i) {
                pq.add(new Recognition("" + i, labels.size() > i ? labels.get(i) : "Unknown", outputs[i], null));
        }

        final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
        int recognitionsSize = Math.min(pq.size(), outputSize);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        Trace.endSection(); // "recognizeImage"

        for(Recognition recognition : recognitions) {

            log(String.format("Bird \"%s\": %.4f.\n", recognition.getTitle(), recognition.getConfidence()));
        }

        double mean = 0;
        int n = 0;
        for(Recognition recognition : recognitions) {

            mean += recognition.getConfidence();
            n += 1;
        }

        mean /= n;

        double stddev = 0;
        for(Recognition recognition : recognitions) {

            double d = (mean - recognition.getConfidence());

            stddev += d * d;
        }

        stddev = Math.sqrt(stddev / n);

        log(String.format("Mean = %.4f Deviation = %.4f\n", mean, stddev));

        for (Recognition recognition : recognitions) {

            double low = mean - 2 * stddev;
            double high = mean + 2 * stddev;

            if (recognition.getConfidence() > high)
            {
                log(String.format("Bird \"%s\" is detected: %.4f.\n", recognition.getTitle(), recognition.getConfidence()));
                activity.showNotification(String.format("%s: %.4f.\n", recognition.getTitle(), recognition.getConfidence()));
            }
        }

        return recognitions;
    }

    @Override
    public void enableStatLogging(boolean debug) {
        inferenceInterface.enableStatLogging(debug);
    }

    @Override
    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }

    private static void log(String message) {

        Log.i(TAG, message);
        activity.appendText(message);
    }
}