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
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;


public class TensorFlowImageClassifier implements Classifier {

    private static final String TAG = "TensorFlowImageClassifier";
    private static final double THRESHOLD = 0.05;

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
    private ArrayList<Recognition> recognitions;

    private static MainActivity activity;

    private TensorFlowInferenceInterface inferenceInterface;

    private TensorFlowImageClassifier() {
    }

    public static ArrayList<Recognition> CreateRecognitions() {

        ArrayList<Recognition> recognitions = new ArrayList<>();

        int id = 0;
        recognitions.add(new Recognition(id++, "Brown Creeper", 0.0));
        recognitions.add(new Recognition(id++, "Pacific Wren", 0.2849));
        recognitions.add(new Recognition(id++, "Pacific-slope Flycatcher", 0.0667f));
        recognitions.add(new Recognition(id++, "Red-breasted Nuthatch", 0.0793f));
        recognitions.add(new Recognition(id++, "Dark-eyed Junco", 0.0));

        recognitions.add(new Recognition(id++, "Olive-sided Flycatcher", 0.0));
        recognitions.add(new Recognition(id++, "Hermit Thrush", 0.1092));
        recognitions.add(new Recognition(id++, "Chestnut-backed Chickadee", 0.0553));
        recognitions.add(new Recognition(id++, "Varied Thrush", 0.0));
        recognitions.add(new Recognition(id++, "Hermit Warbler", 0.0));

        recognitions.add(new Recognition(id++, "Swainsons's Thrush", 0.0107));
        recognitions.add(new Recognition(id++, "Hammond's Flycatcher", 0.2906));
        recognitions.add(new Recognition(id++, "Western Tanager", 0.1456));
        recognitions.add(new Recognition(id++, "Black-headed Grosbeak", 0.0091));
        recognitions.add(new Recognition(id++, "Golden Crowned Kinglet", 0.0));

        recognitions.add(new Recognition(id++, "Warbling Vireo", 0.0499));
        recognitions.add(new Recognition(id++, "MacGillivray's Warbler", 0.2488));
        recognitions.add(new Recognition(id++, "Stellar's Jay", 0.0640));
        recognitions.add(new Recognition(id++, "Common Nighthawk", 0.0));

        return recognitions;
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
        TensorFlowImageClassifier classifier = new TensorFlowImageClassifier();
        classifier.inputName = inputName;
        classifier.outputName = outputName;

        // Read the label names into memory.
        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        log("Reading labels from: " + actualFilename + ".\n");
        BufferedReader br = null;
        br = new BufferedReader(new InputStreamReader(assetManager.open(actualFilename)));
        String line;
        while ((line = br.readLine()) != null) {
            classifier.labels.add(line);
        }
        br.close();

        classifier.inferenceInterface = new TensorFlowInferenceInterface();
        if (classifier.inferenceInterface.initializeTensorFlow(assetManager, modelFilename) != 0) {
            throw new RuntimeException("TensorFlow initialization failed");
        }
        log("Read " + classifier.labels.size() + " labels.\n");

        // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
        String output_shape = classifier.inferenceInterface.graph().operation(outputName).output(0).shape().toString();
        classifier.outputSize = (int)classifier.inferenceInterface.graph().operation(outputName).output(0).shape().size(3);
        log("Output layer: " + output_shape + ".\n");

        //String input = c.inferenceInterface.graph().operation(inputName).type();
        //log("Input layer is " + input + ".\n");

        log("Model file: " + modelFilename + ".\n");

        classifier.recognitions = CreateRecognitions();

        // Ideally, inputSize could have been retrieved from the shape of the input operation.  Alas,
        // the placeholder node for input in the graphdef typically used does not specify a shape, so it
        // must be passed in as a parameter.
        classifier.inputWidth = inputWidth;
        classifier.inputHeight = inputHeight;

        // Pre-allocate buffers.
        classifier.outputNames = new String[]{outputName};
        classifier.outputs = new float[classifier.outputSize];

        return classifier;
    }

    @Override
    public List<Recognition> recognizeImage(final float[] pixels) {

        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        // Copy the input data into TensorFlow.
        Trace.beginSection("fillNodeFloat");
        inferenceInterface.fillNodeFloat(inputName, new int[]{inputHeight * inputWidth}, pixels);
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
        ArrayList<Recognition> list = new ArrayList<>();

        for (int i = 0; i < outputs.length; ++i) {

            list.add(new Recognition("" + i, labels.size() > i ? labels.get(i) : "Unknown", outputs[i], null));
        }

        Trace.endSection(); // "recognizeImage"


        for (int i = 0; i < list.size(); ++i) {

            float d = list.get(i).getConfidence() - recognitions.get(i).getConfidence();
            list.get(i).setDifference(d);
        }

        Collections.sort(list, new Comparator<Recognition>() {

            @Override
            public int compare(Recognition o1, Recognition o2) {

                return (int) Math.signum(o2.getDifference() - o1.getDifference());
            }
        });

        for (int i = 0; i < list.size(); ++i) {

            log(String.format("Bird \"%s\": diff = %.4f\n", list.get(i).getTitle(), list.get(i).getDifference()));

            if (list.get(i).getDifference() >= THRESHOLD) {

                activity.showNotification(String.format("%s: %.4f\n", list.get(i).getTitle(), list.get(i).getDifference()));
            }
        }

        activity.displayRecognitionData(pixels, inputHeight, inputWidth);

        return recognitions;
    }

    @Override
    public void enableStatisticsLogging(boolean debug) {

        inferenceInterface.enableStatLogging(debug);
    }

    @Override
    public String getStatisticsString() {

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