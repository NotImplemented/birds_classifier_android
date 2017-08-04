package com.example.company.birds_classifier_android;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Handler;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.ActionBar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.text.style.ImageSpan;
import android.util.Log;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import java.io.IOException;
import java.util.List;


public class MainActivity extends AppCompatActivity {

    private static final String LOG_TAG = "birds_classification";
    private static final int REQUEST_RECORD_AUDIO_PERMISSION = 200;
    private static final int AUDIO_ECHO_REQUEST = 0;
    private static final int SAMPLE_RATE = 16000;

    private boolean recordingActive = false;

    private final Handler handler = new Handler();
    private final String [] permissions = {Manifest.permission.RECORD_AUDIO};

    private TextView textView;
    private ToggleButton recordButton;
    private SoundBuffer soundBuffer;

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {

        if (AUDIO_ECHO_REQUEST != requestCode) {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
            return;
        }

        if (grantResults.length != 1 || grantResults[0] != PackageManager.PERMISSION_GRANTED) {

            textView.append("\nError: Permission for RECORD_AUDIO was denied.");
            return;
        }

        textView.append("\nRECORD_AUDIO permission granted.");
    }

    public void onToggleRecord(View view) {

        if (recordingActive)
            stopRecording();
        else
            startRecording();

        recordButton.setChecked(recordingActive);
    }

    private static final int INPUT_HEIGHT = SoundParameters.SampleSize;
    private static final int INPUT_WIDTH = SoundParameters.SpectrogramLength ;

    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "predictions";

    private static final String MODEL_FILE = "file:///android_asset/frozen_model.pb";
    private static final String LABEL_FILE = "file:///android_asset/graph_label_strings.txt";

    private Classifier classifier;

    public void appendText(final String text) {

        handler.post(new Runnable() { public void run() {

            textView.append(text);
        }
        });
    }

    private void startRecording() {

        textView.append("\n\nRecording has started.\n");
        recordingActive = true;

        textView.append("Checking RECORD_AUDIO permissions.\n");

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {

            textView.append("Requesting RECORD_AUDIO permissions.\n");

            ActivityCompat.requestPermissions(this, new String[] { Manifest.permission.RECORD_AUDIO }, AUDIO_ECHO_REQUEST);

            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {

                recordingActive = false;
                recordButton.setChecked(recordingActive);
                return;
            }
        }
        else {

            textView.append("Permissions RECORD_AUDIO were granted.\n");
        }

        if (classifier == null) {

            try {
                classifier = TensorFlowImageClassifier.create(this, getAssets(), MODEL_FILE, LABEL_FILE, INPUT_HEIGHT, INPUT_WIDTH, INPUT_NAME, OUTPUT_NAME);
            }
            catch (final Exception e) {

                handler.post(new Runnable() { public void run() {

                    textView.append(String.format("Cannot create tensorflow instance.\nError message: %s\n", e.getMessage()));

                    recordingActive = false;
                    recordButton.setChecked(recordingActive);
                    }
                });

                return;
            }
        }

        new Thread(new Runnable() {

            @Override
            public void run() {

                try {

                    android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);

                    final int sampleRate = 44100;
                    final int encoding = AudioFormat.ENCODING_PCM_16BIT;

                    handler.post(new Runnable() { public void run() { textView.append("Requesting minimal buffer size for rate = " + sampleRate + ", encoding = " + encoding + ".\n"); } });

                    final int bufferSize = AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, encoding);
                    handler.post(new Runnable() { public void run() { textView.append("Minimal buffer size = " + bufferSize + ".\n"); } });

                    int recordBufferSize = bufferSize < 0 ? 32768 : bufferSize;
                    AudioRecord record = new AudioRecord(MediaRecorder.AudioSource.MIC, sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, recordBufferSize);
                    handler.post(new Runnable() { public void run() { textView.append("AudioRecord was created.\n"); } });

                    short[] audioBuffer = new short[bufferSize];

                    if (record.getState() != AudioRecord.STATE_INITIALIZED) {

                        handler.post(new Runnable() { public void run() { textView.append("Cannot initialize audio record.\n"); } });
                    }

                    soundBuffer = new SoundBuffer(classifier, sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
                    record.startRecording();

                    handler.post(new Runnable() { public void run() { textView.append("Recording is active.\n"); } });

                    long frames = 0;
                    while (recordingActive) {

                        int read = record.read(audioBuffer, 0, audioBuffer.length);
                        soundBuffer.append(audioBuffer, read);
                        frames += read;
                    }

                    record.stop();
                    record.release();

                    final long totalFrames = frames;
                    handler.post(new Runnable() { public void run() { textView.append(String.format("Recording was stopped.\nFrames read: %d.\n", totalFrames)); } });
                }
                catch(final Exception e) {

                    handler.post(new Runnable() { public void run() {

                             textView.append(String.format("Recording was stopped unexpectedly.\nError message: %s.\n", e.getMessage()));

                             recordingActive = false;
                             recordButton.setChecked(recordingActive);

                        }
                    });
                }
            }
        }).start();
    }

    private void stopRecording() {

        textView.append("Stopping recording.\n");
        recordingActive = false;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);

        getSupportActionBar().setDisplayOptions(ActionBar.DISPLAY_SHOW_CUSTOM);
        getSupportActionBar().setCustomView(R.layout.action_bar);

        ActivityCompat.requestPermissions(this, permissions, REQUEST_RECORD_AUDIO_PERMISSION);

        textView = (TextView) findViewById(R.id.text_view);
        recordButton = (ToggleButton) findViewById(R.id.toggle_button);
    }

    @Override
    public void onStop() {

        super.onStop();
    }
}