package com.example.company.birds_classifier_android;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.AudioTimestamp;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.ActionBar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.text.method.ScrollingMovementMethod;
import android.text.style.ImageSpan;
import android.util.Log;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;


public class MainActivity extends AppCompatActivity
        implements ClassifierNotification {

    private static final String LOG_TAG = "birds_classification";
    private static final int REQUEST_RECORD_AUDIO_PERMISSION = 200;
    private static final int AUDIO_ECHO_REQUEST = 0;

    private boolean recordingActive = false;

    private final Handler handler = new Handler();
    private final String [] permissions = {Manifest.permission.RECORD_AUDIO};

    private TextView textView;
    private ToggleButton recordButton;
    private SoundBuffer soundBuffer;
    private ImageView imageView;

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

    private static final int INPUT_HEIGHT = SoundParameters.SampleSize / 2;
    private static final int INPUT_WIDTH = SoundParameters.SpectrogramLength;

    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "predictions";

    private static final String MODEL_FILE = "file:///android_asset/optimized_mlsp_birds.pb";
    private static final String FROZEN_MODEL_FILE = "file:///android_asset/frozen_mlsp_birds.pb";
    private static final String LABEL_FILE = "file:///android_asset/graph_label_strings.txt";

    private Classifier classifier;

    public void appendText(final String text) {

        handler.post(new Runnable() { public void run() {

            textView.append(text);
        }
        });
    }

    public String getTime() {

        SimpleDateFormat sdf = new SimpleDateFormat("HH:mm:ss.SSS");
        final String time = sdf.format(new Date());

        return time;
    }

    public void showNotification(String text) {

        showToast(text);
    }

    public void displayRecognitionData(float[] pixels, int height, int width) {

        float mn = Float.MAX_VALUE;
        float mx = Float.MIN_VALUE;

        for(int i = 0; i < pixels.length; ++i) {

            mn = Math.min(mn, pixels[i]);
            mx = Math.max(mx, pixels[i]);
        }


        final int[] imageArray = new int[width * height];

        for (int i = 0; i < height; i++) {

            for (int j = 0; j < width; j++) {

                int a = 0xFF000000;
                int p = (int) (256 * (pixels[i * width + j] - mn) / (mx - mn));

                imageArray[i * width + j] = a | p | (p << 8) | (p << 16);

            }

        }

        final Bitmap bitmap = Bitmap.createBitmap(imageArray, width, height, Bitmap.Config.ARGB_8888);

        handler.post(new Runnable() { public void run() {

            imageView.setImageBitmap(bitmap);
            imageView.refreshDrawableState();
        }});
    }

    private void showToast(final String text) {

        handler.post(new Runnable() { public void run() {

            LayoutInflater inflater = getLayoutInflater();
            View layout = inflater.inflate(R.layout.custom_toast, (ViewGroup) findViewById(R.id.custom_toast_container));

            TextView textView = (TextView) layout.findViewById(R.id.text);
            textView.setText(text);

            Toast toast = new Toast(getApplicationContext());
            toast.setGravity(Gravity.FILL_HORIZONTAL | Gravity.BOTTOM, 0, 0);
            toast.setDuration(Toast.LENGTH_SHORT);
            toast.setView(layout);
            toast.show();

        }
        });
    }

    byte[] ShortToByte(short[] buffer) {

        int N = buffer.length;
        ByteBuffer byteBuffer = ByteBuffer.allocate(N);

        int i = 0;
        while (i < N) {

            byte b = (byte)(buffer[i] / 256);  /*convert to byte. */
            byteBuffer.put(b);
            i++;
        }
        return byteBuffer.array();
    }

    private void startRecording() {

        textView.append(String.format("\n\n[%s] Recording has started.\n", getTime()));
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


        try {

            classifier = TensorFlowImageClassifier.create(this, getAssets(), FROZEN_MODEL_FILE, LABEL_FILE, INPUT_HEIGHT, INPUT_WIDTH, INPUT_NAME, OUTPUT_NAME);
        }
        catch (final Exception e) {

            handler.post(new Runnable() { public void run() {

                textView.append(String.format("Cannot create tensorflow instance.\nError message: %s.\n", e.getMessage()));

                recordingActive = false;
                recordButton.setChecked(recordingActive);
                }
            });

            return;
        }

        new Thread(new Runnable() {

            @Override
            public void run() {

                AudioRecord record = null;

                try {

                    android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);

                    final int sampleRate = SoundParameters.MaxFrameRate;
                    final int encoding = AudioFormat.ENCODING_PCM_16BIT;

                    handler.post(new Runnable() { public void run() { textView.append("Requesting minimal buffer size for rate = " + sampleRate + ", encoding = " + encoding + ".\n"); } });

                    final int bufferSize = AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, encoding);
                    handler.post(new Runnable() { public void run() { textView.append("Minimal buffer size = " + bufferSize + ".\n"); } });

                    final int recordBufferSize = bufferSize * 8;
                    record = new AudioRecord(MediaRecorder.AudioSource.MIC, sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, recordBufferSize);
                    handler.post(new Runnable() { public void run() { textView.append("AudioRecord was created.\n"); } });

                    short[] audioBuffer = new short[recordBufferSize];

                    if (record == null || record.getState() != AudioRecord.STATE_INITIALIZED) {

                        handler.post(new Runnable() { public void run() { textView.append("Cannot initialize audio record.\n"); } });
                        return;
                    }

                    soundBuffer = new SoundBuffer(classifier, sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);

                    handler.post(new Runnable() { public void run() { textView.append(String.format("[%s] Recording is active.\n", getTime())); } });

                    ArrayList<Integer> debugInfo = new ArrayList<Integer>();

                    if (record.getState() != AudioRecord.STATE_INITIALIZED)
                    {
                        handler.post(new Runnable() { public void run() { textView.append(String.format("[%s] AudioRecord is not initialized.\n", getTime())); } });
                        return;
                    }

                    record.startRecording();

                    if (record.getRecordingState() != AudioRecord.RECORDSTATE_RECORDING)
                    {
                        handler.post(new Runnable() { public void run() { textView.append(String.format("[%s] AudioRecord is not in recording state.\n", getTime())); } });
                        return;
                    }

                    // This is workaround of audio record problem.
                    // At the beginning it rewrites audio buffer several times.
                    // Let it initialize properly.
                    Thread.sleep(512);

                    int read = 0;
                    long frames = 0;
                    while (recordingActive) {

                        read = record.read(audioBuffer, 0, recordBufferSize);

                        if (read > 0) {

                            soundBuffer.append(audioBuffer, 0, read);
                            frames += read;
                        }
                    }

                    record.stop();
                    record.release();

                    final long totalFrames = frames;

                    handler.post(new Runnable() { public void run() { textView.append(String.format("[%s] Recording was stopped.\nFrames read: %d.\n", getTime(), totalFrames)); } });
                }
                catch(final Exception e) {

                    handler.post(new Runnable() { public void run() {

                        textView.append(String.format("[%s] Recording was stopped unexpectedly.\nError message: %s.\n", getTime(), e.getMessage()));
                        e.printStackTrace();

                        }
                    });
                }
                finally {

                    handler.post(new Runnable() { public void run() {

                        recordingActive = false;
                        recordButton.setChecked(recordingActive);

                        }
                    });
                }

            }
        }).start();
    }

    private void stopRecording() {

        textView.append(String.format("[%s] Stopping recording.\n", getTime()));
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
        imageView = (ImageView) findViewById(R.id.image_view);

        textView.setMovementMethod(new ScrollingMovementMethod());
    }

    @Override
    public void onStop() {

        super.onStop();
    }
}