package com.example.tfliteprototype;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;



public class MainActivity extends AppCompatActivity {

    private static final int PICK_IMAGE = 1;
    Button button;
    private ImageView imageView;
    protected TextView recognitionTextView;
    /**
     * Name of the model,label file stored in Assets.
     */
    private static final String MODEL_PATH = "converted_model.tflite";
    private static final String LABEL_PATH = "labels.txt";
    /**
     * Interpreter to run model with Tensorflow Lite.
     */
    private Interpreter tflite;
    /**
     * Labels corresponding to the output of the vision model.
     */
    private List<String> labelList;
    /**
     * A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs.
     */
    private ByteBuffer imgData = null;
    /**
     * An array to hold inference results, to be feed into Tensorflow Lite as outputs. Holds output.
     */
    private float[][] labelProbArray = null;
    /**
     * Dimensions of inputs.
     */
    static final int DIM_IMG_SIZE_X = 160;
    static final int DIM_IMG_SIZE_Y = 160;

    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;

    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE = 3;
    /**
     *  Pre-allocated buffers for storing image data in.
     */
    private int[] intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Context activity = getApplicationContext();
        verifyPermissions();

        imageView = (ImageView)findViewById(R.id.imageView1);
        button = (Button)findViewById(R.id.buttonLoadPicture);
        recognitionTextView = (TextView) findViewById(R.id.recognitionTextView);

        try {
            tflite = new Interpreter(loadModelFile(activity));
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            labelList = loadLabelList(activity);
        } catch (IOException e) {
            e.printStackTrace();
        }
        imgData = ByteBuffer.allocateDirect(4 * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
        imgData.order(ByteOrder.nativeOrder());
        labelProbArray = new float[1][1];
        Log.d("TFlite: onCreate()", "Created a Tensorflow Lite binary Image Classifier.");

        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                pickFromGallery();
            }
        });

    }

    private void pickFromGallery(){
        //Create an Intent with action as ACTION_PICK
        Intent intent=new Intent(Intent.ACTION_PICK,MediaStore.Images.Media.INTERNAL_CONTENT_URI);
        // Sets the type as image/*. This ensures only components of type image are selected
        intent.setType("image/*");
        // Launching the Intent
        startActivityForResult(intent,PICK_IMAGE);
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        // Result code is RESULT_OK only if the user selects an Image
        if (data != null && resultCode == RESULT_OK && requestCode == PICK_IMAGE){
            //data.getData return the content URI for the selected Image
            Uri selectedImage = data.getData();
            Log.d("Uri of Selected image", String.valueOf(selectedImage));
            imageView.setImageURI(Uri.parse(String.valueOf(selectedImage)));

            Bitmap bitmap = ((BitmapDrawable)imageView.getDrawable()).getBitmap();
            Log.d("Bitmap to be used", String.valueOf(bitmap));
            //Calling classifyImages(bitmap) to feed it into model
            classifyImages(bitmap);

        }else{ Log.d("Eror:onActivityResult()","No data"); }

    }

    /**
     * Classifies the given bitmap .Runs model.
     */
    public void classifyImages(Bitmap bitmap) {
        if (tflite == null) {
            Log.e("TfLite:classifyImages()", "Image classifier has not been initialized; Skipped.");
            return ;
        }
        tflite.setUseNNAPI(false); //Since we are not using GPU
        Log.d("TfLite:classifyImages()", "Image classifier has been initialized;");

        Bitmap reshapeBitmap = Bitmap.createScaledBitmap(bitmap, 160, 160, false);
        convertBitmapToByteBuffer(reshapeBitmap);

        // Here's where the magic happens!
        long startTime = SystemClock.uptimeMillis();
        tflite.run(imgData, labelProbArray); //output shape (1,1)
        long endTime = SystemClock.uptimeMillis();
        Log.d("TfLite:classifyImages()", "Time-cost to run model inference: " + Long.toString(endTime - startTime)+" ms");

        float output = labelProbArray[0][0]; //output in range (0,1) since we used sigmoid activation function in last layer of our Neural Network
        Log.e("Model Output: ", "Output : " + output);

        CharSequence result;
        if(output < 0.5) {
            result = "MEME";
            recognitionTextView.setText(result);
            //Toast.makeText(MainActivity.this,result, Toast.LENGTH_LONG).show();
        } else {
            result = "NOT MEME";
            recognitionTextView.setText(result);
            //Toast.makeText(MainActivity.this,result, Toast.LENGTH_LONG).show();
        }
    }

    /**
     * Writes Image data into a {@code ByteBuffer}.
     */
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // Convert the image to floating point.
        int pixel = 0;
        long startTime = SystemClock.uptimeMillis();
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                imgData.putFloat((((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                imgData.putFloat((((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                imgData.putFloat((((val) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
            }
        }
        long endTime = SystemClock.uptimeMillis();
        Log.d("BitmapToByteBuffer", "Time-cost to put values into ByteBuffer: " + Long.toString(endTime - startTime) + " ms");
    }

    /**
     * Memory-map the model file in Assets.
     */
    private MappedByteBuffer loadModelFile(Context activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Reads label list from Assets.
     */
    private List<String> loadLabelList(Context activity) throws IOException {
        List<String> labelList = new ArrayList<String>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(activity.getAssets().open(LABEL_PATH)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }


    private void verifyPermissions(){
        Log.i("PERMISSIONS","verifyPermissions(): Asking for user permissions");

        String [] Permissions = {Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE};

        if(ContextCompat.checkSelfPermission(this.getApplicationContext(),Permissions[0]) == PackageManager.PERMISSION_GRANTED &&
                ContextCompat.checkSelfPermission(this.getApplicationContext(),Permissions[1]) == PackageManager.PERMISSION_GRANTED ){
            Toast.makeText(MainActivity.this,"Permission granted", Toast.LENGTH_SHORT).show();

        }else{
            ActivityCompat.requestPermissions(this,Permissions,1);
        }

    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if(requestCode == 1){
            if(grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED){
                Toast.makeText(MainActivity.this,"Permission granted", Toast.LENGTH_SHORT).show();
            }else{
                Toast.makeText(MainActivity.this,"Permission Denied", Toast.LENGTH_SHORT).show();
            }
        }
    }


}
