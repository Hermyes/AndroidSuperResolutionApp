package me.aslan.superapp;
//7045543
import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.OptIn;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.imgproc.Imgproc;

import ai.onnxruntime.*;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.concurrent.ExecutionException;

public class MainActivity extends AppCompatActivity {

    static {
        if (!OpenCVLoader.initDebug()) {
            Log.e("OpenCV", "Unable to load OpenCV");
        } else {
            System.loadLibrary("opencv_java4");
        }
    }
    private boolean isUpscaleEnabled = true;
    private static final int REQUEST_CODE_PERMISSIONS = 10;
    private final String[] REQUIRED_PERMISSIONS = new String[]{Manifest.permission.CAMERA};

    private PreviewView previewView;
    private ImageView upscaledImageView;
    private TextView blurTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {


        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Button toggleButton = findViewById(R.id.toggleUpscaleButton);
        toggleButton.setOnClickListener(v -> {
            isUpscaleEnabled = !isUpscaleEnabled;
            toggleButton.setText(isUpscaleEnabled ? "Отключить апскейл" : "Включить апскейл");
        });
        previewView = findViewById(R.id.previewView);
        upscaledImageView = findViewById(R.id.upscaledImageView);
        blurTextView = findViewById(R.id.blurLevelText);

        if (allPermissionsGranted()) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                Preview preview = new Preview.Builder().build();
                CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), this::analyzeFrame);
                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);

            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    @OptIn(markerClass = ExperimentalGetImage.class)
    private void analyzeFrame(@NonNull ImageProxy imageProxy) {
        Image mediaImage = imageProxy.getImage();
        if (mediaImage != null) {
            Bitmap bitmap = toBitmap(mediaImage);
            double blur = calculateBlur(bitmap);

            runOnUiThread(() -> blurTextView.setText(String.format("Blur: %.2f", blur)));

            if (isUpscaleEnabled && blur < 100.0) {
                Bitmap upscaled = applyOnnxUpscale(bitmap);
                Bitmap rotated = rotateBitmap(upscaled, 90);

                runOnUiThread(() -> {
                    previewView.setVisibility(View.GONE);
                    upscaledImageView.setVisibility(View.VISIBLE);
                    upscaledImageView.setImageBitmap(rotated);
                });
            } else {
                runOnUiThread(() -> {
                    upscaledImageView.setVisibility(View.GONE);
                    previewView.setVisibility(View.VISIBLE);
                });
            }
        }
        imageProxy.close();
    }

    private Bitmap toBitmap(Image image) {
        ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
        ByteBuffer uBuffer = image.getPlanes()[1].getBuffer();
        ByteBuffer vBuffer = image.getPlanes()[2].getBuffer();
        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();
        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);
        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new android.graphics.Rect(0, 0, image.getWidth(), image.getHeight()), 100, out);
        return android.graphics.BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size());
    }

    private double calculateBlur(Bitmap bitmap) {
        Mat mat = new Mat();
        Utils.bitmapToMat(bitmap, mat);
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2GRAY);
        Mat laplacian = new Mat();
        Imgproc.Laplacian(mat, laplacian, CvType.CV_64F);
        MatOfDouble std = new MatOfDouble();
        Core.meanStdDev(laplacian, new MatOfDouble(), std);
        return std.get(0, 0)[0] * std.get(0, 0)[0];
    }

    private Bitmap rotateBitmap(Bitmap bitmap, float angle) {
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }

    private Bitmap applyOnnxUpscale(Bitmap inputBitmap) {
        try {
            // Загружаем модель
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            InputStream is = getAssets().open("espcn_x4.onnx");
            byte[] modelBytes = new byte[is.available()];
            is.read(modelBytes);
            is.close();
            OrtSession session = env.createSession(modelBytes, options);

            // Преобразуем в grayscale
            Bitmap grayBitmap = Bitmap.createBitmap(inputBitmap.getWidth(), inputBitmap.getHeight(), Bitmap.Config.ARGB_8888);
            android.graphics.Canvas canvas = new android.graphics.Canvas(grayBitmap);
            android.graphics.Paint paint = new android.graphics.Paint();
            android.graphics.ColorMatrix cm = new android.graphics.ColorMatrix();
            cm.setSaturation(0);
            android.graphics.ColorMatrixColorFilter f = new android.graphics.ColorMatrixColorFilter(cm);
            paint.setColorFilter(f);
            canvas.drawBitmap(inputBitmap, 0, 0, paint);

            // Масштабируем до 200x200 (размер входа модели)
            grayBitmap = Bitmap.createScaledBitmap(grayBitmap, 200, 200, true);

            // Готовим входной буфер (1 канал)
            float[] inputBuffer = new float[1 * 1 * 200 * 200];
            int[] pixels = new int[200 * 200];
            grayBitmap.getPixels(pixels, 0, 200, 0, 0, 200, 200);

            for (int y = 0; y < 200; y++) {
                for (int x = 0; x < 200; x++) {
                    int gray = Color.red(pixels[y * 200 + x]); // grayscale: R = G = B
                    inputBuffer[y * 200 + x] = gray / 255.0f;
                }
            }

            // Создаём тензор
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputBuffer), new long[]{1, 1, 200, 200});
            OrtSession.Result result = session.run(Collections.singletonMap(session.getInputNames().iterator().next(), inputTensor));
            float[][][][] output = (float[][][][]) result.get(0).getValue();

            // Достаём размеры
            int outH = output[0][0].length;
            int outW = output[0][0][0].length;

            // Преобразуем обратно в grayscale Bitmap
            Bitmap outputBitmap = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888);
            for (int y = 0; y < outH; y++) {
                for (int x = 0; x < outW; x++) {
                    int v = (int) (Math.min(1.0f, Math.max(0f, output[0][0][y][x])) * 255);
                    outputBitmap.setPixel(x, y, Color.rgb(v, v, v));
                }
            }

            return outputBitmap;
        } catch (Exception e) {
            Log.e("ONNX", "Upscale error: " + e.getMessage());
            return inputBitmap;
        }
    }


    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) return false;
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) startCamera();
            else {
                Toast.makeText(this, "CAMERA permission required", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }
}
