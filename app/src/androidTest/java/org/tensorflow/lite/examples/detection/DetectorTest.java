/*
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import static androidx.test.core.app.ApplicationProvider.getApplicationContext;
import static com.google.common.truth.Truth.assertThat;
import static java.lang.Math.abs;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.round;

import android.content.ContentResolver;
import android.content.ContentUris;
import android.content.ContentValues;
import android.content.Context;
import android.content.res.AssetManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.graphics.drawable.Drawable;
import android.media.MediaMetadataRetriever;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Environment;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Size;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import android.util.Log;
import android.widget.ImageView;
import android.view.View;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.BufferedReader;
import java.io.OutputStream;
import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.List;
import java.util.Scanner;
import java.util.Vector;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.Classifier.Recognition;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;

import wseemann.media.FFmpegMediaMetadataRetriever;

/** Golden test for Object Detection Reference app. */
@RunWith(AndroidJUnit4.class)
public class DetectorTest {

  private static final Size MODEL_INPUT_SIZE = new Size(704, 320);
  private static final boolean IS_MODEL_QUANTIZED = false;  // model input is flaot32 for compatiblity.
  private static final String MODEL_FILE = "f4aec0_v7.0rc/sdrs_pvb_det_model_v9.1_int8_1x320x704x3.tflite";
  //private static final String MODEL_FILE = "f4aec0_v7.0rc/tflite_int8.tflite";
  private static final String LABELS_FILE = "file:///android_asset/labelmap_sdrs_v7.txt";
  private static final String IMAGE_LIST_FILE = "image_list.txt";
  private static final String IMAGE_DIR = "images";
  private static final Size IMAGE_SIZE = new Size(1440, 1080);
  private static final float CROP_XL = 0.0f;
  private static final float CROP_YT = 0.32f;
  private static final float CROP_XR = 1.0f;
  private static final float CROP_YB = 1.0f;
  private static final boolean USE_NNAPI = true;

  private Classifier detector;
  private Bitmap croppedBitmap;
  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  @Before
  public void setUp() throws IOException {
    AssetManager assetManager =
        InstrumentationRegistry.getInstrumentation().getContext().getAssets();
    int cropWidth = MODEL_INPUT_SIZE.getWidth();
    int cropHeight = MODEL_INPUT_SIZE.getHeight();
    detector =
        TFLiteObjectDetectionAPIModel.create(
            assetManager,
            MODEL_FILE,
            LABELS_FILE,
            cropWidth,
            cropHeight,
            IS_MODEL_QUANTIZED);
    detector.setUseNNAPI(USE_NNAPI);
    int previewWidth = IMAGE_SIZE.getWidth();
    int previewHeight = IMAGE_SIZE.getHeight();
    int sensorOrientation = 0;
    // ref. https://developer.android.com/reference/android/graphics/Bitmap#createBitmap(int,%20int,%20android.graphics.Bitmap.Config)
    croppedBitmap = Bitmap.createBitmap(MODEL_INPUT_SIZE.getWidth(), MODEL_INPUT_SIZE.getHeight(), Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropWidth, cropHeight,
            sensorOrientation, false);
    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);
  }

  @Test
  public void runDetectorOnImages() throws Exception {
    String dirname = "/indonesia_data/vehicle_rear_sample/images/";
    String resultText = "";
    String inferenceTimeText = "";
    String metaText = "";
    metaText += String.format("model: %s\n", MODEL_FILE);
    metaText += String.format("use_nnapi: %b\n", USE_NNAPI);

    String tag = "DetectorTest";
    Log.v(tag, "start DetectorOnImages");
    Log.v(tag, metaText);

    Uri contentUri = MediaStore.Files.getContentUri("external");
    String selection = MediaStore.MediaColumns.RELATIVE_PATH + "=?";
    String[] selectionArgs = new String[]{Environment.DIRECTORY_PICTURES + dirname};

    Context context = InstrumentationRegistry.getInstrumentation().getContext();
    ContentResolver resolver = context.getContentResolver();

    AssetManager assetManager =
        InstrumentationRegistry.getInstrumentation().getContext().getAssets();
    Vector<String> image_files = new Vector<String>();
    InputStream imageFilesInput = assetManager.open(IMAGE_LIST_FILE);
    BufferedReader br = new BufferedReader(new InputStreamReader(imageFilesInput));
    String line;
    while ((line = br.readLine()) != null) {
      image_files.add(line);
    }
    br.close();

    Uri uri = null;
    //ImageView mImageView;
    //mImageView = (ImageView) findViewById(R.id.myImageview);
    for (int i = 0; i < image_files.size(); i++) {

      String filename = IMAGE_DIR + "/" + (String)image_files.get(i);
      InputStream inputStream = assetManager.open(filename);
      /*InputStream ims = null;
      try {
        ims = assetManager.open(filename);
      } catch (IOException e) {
        e.printStackTrace();
      }*/
      // load image as Drawable
      //Drawable d = Drawable.createFromStream(ims, null);
      //mImageView.setImageDrawable(d);
      Canvas canvas = new Canvas(croppedBitmap);

      // calculate cropping parameters
      int crop_x = Math.round(IMAGE_SIZE.getWidth() * CROP_XL);
      int crop_y = Math.round(IMAGE_SIZE.getHeight() * CROP_YT); 
      int crop_w = Math.round(IMAGE_SIZE.getWidth() * (CROP_XR - CROP_XL));
      int crop_h = Math.round(IMAGE_SIZE.getHeight() * (CROP_YB - CROP_YT));

      Bitmap croppedBitmapOriginalSize = Bitmap.createBitmap(loadImageFromInputStream(
            inputStream), crop_x, crop_y, crop_w, crop_h);
      canvas.drawBitmap(croppedBitmapOriginalSize, frameToCropTransform, null);

      Log.v(tag, "Process " + filename);

      // run inference
      final long startTime = SystemClock.uptimeMillis();
      final List<Recognition> results = detector.recognizeImage(croppedBitmap);
      final long lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

      Log.v(tag, "recognition result: " + results);
      //File imgFile = new  File(filename);
      //if(imgFile.exists()){
      //  Bitmap myBitmap = BitmapFactory.decodeFile(imgFile.getAbsolutePath());
      //  //ImageView myImageview = (ImageView) findViewById(R.id.myImageView);
      //  ImageView imageView = new ImageView(getApplicationContext());
      //  imageView.setImageBitmap(myBitmap);
      //}
      // get result
      for (Recognition item : results) {
        RectF bbox = new RectF();
        cropToFrameTransform.mapRect(bbox, item.getLocation());
        String resultTextSingle = String.format("%s,%s,%f,%f,%f,%f,%f\n",
                filename,
                item.getTitle(),
                bbox.left * crop_w + crop_x,
                bbox.top * crop_h + crop_y,
                bbox.right * crop_w + crop_x,
                bbox.bottom * crop_h + crop_y,
                item.getConfidence());
        Log.v(tag, "detection result: " + resultTextSingle);
        resultText += resultTextSingle;
      }

      String inferenceTimeTextSingle = String.format("%s,%d\n", filename, lastProcessingTimeMs);
      Log.v(tag, "inference time [ms]: " + inferenceTimeTextSingle);
      inferenceTimeText += inferenceTimeText;
    }

    // save result
    ContentValues values = new ContentValues();
    Calendar cal = Calendar.getInstance();
    Date date = cal.getTime();
    SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd_HH:mm:ss");
    String formattedDate = dateFormat.format(date);
    dirname = "/TensorFlowLiteDetections/runDetectorOnImages_" + formattedDate;
    if (USE_NNAPI) {
      dirname += "_nnapi";
    } else {
      dirname += "_tfliteapi";
    }
    String filename = String.format("%s.txt", "detections");

    values.put(MediaStore.MediaColumns.DISPLAY_NAME, filename);
    values.put(MediaStore.MediaColumns.MIME_TYPE, "text/plain");
    values.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOCUMENTS + dirname);
    uri = resolver.insert(MediaStore.Files.getContentUri("external"), values);

    OutputStream outputStream = resolver.openOutputStream(uri);
    outputStream.write(resultText.getBytes());
    outputStream.close();

    // Write inference time
    filename = String.format("%s.txt", "inference_time");
    values.put(MediaStore.MediaColumns.DISPLAY_NAME, filename);
    uri = resolver.insert(MediaStore.Files.getContentUri("external"), values);

    outputStream = resolver.openOutputStream(uri);
    outputStream.write(inferenceTimeText.getBytes());
    outputStream.close();

    // Write meta info
    filename = String.format("%s.txt", "meta");
    values.put(MediaStore.MediaColumns.DISPLAY_NAME, filename);
    uri = resolver.insert(MediaStore.Files.getContentUri("external"), values);

    outputStream = resolver.openOutputStream(uri);
    outputStream.write(metaText.getBytes());
    outputStream.close();
  }

  // @Test
  public void detectionResultsShouldNotChange() throws Exception {
    Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(loadImage("table.jpg"), frameToCropTransform, null);
    final List<Recognition> results = detector.recognizeImage(croppedBitmap);
    final List<Recognition> expected = loadRecognitions("table_results.txt");

    for (Recognition target : expected) {
      // Find a matching result in results
      boolean matched = false;
      for (Recognition item : results) {
        RectF bbox = new RectF();
        cropToFrameTransform.mapRect(bbox, item.getLocation());
        if (item.getTitle().equals(target.getTitle())
            && matchBoundingBoxes(bbox, target.getLocation())
            && matchConfidence(item.getConfidence(), target.getConfidence())) {
          matched = true;
          break;
        }
      }
      System.out.println("target: " + target.getTitle());
      assertThat(matched).isTrue();
    }
  }

  // Confidence tolerance: absolute 1%
  private static boolean matchConfidence(float a, float b) {
    return abs(a - b) < 0.01;
  }

  // Bounding Box tolerance: overlapped area > 95% of each one
  private static boolean matchBoundingBoxes(RectF a, RectF b) {
    float areaA = a.width() * a.height();
    float areaB = b.width() * b.height();
    RectF overlapped =
        new RectF(
            max(a.left, b.left), max(a.top, b.top), min(a.right, b.right), min(a.bottom, b.bottom));
    float overlappedArea = overlapped.width() * overlapped.height();
    return overlappedArea > 0.95 * areaA && overlappedArea > 0.95 * areaB;
  }

  private static Bitmap loadImage(String fileName) throws Exception {
    AssetManager assetManager =
        InstrumentationRegistry.getInstrumentation().getContext().getAssets();
    InputStream inputStream = assetManager.open(fileName);
    Bitmap imageBitmap = BitmapFactory.decodeStream(inputStream);
    if (imageBitmap.getWidth() != IMAGE_SIZE.getWidth() || imageBitmap.getHeight() != IMAGE_SIZE.getHeight()) {
      imageBitmap = Bitmap.createScaledBitmap(imageBitmap, IMAGE_SIZE.getWidth(), IMAGE_SIZE.getWidth(), true);
    }
    return imageBitmap;
  }

  private static Bitmap loadImageFromInputStream(InputStream is) throws Exception {
    Bitmap imageBitmap = BitmapFactory.decodeStream(is);
    if (imageBitmap.getWidth() != IMAGE_SIZE.getWidth() || imageBitmap.getHeight() != IMAGE_SIZE.getHeight()) {
      imageBitmap = Bitmap.createScaledBitmap(imageBitmap, IMAGE_SIZE.getWidth(), IMAGE_SIZE.getWidth(), true);
    }
    return imageBitmap;
  }

  // The format of result:
  // category bbox.left bbox.top bbox.right bbox.bottom confidence
  // ...
  // Example:
  // Apple 99 25 30 75 80 0.99
  // Banana 25 90 75 200 0.98
  // ...
  private static List<Recognition> loadRecognitions(String fileName) throws Exception {
    AssetManager assetManager =
        InstrumentationRegistry.getInstrumentation().getContext().getAssets();
    InputStream inputStream = assetManager.open(fileName);
    Scanner scanner = new Scanner(inputStream);
    List<Recognition> result = new ArrayList<>();
    while (scanner.hasNext()) {
      String category = scanner.next();
      category = category.replace('_', ' ');
      if (!scanner.hasNextFloat()) {
        break;
      }
      float left = scanner.nextFloat();
      float top = scanner.nextFloat();
      float right = scanner.nextFloat();
      float bottom = scanner.nextFloat();
      RectF boundingBox = new RectF(left, top, right, bottom);
      float confidence = scanner.nextFloat();
      Recognition recognition = new Recognition(null, category, confidence, boundingBox);
      result.add(recognition);
    }
    return result;
  }
}
