A new Flutter project.

## Getting Started

This project is a starting point for a Flutter application.

A few resources to get you started if this is your first Flutter project:

- [Lab: Write your first Flutter app](https://flutter.dev/docs/get-started/codelab)
- [Cookbook: Useful Flutter samples](https://flutter.dev/docs/cookbook)

For help getting started with Flutter, view our
[online documentation](https://flutter.dev/docs), which offers tutorials,
samples, guidance on mobile development, and a full API reference.

# TensorFlow Lite Flutter Helper Library

Makes use of TensorFlow Lite Interpreter on Flutter easier by
providing simple architecture for processing and manipulating
input and output of TFLite Models.

API design and documentation is identical to the TensorFlow Lite
Android Support Library.

## Getting Started

## Screenshots 

![WhatsApp Image 2021-08-26 at 3 47 03 PM](https://user-images.githubusercontent.com/86112651/130947577-b044602d-8cfc-414b-b232-9080af3cd08d.jpeg)
![WhatsApp Image 2021-08-26 at 3 47 03 PM (1)](https://user-images.githubusercontent.com/86112651/130947613-d29e6549-eef9-49ab-ab1b-4dd9806a11fe.jpeg)
![WhatsApp Image 2021-08-26 at 3 47 03 PM (2)](https://user-images.githubusercontent.com/86112651/130947630-e1aea2fc-1b4d-4ddb-a883-e095e884b8c5.jpeg)


## Screenshots

### Setup TFLite Flutter Plugin

Include `tflite_flutter: ^<latest_version>` in your pubspec.yaml. Follow the initial setup
instructions given [here](https://github.com/am15h/tflite_flutter_plugin#most-important-initial-setup)

## Image Processing

TFLite Helper depends on [flutter image package](https://pub.dev/packages/image) internally for
Image Processing.

### Basic image manipulation and conversion

The TensorFlow Lite Support Library has a suite of basic image manipulation methods such as crop
and resize. To use it, create an `ImageProcessor` and add the required operations.
To convert the image into the tensor format required by the TensorFlow Lite interpreter,
create a `TensorImage` to be used as input:

```dart
// Initialization code
// Create an ImageProcessor with all ops required. For more ops, please
// refer to the ImageProcessor Ops section in this README.
ImageProcessor imageProcessor = ImageProcessorBuilder()
  .add(ResizeOp(224, 224, ResizeMethod.NEAREST_NEIGHBOUR))
  .build();

// Create a TensorImage object from a File
TensorImage tensorImage = TensorImage.fromFile(imageFile);

// Preprocess the image.
// The image for imageFile will be resized to (224, 224)
tensorImage = imageProcessor.process(tensorImage);
```

### Create output objects and run the model

```dart
// Create a container for the result and specify that this is a quantized model.
// Hence, the 'DataType' is defined as UINT8 (8-bit unsigned integer)
TensorBuffer probabilityBuffer =
    TensorBuffer.createFixedSize(<int>[1, 1001], TfLiteType.uint8);
```

#### Loading the model and running inference:

```dart
import 'package:tflite_flutter/tflite_flutter.dart';

try {
    // Create interpreter from asset.
    Interpreter interpreter =
        await Interpreter.fromAsset("mobilenet_v1_1.0_224_quant.tflite");
    interpreter.run(tensorImage.buffer, probabilityBuffer.buffer);
} catch (e) {
    print('Error loading model: ' + e.toString());
}
```

### Accessing the result

Developers can access the output directly through `probabilityBuffer.getDoubleList()`.
If the model produces a quantized output, remember to convert the result.
For the MobileNet quantized model, the developer needs to divide each output value by 255
to obtain the probability ranging from 0 (least likely) to 1 (most likely) for each category.


### Optional: Mapping results to labels

Developers can also optionally map the results to labels. First, copy the text
file containing labels into the module’s assets directory. Next, load the
label file using the following code:

```dart
List<String> labels = await FileUtil.loadLabels("assets/labels.txt");
```

The following snippet demonstrates how to associate the probabilities with category labels:

```dart
TensorLabel tensorLabel = TensorLabel.fromList(
      labels, probabilityProcessor.process(probabilityBuffer));

Map<String, double> doubleMap = tensorLabel.getMapWithFloatValue();
```

### ImageProcessor Architecture

The design of the ImageProcessor allowed the image manipulation operations to be defined up
front and optimised during the build process. The ImageProcessor currently supports
three basic preprocessing operations:

```dart
int cropSize = min(_inputImage.height, _inputImage.width);

ImageProcessor imageProcessor = ImageProcessorBuilder()
    // Center crop the image to the largest square possible
    .add(ResizeWithCropOrPadOp(cropSize, cropSize))
    // Resize using Bilinear or Nearest neighbour
    .add(ResizeOp(224, 224, ResizeMethod.NEAREST_NEIGHBOUR))
    // Rotation clockwise in 90 degree increments
    .add(Rot90Op(rotationDegrees ~/ 90))
    .add(NormalizeOp(127.5, 127.5))
    .add(QuantizeOp(128.0, 1 / 128.0))
    .build();

### Quantization

The TensorProcessor can be used to quantize input tensors or dequantize output tensors.
For example, when processing a quantized output TensorBuffer, the developer can use DequantizeOp to
dequantize the result to a floating point probability between 0 and 1:

```dart
// Post-processor which dequantize the result
TensorProcessor probabilityProcessor =
    TensorProcessorBuilder().add(DequantizeOp(0, 1 / 255.0)).build();
TensorBuffer dequantizedBuffer =
    probabilityProcessor.process(probabilityBuffer);
```

#### Reading Qunatization Params

```dart
// Quantization Params of input tensor at index 0
QuantizationParams inputParams = interpreter.getInputTensor(0).params;

// Quantization Params of output tensor at index 0
QuantizationParams outputParams = interpreter.getOutputTensor(0).params;
