# TP7 report

The real TP6 full-int8 CNN (63.375 KB) was converted to
src/model_data.h. This Python byte-to-array export is equivalent to xxd -i,
with alignment and const storage added. Ten real Fashion-MNIST test images
(one per class) were quantized using the actual input scale/zero point and
exported to src/image_data.h. Input shape is [1, 28, 28, 1].

| True class | Prediction | Correct? | Desktop invoke ms |
|---|---|---|---|
| T-shirt/top | T-shirt/top | True | 0.140 |
| Trouser | Trouser | True | 0.047 |
| Pullover | Pullover | True | 0.049 |
| Dress | Dress | True | 0.042 |
| Coat | Coat | True | 0.042 |
| Sandal | Sandal | True | 0.041 |
| Shirt | T-shirt/top | False | 0.041 |
| Sneaker | Sneaker | True | 0.041 |
| Bag | Bag | True | 0.040 |
| Ankle boot | Ankle boot | True | 0.042 |

This is a ten-image **desktop** inference check, not a full-dataset accuracy
evaluation and not ESP32 latency. Firmware uses the same model and image bytes,
checks schema and tensor type/size, copies 784 bytes, invokes TFLite Micro,
finds argmax, and prints predicted/true labels. LCD shows the predicted class.
The initial tensor arena budget is 128 KB; AllocateTensors() must succeed on
the target, and arena_used_bytes()/free heap are printed. Increase the budget
only if the real allocator reports insufficient memory and the board permits it.
Weights are const data in Flash; no assumption that the file size equals RAM use.

The button uses GPIO 4 (XIAO D3). I2C uses GPIO 5/6 (XIAO D4/D5).
Press once per image; release before the next inference. The sketch also
performs an initial prediction at boot. PlatformIO/Wokwi assets are included.
Hardware execution and board latency need an actual board or simulator run;
the report does not invent those observations.

Source: https://github.com/ahmed3991/AIoT-2025/blob/main/TP7/TP7.md
