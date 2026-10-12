"""TP7: embed the real quantized CNN and correctly quantized mock images."""
# %% 1. Locate the trained full-int8 model
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
MODEL_PATH = ROOT / "models" / "cnn_int8.tflite"
OUT = ROOT / "results"
SRC = ROOT / "src"
for directory in [MODEL_PATH.parent, OUT, SRC]:
    directory.mkdir(parents=True, exist_ok=True)
if not MODEL_PATH.exists():
    try:
        from google.colab import files
    except ImportError as error:
        raise FileNotFoundError("Copy TP6 results/cnn_int8.tflite into models/.") from error
    print("Upload cnn_int8.tflite from TP6")
    uploaded = files.upload()
    if "cnn_int8.tflite" not in uploaded:
        raise FileNotFoundError("cnn_int8.tflite was not uploaded.")
    MODEL_PATH.write_bytes(uploaded["cnn_int8.tflite"])

# %% 2. Read the real tensor shape and quantization metadata
interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH), num_threads=1)
interpreter.allocate_tensors()
input_info = interpreter.get_input_details()[0]
output_info = interpreter.get_output_details()[0]
assert input_info["dtype"] == np.int8 and tuple(input_info["shape"]) == (1, 28, 28, 1)
assert output_info["dtype"] == np.int8
scale, zero_point = input_info["quantization"]
print("Input:", input_info["shape"], input_info["dtype"], scale, zero_point)
print("Operators:", [item["op_name"] for item in interpreter._get_ops_details()])

# %% 3. Convert the binary to an aligned constant C array (xxd -i equivalent)
binary = MODEL_PATH.read_bytes()
hex_lines = [", ".join(f"0x{value:02x}" for value in binary[i:i + 16])
             for i in range(0, len(binary), 16)]
model_header = (
    "#pragma once\n#include <stdint.h>\n"
    "alignas(16) const unsigned char fashion_mnist_cnn_int8_tflite[] = {\n  "
    + ",\n  ".join(hex_lines) + "\n};\n"
    + f"const unsigned int fashion_mnist_cnn_int8_tflite_len = {len(binary)};\n"
)
(SRC / "model_data.h").write_text(model_header, encoding="utf-8")

# %% 4. Select one real test image per class and quantize using model metadata
(_, _), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
indices = [int(np.flatnonzero(test_labels == label)[0]) for label in range(10)]
raw_images = test_images[indices]
normalized = raw_images.astype(np.float32) / 255.0
quantized = np.clip(np.rint(normalized / scale + zero_point), -128, 127).astype(np.int8)
image_header = "#pragma once\n#include <stdint.h>\nconst int NUM_IMAGES = 10;\n"
image_header += "const int8_t image_data[NUM_IMAGES][784] = {\n"
for image in quantized:
    image_header += "  {" + ", ".join(str(int(value)) for value in image.flatten()) + "},\n"
image_header += "};\nconst int label_list[NUM_IMAGES] = {0,1,2,3,4,5,6,7,8,9};\n"
image_header += "const char* class_names[10] = {" + ", ".join(json.dumps(name) for name in class_names) + "};\n"
(SRC / "image_data.h").write_text(image_header, encoding="utf-8")
np.savez(OUT / "mock_images.npz", raw_images=raw_images, labels=np.arange(10),
         quantized=quantized, dataset_indices=np.asarray(indices))
metadata = {"shape": input_info["shape"].tolist(), "scale": float(scale),
            "zero_point": int(zero_point), "labels": class_names}
(OUT / "input_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

# %% 5. Verify all mock images using the actual desktop TFLite interpreter
rows = []
for label, image in enumerate(quantized):
    interpreter.set_tensor(input_info["index"], image.reshape(input_info["shape"]))
    start = perf_counter()
    interpreter.invoke()
    elapsed_ms = (perf_counter() - start) * 1000
    output = interpreter.get_tensor(output_info["index"])[0]
    predicted = int(np.argmax(output))
    rows.append({"True_label": class_names[label], "Predicted_label": class_names[predicted],
                 "Correct": predicted == label, "Desktop_invoke_ms": elapsed_ms})
results = pd.DataFrame(rows)
results.to_csv(OUT / "predictions.csv", index=False)
print(results.to_string(index=False))
table = "| True class | Prediction | Correct? | Desktop invoke ms |\n|---|---|---|---|\n"
for row in rows:
    table += f"| {row['True_label']} | {row['Predicted_label']} | {row['Correct']} | {row['Desktop_invoke_ms']:.3f} |\n"
report = f"""# TP7 report

The real TP6 full-int8 CNN ({len(binary)/1024:.3f} KB) was converted to
src/model_data.h. This Python byte-to-array export is equivalent to xxd -i,
with alignment and const storage added. Ten real Fashion-MNIST test images
(one per class) were quantized using the actual input scale/zero point and
exported to src/image_data.h. Input shape is {metadata['shape']}.

{table}
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
"""
(OUT / "REPORT.md").write_text(report, encoding="utf-8")
print("Exported model_data.h and image_data.h in", SRC)
