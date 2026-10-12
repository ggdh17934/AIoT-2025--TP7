// TP7: actual full-int8 CNN inference on the XIAO ESP32-S3.
#include <Arduino.h>
#include <Wire.h>
#include <MicroTFLite.h>
#include <LiquidCrystal_I2C.h>
#include "model_data.h"
#include "image_data.h"

const int BUTTON_PIN = 4; // XIAO D3; button to 3V3, external pulldown to GND
const int TENSOR_ARENA_SIZE = 128 * 1024;
alignas(16) uint8_t tensor_arena[TENSOR_ARENA_SIZE];
LiquidCrystal_I2C lcd(0x27, 16, 2);
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
tflite::AllOpsResolver resolver;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;
bool previousButton = false;
int imageIndex = 0;

void fail(const char *message) {
  Serial.println(message);
  lcd.clear(); lcd.print("Error - Serial");
  while (true) delay(100);
}

void predictImage() {
  memcpy(input->data.int8, image_data[imageIndex], 784);
  const unsigned long start = micros();
  if (interpreter->Invoke() != kTfLiteOk) fail("Invoke failed");
  const unsigned long elapsed = micros() - start;
  int predicted = 0;
  for (int i = 1; i < 10; ++i)
    if (output->data.int8[i] > output->data.int8[predicted]) predicted = i;
  Serial.printf("Image=%d True=%s Predicted=%s Inference_us=%lu\n",
                imageIndex, class_names[label_list[imageIndex]], class_names[predicted], elapsed);
  Serial.printf("Arena used=%u, Free heap=%u\n",
                unsigned(interpreter->arena_used_bytes()), ESP.getFreeHeap());
  lcd.clear(); lcd.print("Image " + String(imageIndex));
  lcd.setCursor(0, 1); lcd.print(class_names[predicted]);
  imageIndex = (imageIndex + 1) % NUM_IMAGES;
}

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT);
  Wire.begin(5, 6);
  lcd.init(); lcd.backlight();
  model = tflite::GetModel(fashion_mnist_cnn_int8_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) fail("Schema mismatch");
  static tflite::MicroInterpreter runner(model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
  interpreter = &runner;
  if (interpreter->AllocateTensors() != kTfLiteOk) fail("Tensor arena allocation failed");
  input = interpreter->input(0);
  output = interpreter->output(0);
  if (input->type != kTfLiteInt8 || input->bytes != 784 ||
      output->type != kTfLiteInt8 || output->bytes != 10) fail("Unexpected tensor type/size");
  Serial.printf("Model bytes=%u, Arena budget=%d\n",
                fashion_mnist_cnn_int8_tflite_len, TENSOR_ARENA_SIZE);
  predictImage();
}

void loop() {
  const bool pressed = digitalRead(BUTTON_PIN) == HIGH;
  if (pressed && !previousButton) { predictImage(); delay(50); }
  previousButton = pressed;
  delay(10);
}
