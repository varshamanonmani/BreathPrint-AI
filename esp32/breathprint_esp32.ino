/*
  BreathPrint ESP32-S3 Audio Acquisition

  Purpose:
  - Capture respiratory audio using I2S microphone
  - Prepare data transfer to AI processing device

  AI inference is performed externally
  using BreathPrint TensorFlow Lite model.
*/


#include <Arduino.h>
#include "driver/i2s.h"


// ============================
// I2S Microphone Pins
// Change according to your board
// ============================

#define I2S_WS  15
#define I2S_SD  32
#define I2S_SCK 14


#define SAMPLE_RATE 16000


// Audio buffer

int16_t audioBuffer[512];


// ============================
// I2S Configuration
// ============================

void setupI2S()
{

  i2s_config_t i2s_config =
  {
    .mode = (i2s_mode_t)
    (I2S_MODE_MASTER | I2S_MODE_RX),

    .sample_rate = SAMPLE_RATE,

    .bits_per_sample =
    I2S_BITS_PER_SAMPLE_32BIT,

    .channel_format =
    I2S_CHANNEL_FMT_ONLY_LEFT,

    .communication_format =
    I2S_COMM_FORMAT_STAND_I2S,

    .intr_alloc_flags = 0,

    .dma_buf_count = 8,

    .dma_buf_len = 64,

    .use_apll = false
  };


  i2s_pin_config_t pin_config =
  {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = -1,
    .data_in_num = I2S_SD
  };


  i2s_driver_install(
    I2S_NUM_0,
    &i2s_config,
    0,
    NULL
  );


  i2s_set_pin(
    I2S_NUM_0,
    &pin_config
  );

}


// ============================
// Setup
// ============================

void setup()
{

  Serial.begin(115200);

  delay(1000);

  Serial.println(
    "BreathPrint ESP32 Starting..."
  );


  setupI2S();


  Serial.println(
    "Microphone Ready"
  );

}


// ============================
// Main Loop
// ============================

void loop()
{

  size_t bytesRead;


  i2s_read(
    I2S_NUM_0,
    &audioBuffer,
    sizeof(audioBuffer),
    &bytesRead,
    portMAX_DELAY
  );


  Serial.print(
    "Audio samples received: "
  );


  Serial.println(
    bytesRead
  );


  delay(100);

}