#ifndef IMAGE_PROVIDER_H
#define IMAGE_PROVIDER_H

#include <cstdint>

#define TF_INPUT_WIDTH 48
#define TF_INPUT_HEIGHT 48
#define TF_INPUT_CHANNELS 3

bool InitCamera();

// image_data: ponteiro para o buffer de entrada do TFLite (tensor->data.int8)
bool GetImage(int8_t *image_data);

#endif
