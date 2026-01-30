#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Adicione este parâmetro float gamma
void classifier_init(float gamma); 

float classifier_predict(uint8_t* img_buffer, size_t img_len);

#ifdef __cplusplus
}
#endif