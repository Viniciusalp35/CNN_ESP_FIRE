#include "classifier.h"
#include "fire_model.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "img_converters.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <math.h>

static const char *TAG = "CLASS";

// Dimensões da Imagem Fonte ov3660
#define SRC_W 320
#define SRC_H 240

// Entradas da IA (
#define DST_W 96
#define DST_H 96

// Tamanho da Arena Seguro
const int kTensorArenaSize = 250 * 1024; 

// ================= GLOBAIS =================
static uint8_t *tensor_arena = nullptr;
static const tflite::Model *model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input = nullptr;
static TfLiteTensor *output = nullptr;
static tflite::MicroMutableOpResolver<25> resolver;
static uint8_t gamma_lut[256];

// Look Up Table de Correção Gama, para um menor custo computacional
static void build_gamma_table(float gamma) {
    if (gamma <= 0.0f) gamma = 0.1f;
    for (int i = 0; i < 256; i++) {
        float norm = (float)i / 255.0f;
        float res = powf(norm, gamma) * 255.0f;
        if (res > 255.0f) res = 255.0f;
        if (res < 0.0f) res = 0.0f;
        gamma_lut[i] = (uint8_t)res;
    }
}

// Função de Inicialização do Classificador
void classifier_init(float gamma) {
    ESP_LOGI(TAG, "Iniciando Classificador (Square Crop Mode)");
    build_gamma_table(gamma);

    // Aloca memória para o TensorFlow (SPIRAM preferencialmente)
    tensor_arena = (uint8_t *)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);
    
    if (!tensor_arena) {
        // Fallback para memória interna se SPIRAM falhar
        tensor_arena = (uint8_t *)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL);
    }

    if (!tensor_arena) {
        ESP_LOGE(TAG, "ERRO CRITICO: Falha de Memoria!");
        return;
    }

    model = tflite::GetModel(fire_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Erro versao modelo");
        return;
    }

    // Adiciona as operações suportadas pelo modelo
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAveragePool2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddAdd();
    resolver.AddMean();
    resolver.AddPad();
    resolver.AddConcatenation();
    resolver.AddMul();
    resolver.AddMinimum();
    resolver.AddMaximum();
    resolver.AddLogistic();
    resolver.AddQuantize();
    resolver.AddDequantize();

    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors falhou");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
    ESP_LOGI(TAG, "Classificador Pronto");
}

// Função de Predição do Classificador
float classifier_predict(uint8_t *jpg_buf, size_t jpg_len) {
    // 1. Verificações de Segurança
    if (!interpreter || !input || !jpg_buf) return 0.0f;

    // 2. Decode JPEG para RGB888 (Usa SPIRAM para o buffer temporário)
    uint8_t *rgb = (uint8_t *)heap_caps_malloc(SRC_W * SRC_H * 3, MALLOC_CAP_SPIRAM);
    if (!rgb || !fmt2rgb888(jpg_buf, jpg_len, PIXFORMAT_JPEG, rgb)) {
        if (rgb) free(rgb);
        ESP_LOGE(TAG, "Falha no Decode JPEG");
        return 0.0f;
    }

    // 3. Geometria de Recorte (Square Crop)
    // Objetivo: Pegar um quadrado de 240x240 do centro da imagem de 320x240
    int crop_size = SRC_H; // 240 pixels (altura total)
    int start_x = (SRC_W - crop_size) / 2; // (320 - 240) / 2 = 40 pixels de margem à esquerda
    
    // Razão de redução (240 / 96 = 2.5)
    float ratio = (float)crop_size / DST_W;

    // 4. Preparação dos Ponteiros do Tensor
    TfLiteType in_type = input->type;
    uint8_t *in_u8 = input->data.uint8;
    int8_t *in_i8 = (int8_t *)input->data.data;
    float *in_f = input->data.f;
    float scale = input->params.scale;
    int32_t zero = input->params.zero_point;

    // 5. Loop de Processamento (Resize + Crop + Gamma + Normalização)
    for (int y = 0; y < DST_H; y++) {
        // Mapeia Y destino -> Y fonte
        int sy = (int)(y * ratio);
        if (sy >= SRC_H) sy = SRC_H - 1;

        // Otimização: ponteiro para o início da linha
        uint8_t *src_row = rgb + (sy * SRC_W * 3);

        for (int x = 0; x < DST_W; x++) {
            // Mapeia X destino -> X fonte (com deslocamento start_x)
            int sx = start_x + (int)(x * ratio);
            
            // Proteção de limites
            if (sx < 0) sx = 0;
            if (sx >= SRC_W) sx = SRC_W - 1;

            int src_idx = sx * 3;
            int dst_idx = (y * DST_W + x) * 3;

            // Aplica Gamma (LUT)
            uint8_t r = gamma_lut[src_row[src_idx + 0]];
            uint8_t g = gamma_lut[src_row[src_idx + 1]];
            uint8_t b = gamma_lut[src_row[src_idx + 2]];

            // Preenche o Tensor de Entrada
            if (in_type == kTfLiteUInt8) {
                in_u8[dst_idx + 0] = r;
                in_u8[dst_idx + 1] = g;
                in_u8[dst_idx + 2] = b;
            } else if (in_type == kTfLiteInt8) {
                in_i8[dst_idx + 0] = (int8_t)((r / 255.0f) / scale + zero);
                in_i8[dst_idx + 1] = (int8_t)((g / 255.0f) / scale + zero);
                in_i8[dst_idx + 2] = (int8_t)((b / 255.0f) / scale + zero);
            } else {
                in_f[dst_idx + 0] = r / 255.0f;
                in_f[dst_idx + 1] = g / 255.0f;
                in_f[dst_idx + 2] = b / 255.0f;
            }
        }
    }

    // 6. Libera memória temporária 
    free(rgb);

    // 7. Executa a Inferência
    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke falhou");
        return 0.0f;
    }

    // 8. Processa a Saída
    float prob = 0.0f;

    int fire_idx = (output->dims->data[1] == 1) ? 0 : 1; 

    if (output->type == kTfLiteFloat32) {
        prob = output->data.f[fire_idx];
    } else if (output->type == kTfLiteUInt8) {
        prob = output->data.uint8[fire_idx] / 255.0f;
    } else if (output->type == kTfLiteInt8) {
        prob = ((int)output->data.int8[fire_idx] - output->params.zero_point) * output->params.scale;
    }

    return prob;
}