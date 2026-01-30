#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_camera.h"
#include "nvs_flash.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "classifier.h"

// --- CONFIG ---
#define SSID "NOME_REDE"
#define PASS "SENHA_REDE"

// Pinos AI-Thinker
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

void start_camera_server(); // do server.cpp
static const char *TAG = "MAIN";

static void wifi_event_handler(void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data) {
    if (event_id == WIFI_EVENT_STA_START) esp_wifi_connect();
    else if (event_id == WIFI_EVENT_STA_DISCONNECTED) esp_wifi_connect();
    else if (event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI("MAIN", "IP: " IPSTR, IP2STR(&event->ip_info.ip));
        start_camera_server();
    }
}

void init_wifi() {
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);
    esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL);
    esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL);
    wifi_config_t wifi_config = {};
    strcpy((char*)wifi_config.sta.ssid, SSID);
    strcpy((char*)wifi_config.sta.password, PASS);
    wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    esp_wifi_start();
}

static void init_camera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    
    // Configurações de Qualidade
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size = FRAMESIZE_QVGA; // 320x240
    config.jpeg_quality = 12; // Menor número = Melhor qualidade (10-63)
    config.fb_count = 2;

    // Inicializa Camera
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed with error 0x%x", err);
        return;
    }

    // Configurações do Sensor
    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        // Desliga ajustes automáticos que atrapalham a IA
        s->set_gain_ctrl(s, 0);      
        s->set_exposure_ctrl(s, 0);  
        s->set_awb_gain(s, 0);       
        
        // Configurações manuais de exposição e ganho
        s->set_aec_value(s, 300);  
        s->set_agc_gain(s, 0);     
        
        ESP_LOGI(TAG, "Configurações Manuais de Sensor Aplicadas!");
    }
} 

extern "C" void app_main() {
    nvs_flash_init();
    init_camera();
    classifier_init(12.0f);
    init_wifi();
    start_camera_server();
    while (1) vTaskDelay(1000);
}
