#include "esp_http_server.h"
#include "esp_camera.h"
#include "esp_log.h"
#include "img_converters.h"
#include "classifier.h"

static const char *TAG = "SERVER";

// Variáveis de Estado
static bool g_fire_detected = false;
static float g_fire_score = 0.0f;
static int g_frame_counter = 0;

// Handler de STATUS (JSON para a UI do Qt)
esp_err_t status_handler(httpd_req_t *req) {
    char json_response[128];
    snprintf(json_response, sizeof(json_response), "{\"fire\":%s, \"score\":%.1f}", 
             g_fire_detected ? "true" : "false", 
             g_fire_score * 100.0f);
            
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    // No-Cache é vital para tempo real
    httpd_resp_set_hdr(req, "Cache-Control", "no-store, no-cache, must-revalidate, max-age=0");
    
    httpd_resp_send(req, json_response, strlen(json_response));
    return ESP_OK;
}

// Handler de CAPTURA (Imagem Original)
esp_err_t capture_handler(httpd_req_t *req) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGE(TAG, "Capture Failed");
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    // --- IA (Análise) ---
    // Roda a cada 3 frames para economizar CPU.
    // O classifier_predict faz uma cópia interna para RGB, 
    // então o buffer fb->buf (JPEG) permanece intacto/original.
    if (g_frame_counter++ % 3 == 0) {
        float score = classifier_predict(fb->buf, fb->len);
        g_fire_score = score;
        g_fire_detected = (score > 0.60f); // Threshold 60%
        
        if (g_fire_detected) {
            ESP_LOGW(TAG, "FOGO DETECTADO: %.1f%%", score * 100);
        }
    }

    // Configura Headers para JPEG
    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store, no-cache, must-revalidate, max-age=0");
    
    // Envia o buffer ORIGINAL da câmera (sem cortes, sem gamma visual)
    esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
    
    esp_camera_fb_return(fb);
    return res;
}

void start_camera_server() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;
    config.stack_size = 4096; // Stack seguro

    httpd_handle_t server = NULL;
    
    // Rotas
    httpd_uri_t capture_uri = { .uri = "/capture", .method = HTTP_GET, .handler = capture_handler, .user_ctx = NULL };
    httpd_uri_t status_uri = { .uri = "/status", .method = HTTP_GET, .handler = status_handler, .user_ctx = NULL };

    if (httpd_start(&server, &config) == ESP_OK) {
        httpd_register_uri_handler(server, &capture_uri);
        httpd_register_uri_handler(server, &status_uri);
        ESP_LOGI(TAG, "Servidor Iniciado");
    }
}