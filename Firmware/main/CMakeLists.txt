idf_component_register(SRCS "server.cpp" "classifier.cpp" "main.cpp" 
                    INCLUDE_DIRS ""
                    REQUIRES esp_wifi nvs_flash esp_http_server esp32-camera esp_psram esp_driver_gpio esp_timer json esp_driver_sdmmc )