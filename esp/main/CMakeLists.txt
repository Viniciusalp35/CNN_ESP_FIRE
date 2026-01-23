idf_component_register(
    SRCS "main.cpp" 
         "model_data.cc"  
    
    INCLUDE_DIRS "."
    
    PRIV_REQUIRES esp-tflite-micro esp32-camera nvs_flash driver
)
