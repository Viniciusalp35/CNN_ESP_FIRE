cmake_minimum_required(VERSION 3.16)
project(fire_simulation_pc)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

set(TFLM_ROOT "../components/esp-tflite-micro")
set(TF_SRC "${TFLM_ROOT}/tensorflow/lite/micro")

include_directories(
    ../main                              
    ${TFLM_ROOT}                         
    ${TFLM_ROOT}/tensorflow              
    ${TFLM_ROOT}/third_party/flatbuffers/include 
    ${TFLM_ROOT}/third_party/gemmlowp   
)

file(GLOB USER_SOURCES 
    "../main/main.cpp"
    "../main/model_data.cc"
)

add_executable(fire_sim ${USER_SOURCES})

target_link_libraries(fire_sim ${OpenCV_LIBS})

target_compile_definitions(fire_sim PRIVATE IS_DESKTOP_SIMULATION)
