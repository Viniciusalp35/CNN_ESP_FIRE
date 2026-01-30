cmake_minimum_required(VERSION 3.16)

project(FireMonitor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- CRÍTICO: Ativa a geração automática de MOC (Meta-Object Compiler) ---
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Encontra os pacotes do Qt6 no Fedora
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets Network)

add_executable(FireMonitor main.cpp)

# Linka as bibliotecas
target_link_libraries(FireMonitor PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Network
)
