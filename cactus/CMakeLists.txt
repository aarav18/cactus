cmake_minimum_required(VERSION 3.10)
project(Cactus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message(STATUS "CMAKE_BUILD_TYPE was not set, defaulting to Release.")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -arch arm64 -march=armv8.2-a+fp16+simd+dotprod+i8mm -pthread -Wall -Wextra -pedantic -O3 -Wno-missing-field-initializers")
    add_compile_definitions(
        __ARM_NEON=1
        __ARM_FEATURE_FP16_VECTOR_ARITHMETIC=1
        __ARM_FEATURE_DOTPROD=1
        __ARM_FEATURE_MATMUL_INT8=1
    )

    enable_language(OBJCXX)
    find_package(CURL REQUIRED)
    find_library(COREML_FRAMEWORK CoreML REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+fp16+simd+dotprod+i8mm -pthread -Wall -Wextra -pedantic -O3 -Wno-missing-field-initializers")
    add_compile_definitions(
        __ARM_NEON=1
        __ARM_FEATURE_FP16_VECTOR_ARITHMETIC=1
        __ARM_FEATURE_DOTPROD=1
        __ARM_FEATURE_MATMUL_INT8=1
    )
endif()

file(GLOB ENGINE_SOURCES "engine/*.cpp")
file(GLOB GRAPH_SOURCES "graph/*.cpp")
file(GLOB KERNEL_SOURCES "kernel/*.cpp")
file(GLOB FFI_SOURCES "ffi/*.cpp")
file(GLOB MODEL_SOURCES "models/*.cpp")

set(SME2_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernel/kernel_sme2.cpp")
list(REMOVE_ITEM KERNEL_SOURCES "${SME2_SOURCE}")

if(APPLE)
    set(NPU_SOURCES "npu/npu_ane.mm")
    set_source_files_properties(npu/npu_ane.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    message(STATUS "Apple platform detected - NPU/ANE acceleration enabled")
else()
    set(NPU_SOURCES "npu/npu.cpp")
endif()

set(COMMON_SOURCES
    ${ENGINE_SOURCES}
    ${GRAPH_SOURCES}
    ${KERNEL_SOURCES}
    ${FFI_SOURCES}
    ${MODEL_SOURCES}
    ${NPU_SOURCES}
)

set(COMMON_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    engine
    graph
    kernel
    ffi
    models
    npu
)

function(configure_cactus_target target_name)
    target_include_directories(${target_name} PUBLIC ${COMMON_INCLUDES})

    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(${target_name} PRIVATE -Wno-c99-extensions)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${target_name} PRIVATE -Wno-pedantic)
    endif()

    if(APPLE)
        target_link_libraries(${target_name} PUBLIC
            ${COREML_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
            ${ACCELERATE_FRAMEWORK}
            ${CURL_LIBRARIES}
        )
        target_include_directories(${target_name} PUBLIC ${CURL_INCLUDE_DIRS})
    endif()
endfunction()

set(ENABLE_SME2 "AUTO" CACHE STRING "SME2 build mode: AUTO or OFF")
set_property(CACHE ENABLE_SME2 PROPERTY STRINGS AUTO OFF)

set(BUILD_SME2 0)
set(SME2_MARCH_FLAG "-march=armv9.2-a+nosve+nosve2+sme2")
if(ENABLE_SME2 STREQUAL "AUTO")
	include(CheckCXXCompilerFlag)

	if(APPLE AND CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
		check_cxx_compiler_flag("-arch arm64 ${SME2_MARCH_FLAG}" COMPILER_SUPPORTS_SME2)

		if(COMPILER_SUPPORTS_SME2)
			set(BUILD_SME2 1)
			message(STATUS "Toolchain supports SME2 flags - building SME2 kernels")
		else()
			message(STATUS "Compiler does not support SME2 flags - building without SME2")
		endif()

	else()
		message(STATUS "Non-arm64 target - building without SME2")
	endif()

endif()

if(BUILD_SME2)
	add_library(cactus_sme2_obj OBJECT ${SME2_SOURCE})
	target_include_directories(cactus_sme2_obj PRIVATE ${COMMON_INCLUDES})
	target_compile_options(cactus_sme2_obj PRIVATE ${SME2_MARCH_FLAG})
	set_target_properties(cactus_sme2_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
	target_compile_definitions(cactus_sme2_obj PRIVATE CACTUS_COMPILE_SME2)
endif()

if(BUILD_SME2)
	add_library(cactus STATIC ${COMMON_SOURCES} $<TARGET_OBJECTS:cactus_sme2_obj>)
else()
	add_library(cactus STATIC ${COMMON_SOURCES})
endif()
configure_cactus_target(cactus)

if(BUILD_SME2)
	target_compile_definitions(cactus PRIVATE CACTUS_COMPILE_SME2)
endif()

if(BUILD_SME2)
	add_library(cactus_ffi SHARED ${COMMON_SOURCES} $<TARGET_OBJECTS:cactus_sme2_obj>)
else()
	add_library(cactus_ffi SHARED ${COMMON_SOURCES})
endif()
configure_cactus_target(cactus_ffi)
set_target_properties(cactus_ffi PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "cactus"
)

if(BUILD_SME2)
	target_compile_definitions(cactus_ffi PRIVATE CACTUS_COMPILE_SME2)
endif()
