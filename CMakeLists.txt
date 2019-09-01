cmake_minimum_required(VERSION 3.12)
set(TARGET_NAME hellovr_vulkan)

#add_executable(${TARGET_NAME}
#  ${SHARED_SRC_FILES}
#  hellovr_vulkan.cpp
#)

# Get platform.

set(OPENVR_DIR $ENV{HOME} "/software/openvr")

if(WIN32)
  set(PLATFORM_NAME "win")
elseif(UNIX AND NOT APPLE)
  if(CMAKE_SYSTEM_NAME MATCHES ".*Linux")
    set(PLATFORM_NAME "linux")
    add_definitions(-DLINUX -DPOSIX)
    if(PROCESSOR_ARCH MATCHES "64")
      add_definitions(-DLINUX64)
    endif()
  endif()
elseif(APPLE)
  if(CMAKE_SYSTEM_NAME MATCHES ".*Darwin.*" OR CMAKE_SYSTEM_NAME MATCHES ".*MacOS.*")
    set(PLATFORM_NAME "osx")
    add_definitions(-DOSX -DPOSIX)
    if(BUILD_UNIVERSAL)
      set(CMAKE_OSX_ARCHITECTURES "i386;x86_64")
    endif()
    if(BUILD_OSX_I386)
      set(PROCESSOR_ARCH "32")
      set(CMAKE_OSX_ARCHITECTURES "i386")
    endif()
  endif()
endif()


add_definitions("-O3 -Wno-suggest-override")

link_directories(/usr/local/cuda/lib64)
find_package(CUDA QUIET REQUIRED)
find_package(OpenCV REQUIRED)
find_package(SDL2 REQUIRED)
find_package(Vulkan REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)

include_directories(${SDL2_INCLUDE_DIRS})
include_directories(${Vulkan_INCLUDE_DIRS})
include_directories(${HDF5_CXX_INCLUDE_DIRS})

file(GLOB SHARED_SRC_FILES
  shared/*.cpp
  shared/*.h
  )


cuda_add_executable(${TARGET_NAME}
  ${SHARED_SRC_FILES}
  buffer.cc
  vrsystem.cc
  vulkansystem.cc
  windowsystem.cc
  global.cc
  script.cc
  flywheel.cc
  scene.cc
  framerenderbuffer.cc
  refactor_hellovr_vulkan.cpp
  learningsystem.cc
  recording.cc
  gzstream.cc
  simplerender.cc
  )
cuda_add_cublas_to_target(${TARGET_NAME})


target_link_libraries(${TARGET_NAME}
  ${SDL2_LIBRARIES}
  ${VULKAN_LIBRARY}
  ${OPENVR_LIBRARIES}
  ${CMAKE_DL_LIBS}
  ${EXTRA_LIBS}
  ${CUDA_curand_LIBRARY}
  ${OpenCV_LIBS}
  ${Vulkan_LIBRARIES}
  ${HDF5_CXX_LIBRARIES}
  -lopenvr_api
  -llua5.3
  -lcapnp
  -lkj
  -lmcdnn
  -lvlstm
  -lcuda
  -lcudnn
  -lz
  )

target_include_directories(${TARGET_NAME} PRIVATE /home/marijnfs/software/openvr/headers)

#target_include_directories(${TARGET_NAME} PRIVATE ${OPENVR_DIR}/headers)
target_include_directories(${TARGET_NAME} PRIVATE /home/marijnfs/dev/vlstm/inc)
target_include_directories(${TARGET_NAME} PRIVATE /home/marijnfs/dev/mcdnn/inc)
target_include_directories(${TARGET_NAME} PRIVATE /usr/local/cuda/include)



#setTargetOutputDirectory(${TARGET_NAME})
