cmake_minimum_required(VERSION 2.8)
project(mcmclib C)
set(LIBSRC infmcmc.c infmcmc.h finmcmc.c finmcmc.h)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${mcmclib_SOURCE_DIR}/CMakeModules")

message(STATUS "module path: ${CMAKE_MODULE_PATH}")

find_package(GSL REQUIRED)
find_package(FFTW REQUIRED)
include_directories(${GSL_INCLUDE_DIR} ${GSLCBLAS_INCLUDE_DIR} ${FFTW_INCLUDE_DIR})

set(LIBS ${LIBS} ${GSL_LIBRARIES} ${GSLCBLAS_LIBRARIES} ${FFTW_LIBRARIES})

message(STATUS "libs: ${LIBS}")
message(STATUS "headers: ${GSL_INCLUDE_DIR} ${GSLCBLAS_INCLUDE_DIR}")

add_library(mcmc SHARED ${LIBSRC})
target_link_libraries(mcmc ${LIBS})
# link_libraries(${LIBS})

install(TARGETS mcmc LIBRARY DESTINATION lib)
install(FILES infmcmc.h finmcmc.h DESTINATION include)

message(STATUS "install dir: ${CMAKE_INSTALL_PREFIX}")
