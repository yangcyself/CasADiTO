# set(CMAKE_C_COMPILER "gcc-8")
# set(CMAKE_CXX_COMPILER "/usr/bin/g++-8")
# set(CC "gcc-8")
# set(CXX "g++-8")
# better call with cmake -DCMAKE_CXX_COMPILER=g++-8 ../

project(cppGenExp)
cmake_minimum_required(VERSION 3.7)
set(CMAKE_BUILD_TYPE Release)
SET(CMAKE_CXX_FLAGS_RELEASE "-O0") # must set the release flag, or it will add -O3 by default

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
# SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
# SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

link_directories(/usr/local/lib)
find_package(Eigen3 REQUIRED HINTS ${Eigen3_HINT})

# THE FOLLOWING TWO LINES LEADS TO COREDUMP
# find_library(COINMUMPS coinmumps /usr/local/lib)
# message(STATUS ${COINMUMPS})

add_subdirectory(generated/flatJump) #nlpFltJmp
add_subdirectory(generated/terrainJump) #nlpTrnJmp

add_executable(nlp main.cpp tonlp.cpp )

# target_compile_options(nlp PRIVATE "-pg")
# target_link_options(nlp PRIVATE "-pg")


# target_link_directories(nlp PRIVATE /usr/local/lib)
target_link_libraries(nlp nlpTrnJmp)
target_link_libraries(nlp ipopt)
target_link_libraries (nlp Eigen3::Eigen)
target_include_directories(nlp PRIVATE /usr/local/include/coin-or /usr/local/include/coin-or/mumps)
