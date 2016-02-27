################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/CoMat.cpp \
../src/GloVe.cpp \
../src/Train.cpp \
../src/VocabBuilder.cpp 

OBJS += \
./src/CoMat.o \
./src/GloVe.o \
./src/Train.o \
./src/VocabBuilder.o 

CPP_DEPS += \
./src/CoMat.d \
./src/GloVe.d \
./src/Train.d \
./src/VocabBuilder.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -std=c++0x -I/usr/local/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


