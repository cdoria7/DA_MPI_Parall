COMPILER = mpicc

SOURCE_FILE = dragonfly_PA
SOURCE_PATH = ./src

PROCESSORS = 2

LIB_FILE = dragonflylib
LIB_PATH = ./lib

BUILD_PATH = ./build
INCLUDE_PATH = ./include
RESOURCE_PATH = ./resource

ITER = 500
DRAGONFLY = 30
DIM = 3
TEST_FUNC = 5
REPEAT = 1


all: lib compile

r: $(SOURCE_FILE).o
	mpirun -np $(PROCESSORS) ./$(SOURCE_FILE).o ${ITER} ${DRAGONFLY} ${DIM} ${TEST_FUNC} ${REPEAT}

rs: $(SOURCE_FILE).o
	mpirun -np 1 ./$(SOURCE_FILE).o ${ITER} ${DRAGONFLY} ${DIM} ${TEST_FUNC} ${REPEAT}


compile: $(LIB_PATH)/$(LIB_FILE).a
	${COMPILER} \
		-o $(SOURCE_FILE).o \
		-I $(INCLUDE_PATH) \
		$(SOURCE_PATH)/$(SOURCE_FILE).c \
		$(LIB_PATH)/$(LIB_FILE).a
	
	rm -rf $(SOURCE_FILE).o.dSYM

lib: $(SOURCE_PATH)/$(LIB_FILE).c 
	clang -c -Wall -o $(BUILD_PATH)/$(LIB_FILE).o \
	 -I $(INCLUDE_PATH) $(SOURCE_PATH)/$(LIB_FILE).c 
	ar -r $(LIB_PATH)/$(LIB_FILE).a $(BUILD_PATH)/$(LIB_FILE).o

clear: $(SOURCE_FILE).o 
	rm $(SOURCE_FILE).o 

clearall: clear clearlib

clearlib: $(BUILD_PATH)/$(LIB_FILE).o $(LIB_PATH)/$(LIB_FILE).a
	rm $(BUILD_PATH)/$(LIB_FILE).o $(LIB_PATH)/$(LIB_FILE).a;

setupfolder:
	mkdir ${SOURCE_PATH} ${LIB_PATH} ${INCLUDE_PATH} ${BUILD_PATH} ${RESOURCE_PATH}
	touch ${SOURCE_PATH}/main.c; chmod 500 ${SOURCE_PATH}/main.c; 
	cd ${SOURCE_PATH}

bench:
	make; clear;
	make rs
	echo
	make r
