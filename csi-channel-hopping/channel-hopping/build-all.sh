#!/bin/bash

SENDER_BDIR=build_receiver
TRANSMITTER_BDIR=build_transmitter
export IDF_PATH=/home/ncherry/esp/esp-idf
. /home/ncherry/esp/esp-idf/export.sh

if [ $? -ne 0 ]; then
    echo "Failed to load ESP-IDF environment. Check your export.sh path."
    exit 1
fi

BUILD_SET="$1"

if [ "$BUILD_SET" = "Tx" ]; then 
    echo "Building Transmitter..." 
    export APP_ROLE=TRANSMITTER
    idf.py -B $TRANSMITTER_BDIR build

elif [ "$BUILD_SET" = "Rx" ]; then
    echo "Building Receiver..." 
    export APP_ROLE=RECEIVER
    idf.py -B $SENDER_BDIR build

elif [ "$BUILD_SET" = "TxRx" ]; then
    echo "Building Transmitter and Receiver..." 
    export APP_ROLE=TRANSMITTER
    idf.py -B $TRANSMITTER_BDIR build
    export APP_ROLE=RECEIVER
    idf.py -B $SENDER_BDIR build
elif [ "$BUILD_SET" = "clean" ]; then
    echo "Wiping build directories..." 
    idf.py -B $TRANSMITTER_BDIR fullclean
    idf.py -B $SENDER_BDIR fullclean
    echo "Clean complete!"
else
    echo "Invalid Build Set. Exiting..." 
    exit 1
fi;