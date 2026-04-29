set(SHARED_SRCS "csi_rx_hopper.c" "csi_tx_hopper.c" "wifi_common.c")

if("$ENV{APP_ROLE}" STREQUAL "RECEIVER")
    message(STATUS "build target: RECEIVER")
    set(ROLE_SRCS "csi_rx_hopper.c")
else()
    message(STATUS "Build target: SENDER")
    set(ROLE_SRCS "csi_tx_hopper.c")
endif()

# idf_component_register(SRCS "${ROLE_SRCS}" "${SHARED_SRCS}"
#                     INCLUDE_DIRS ".")


idf_component_register(SRCS "${ROLE_SRCS}"
                   INCLUDE_DIRS ".")
