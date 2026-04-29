/* Get Start Example

   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/

/*******************************************************************************
 * File: csi_tx_hopper.c
 * Author: Noah Cherry (ncherry@bu.edu)
 * Date: 2026-04-28
 * Description: Channel hopping CSI broadcasting script using ESP-NOW.
 * Transmits on 2 channels in the 2.4GHz band with a period of 5ms. Switches
 * Channels every 50ms. Based on ESP-CSI get-started transmitter.
 * 
 * Revision History:
 * 2026-04-28  Noah Cherry  Initial draft.
 * 
 ******************************************************************************/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include "nvs_flash.h"

#include "esp_mac.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "esp_timer.h"

#define CONFIG_LESS_INTERFERENCE_CHANNEL   11

#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6
    #define CONFIG_WIFI_BAND_MODE   WIFI_BAND_MODE_2G_ONLY
    #define CONFIG_WIFI_2G_BANDWIDTHS           WIFI_BW_HT20
    #define CONFIG_WIFI_5G_BANDWIDTHS           WIFI_BW_HT20
    #define CONFIG_WIFI_2G_PROTOCOL             WIFI_PROTOCOL_11N
    #define CONFIG_WIFI_5G_PROTOCOL             WIFI_PROTOCOL_11N
    #define CONFIG_ESP_NOW_PHYMODE           WIFI_PHY_MODE_HT20
#else
    #define CONFIG_WIFI_BANDWIDTH           WIFI_BW_HT40
#endif

#define CONFIG_ESP_NOW_RATE             WIFI_PHY_RATE_MCS0_LGI
#define CONFIG_SEND_FREQUENCY             1

static const uint8_t CONFIG_CSI_SEND_MAC[] = {0x80, 0xb5, 0x4e, 0xf7, 0x45, 0x30};
static const char *TAG = "csi_send";
static const uint8_t hop_channels[] = {6, 11};

typedef struct{
    uint32_t    send_stamp;
    uint8_t     channel;
    char        msg[14];
} __attribute__((packed)) payload_t;

static void wifi_init()
{
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    ESP_ERROR_CHECK(esp_netif_init());
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));

#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6
    ESP_ERROR_CHECK(esp_wifi_start());
    esp_wifi_set_band_mode(CONFIG_WIFI_BAND_MODE);
    wifi_protocols_t protocols = {
        .ghz_2g = CONFIG_WIFI_2G_PROTOCOL,
        .ghz_5g = CONFIG_WIFI_5G_PROTOCOL
    };
    ESP_ERROR_CHECK(esp_wifi_set_protocols(ESP_IF_WIFI_STA, &protocols));
    wifi_bandwidths_t bandwidth = {
        .ghz_2g = CONFIG_WIFI_2G_BANDWIDTHS,
        .ghz_5g = CONFIG_WIFI_5G_BANDWIDTHS
    };
    ESP_ERROR_CHECK(esp_wifi_set_bandwidths(ESP_IF_WIFI_STA, &bandwidth));
#else  
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(ESP_IF_WIFI_STA, CONFIG_WIFI_BANDWIDTH));
    ESP_ERROR_CHECK(esp_wifi_config_espnow_rate(ESP_IF_WIFI_STA, WIFI_PHY_RATE_MCS0_LGI));
    ESP_ERROR_CHECK(esp_wifi_start());

#endif

#if CONFIG_IDF_TARGET_ESP32 || CONFIG_IDF_TARGET_ESP32C3 || CONFIG_IDF_TARGET_ESP32S3 
    ESP_ERROR_CHECK(esp_wifi_config_espnow_rate(ESP_IF_WIFI_STA, CONFIG_ESP_NOW_RATE));
#endif

    //ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6
    if ((CONFIG_WIFI_BAND_MODE == WIFI_BAND_MODE_2G_ONLY && CONFIG_WIFI_2G_BANDWIDTHS == WIFI_BW_HT20) 
    || (CONFIG_WIFI_BAND_MODE == WIFI_BAND_MODE_5G_ONLY && CONFIG_WIFI_5G_BANDWIDTHS == WIFI_BW_HT20)) {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    }  
    else {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    }
#else
    if (CONFIG_WIFI_BANDWIDTH == WIFI_BW_HT20) {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    }
    else {        
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    }
#endif
    // ESP_ERROR_CHECK(esp_wifi_set_mac(WIFI_IF_STA, CONFIG_CSI_SEND_MAC));
    uint8_t mac_addr[6];
    esp_wifi_get_mac(WIFI_IF_STA, mac_addr);
    ESP_LOGI(TAG, "Transmitter MAC Address: %02x:%02x:%02x:%02x:%02x:%02x",
        mac_addr[0], mac_addr[1], mac_addr[2], mac_addr[3], mac_addr[4], mac_addr[5]);

}

void espnow_send_cb(const uint8_t *mac_addr, esp_now_send_status_t status) {
    if (status == ESP_NOW_SEND_SUCCESS) {
        // Log sparingly on success to avoid clutter
        // ESP_LOGI(TAG, "Packet sent successfully");
    } else {
        ESP_LOGW(TAG, "Packet send failed");
    }
}

static void wifi_esp_now_init(esp_now_peer_info_t peer) 
{
    ESP_ERROR_CHECK(esp_now_init());
    //ESP_ERROR_CHECK(esp_now_set_pmk((uint8_t *)"pmk1234567890123"));
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));
#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6
    esp_now_rate_config_t rate_config = {
        .phymode = CONFIG_ESP_NOW_PHYMODE, 
        .rate = CONFIG_ESP_NOW_RATE,    
        .ersu = false,                    
        .dcm = false                      
    };
    ESP_ERROR_CHECK(esp_now_set_peer_rate_config(peer.peer_addr,&rate_config));
#endif
}

static void tx_hopper_task(void *pvParameters){
    int chan_idx = 0;
    int num_channels = sizeof(hop_channels) / sizeof(hop_channels[0]);

    while(1){
        ESP_LOGI(TAG, "Hopping...");
        ESP_ERROR_CHECK(esp_wifi_set_channel(hop_channels[0], WIFI_SECOND_CHAN_BELOW));
        vTaskDelay(pdMS_TO_TICKS(10)); 
        ESP_LOGI(TAG, "Hopping...");
        ESP_ERROR_CHECK(esp_wifi_set_channel(hop_channels[1], WIFI_SECOND_CHAN_BELOW));
        vTaskDelay(pdMS_TO_TICKS(10)); 
    }
}

void app_main()
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
      ESP_ERROR_CHECK(nvs_flash_erase());
      ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    wifi_init();

    esp_now_peer_info_t peer = {
        .channel   = 0,
        .ifidx     = WIFI_IF_STA,    
        .encrypt   = false,   
        .peer_addr = {0x60, 0x55, 0xf9, 0xdf, 0xfa, 0xde},
    };
    wifi_esp_now_init(peer);

    ESP_LOGI(TAG, "================ CSI SEND ================");
    ESP_LOGI(TAG, "wifi_channel: %d, send_frequency: %d, mac: " MACSTR,
             CONFIG_LESS_INTERFERENCE_CHANNEL, CONFIG_SEND_FREQUENCY, MAC2STR(CONFIG_CSI_SEND_MAC));
    
    payload_t payload;
    strncpy(payload.msg, "Hello CSI!", sizeof(payload.msg));
    int chan_idx = 0;
    int num_channels = sizeof(hop_channels) / sizeof(hop_channels[0]);

    ESP_LOGI(TAG, "Starting Fast Sweeper TX...");

    while (1) {
        ESP_ERROR_CHECK(esp_wifi_set_channel(hop_channels[chan_idx], WIFI_SECOND_CHAN_BELOW));
        
        vTaskDelay(pdMS_TO_TICKS(5)); 
        
        for (int i = 0; i < 10; i++) {
            payload.channel = (uint8_t)hop_channels[chan_idx];
            ESP_LOGI(TAG, "Payload Channel: %d\n", payload.channel);
            payload.send_stamp = (uint32_t)esp_timer_get_time() / 1000;
            esp_err_t ret = esp_now_send(peer.peer_addr, (uint8_t*)&payload, sizeof(payload_t));
            if(ret != ESP_OK && ret != ESP_ERR_ESPNOW_NO_MEM) {
                ESP_LOGW(TAG, "TX Error: %s", esp_err_to_name(ret));
            }
            vTaskDelay(pdMS_TO_TICKS(100)); 
        }
        chan_idx = (chan_idx + 1) % num_channels;
    }
}