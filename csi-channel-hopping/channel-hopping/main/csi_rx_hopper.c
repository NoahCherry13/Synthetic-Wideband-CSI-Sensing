/* Get Start Example

   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/

/*******************************************************************************
 * File: csi_rx_hopper.c
 * Author: Noah Cherry (ncherry@bu.edu)
 * Date: 2026-04-28
 * Description: Channel hopping CSI promiscuous receiver script using ESP-NOW.
 * Switches channels every 50ms. Based on ESP-CSI get-started receiver.
 * 
 * Revision History:
 * 2026-04-28  Noah Cherry  Initial draft.
 * 
 ******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "nvs_flash.h"

#include "esp_mac.h"
#include "rom/ets_sys.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_now.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define CONFIG_LESS_INTERFERENCE_CHANNEL   11

#define CONFIG_WIFI_BANDWIDTH           WIFI_BW_HT40
#define CONFIG_ESP_NOW_RATE                WIFI_PHY_RATE_MCS0_LGI
#define CONFIG_FORCE_GAIN                  1

#if CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32C3 || CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6
    #define CONFIG_GAIN_CONTROL              20
#endif

static const uint8_t CONFIG_CSI_SEND_MAC[] = {0x80, 0xb5, 0x4e, 0xf7, 0x43, 0xbc};
static const char *TAG = "csi_recv";
static const uint8_t hop_channels[] = {6, 11};

typedef struct{
    uint32_t    send_stamp;
    uint8_t     channel;
    char        msg[14];
} __attribute__((packed)) payload_t;

typedef struct
{
    unsigned : 32; /**< reserved */
    unsigned : 32; /**< reserved */
    unsigned : 32; /**< reserved */
    unsigned : 32; /**< reserved */
    unsigned : 32; /**< reserved */
#if CONFIG_IDF_TARGET_ESP32S2
    unsigned : 32; /**< reserved */
#elif CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32C3 || CONFIG_IDF_TARGET_ESP32C5 ||CONFIG_IDF_TARGET_ESP32C6
    unsigned : 16; /**< reserved */
    unsigned fft_gain : 8;
    unsigned agc_gain : 8;
    unsigned : 32; /**< reserved */
#endif
    unsigned : 32; /**< reserved */
#if CONFIG_IDF_TARGET_ESP32S2
      signed : 8;  /**< reserved */
    unsigned : 24; /**< reserved */
#elif CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32C3 || CONFIG_IDF_TARGET_ESP32C5
    unsigned : 32; /**< reserved */
    unsigned : 32; /**< reserved */
    unsigned : 32; /**< reserved */
#endif
    unsigned : 32; /**< reserved */
} wifi_pkt_rx_ctrl_phy_t;

#if CONFIG_FORCE_GAIN
    extern void phy_fft_scale_force(bool force_en, uint8_t force_value);
    extern void phy_force_rx_gain(int force_en, int force_value);
#endif

static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf) {
        ESP_LOGW(TAG, "<%s> wifi_csi_cb", esp_err_to_name(ESP_ERR_INVALID_ARG));
        return;
    }

    if (memcmp(info->mac, CONFIG_CSI_SEND_MAC, 6)) {
        return;
    }
    wifi_pkt_rx_ctrl_phy_t *phy_info = (wifi_pkt_rx_ctrl_phy_t *)info;
    static int s_count = 0;

#if CONFIG_GAIN_CONTROL
    static uint16_t agc_gain_sum=0; 
    static uint16_t fft_gain_sum=0; 
    static uint8_t agc_gain_force_value=0; 
    static uint8_t fft_gain_force_value=0; 
    if (s_count<100) {
        agc_gain_sum += phy_info->agc_gain;
        fft_gain_sum += phy_info->fft_gain;
    } else if (s_count == 100) {
        agc_gain_force_value = agc_gain_sum/100;
        fft_gain_force_value = fft_gain_sum/100;
    #if CONFIG_FORCE_GAIN
        phy_fft_scale_force(1,fft_gain_force_value);
        phy_force_rx_gain(1,agc_gain_force_value);
    #endif
        ESP_LOGI(TAG,"fft_force %d, agc_force %d",fft_gain_force_value,agc_gain_force_value);
    }
#endif
    uint32_t rx_id = *(uint32_t *)(info->payload + 15);
    const wifi_pkt_rx_ctrl_t *rx_ctrl = &info->rx_ctrl;
#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6
    if (!s_count) {
        ESP_LOGI(TAG, "================ CSI RECV ================");
        ets_printf("type,seq,mac,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_state,len,first_word,data\n");
    }
    // rssi removed from here
    ets_printf("CSI_DATA,%d," MACSTR ",%d,%d,%d,%d,%d,%d,%d,%d",
            rx_id, MAC2STR(info->mac), rx_ctrl->rate,
            rx_ctrl->noise_floor, phy_info->fft_gain, phy_info->agc_gain,rx_ctrl->channel,
            rx_ctrl->timestamp, rx_ctrl->sig_len, rx_ctrl->rx_state);
#else
    if (!s_count) {
        ESP_LOGI(TAG, "================ CSI RECV ================");
        ets_printf("type,id,mac,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,len,first_word,data\n");
    }
    // rssi removed from here

    payload_t *payload = (payload_t *)(info->payload);
    ets_printf("CSI_DATA,%d,%u," MACSTR ",%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
            rx_ctrl->channel, payload->send_stamp, MAC2STR(info->mac), rx_ctrl->rssi, rx_ctrl->rate, rx_ctrl->sig_mode,
            rx_ctrl->mcs, rx_ctrl->cwb, rx_ctrl->smoothing, rx_ctrl->not_sounding,
            rx_ctrl->aggregation, rx_ctrl->stbc, rx_ctrl->fec_coding, rx_ctrl->sgi,
            rx_ctrl->noise_floor, rx_ctrl->ampdu_cnt, rx_ctrl->secondary_channel,
            rx_ctrl->timestamp, rx_ctrl->ant, rx_ctrl->sig_len, rx_ctrl->rx_state);
    
#endif
#if CONFIG_IDF_TARGET_ESP32C5 && CSI_FORCE_LLTF
    ets_printf(",%d,%d,\"[%d", (info->len-2)/2, info->first_word_invalid, (int16_t)(((int16_t)info->buf[1]) << 12)>>4 | (uint8_t)info->buf[0]);
    for (int i = 2; i < (info->len-2); i+=2) {
        ets_printf(",%d", (int16_t)(((int16_t)info->buf[i+1]) << 12)>>4 | (uint8_t)info->buf[i]);
    }
#else
    ets_printf(",%d,%d,\"[%d", info->len, info->first_word_invalid, info->buf[0]);
    for (int i = 1; i < info->len; i++) {
        ets_printf(",%d", info->buf[i]);
    }
#endif
    ets_printf("]\"\n");
    s_count++; 
}

static void wifi_init()
{
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(esp_netif_init());
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));

    // Set the channel
    if (CONFIG_WIFI_BANDWIDTH == WIFI_BW_HT20)
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    else
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    
    uint8_t mac_addr[6];
    esp_wifi_get_mac(WIFI_IF_STA, mac_addr);
    ESP_LOGI(TAG, "Receiver MAC Address: %02x:%02x:%02x:%02x:%02x:%02x",
             mac_addr[0], mac_addr[1], mac_addr[2], mac_addr[3], mac_addr[4], mac_addr[5]);
}

static void wifi_csi_init()
{

#if CONFIG_IDF_TARGET_ESP32C5
#elif CONFIG_IDF_TARGET_ESP32C6
#else
    wifi_csi_config_t csi_config = {
        .lltf_en           = true,
        .htltf_en          = true,
        .stbc_htltf2_en    = true,
        .ltf_merge_en      = true,
        .channel_filter_en = true,
        .manu_scale        = false,
        .shift             = false,
    };
#endif
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

static void espnow_recv_cb(const esp_now_recv_info_t *recv_info, const uint8_t *data, int len) {
    if (len > 0) {
        ESP_LOGI(TAG, "ESP-NOW packet received, length: %d. Communication is OK.", len);
    }
}

static void rx_hopper_task(void *pvParameters) {
    int chan_idx = 0;
    int num_channels = sizeof(hop_channels) / sizeof(hop_channels[0]);

    while (1) {
        ESP_LOGI(TAG, "--- INITIATING NUCLEAR HOP TO CHANNEL %d ---", hop_channels[chan_idx]);

        ESP_LOGI(TAG, "CHANNEL: %d", hop_channels[0]);
        esp_wifi_set_channel(hop_channels[0], WIFI_SECOND_CHAN_BELOW);
        vTaskDelay(pdMS_TO_TICKS(25)); 
        ESP_LOGI(TAG, "CHANNEL: %d", hop_channels[1]);
        esp_wifi_set_channel(hop_channels[1], WIFI_SECOND_CHAN_BELOW);
        vTaskDelay(pdMS_TO_TICKS(25)); 

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
    wifi_csi_init();

    ESP_LOGI(TAG, "Starting Receiver Channel Hopper...");
    xTaskCreate(rx_hopper_task, "rx_hopper", 8192, NULL, configMAX_PRIORITIES - 1, NULL);
}