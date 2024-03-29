/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : usbd_cdc_if.c
  * @version        : v1.0_Cube
  * @brief          : Usb device for Virtual Com Port.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2021 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under Ultimate Liberty license
  * SLA0044, the "License"; You may not use this file except in compliance with
  * the License. You may obtain a copy of the License at:
  *                             www.st.com/SLA0044
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "usbd_cdc_if.h"

/* USER CODE BEGIN INCLUDE */

/* USER CODE END INCLUDE */

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/

/* USER CODE BEGIN PV */
/* Private variables ---------------------------------------------------------*/

/* USER CODE END PV */

/** @addtogroup STM32_USB_OTG_DEVICE_LIBRARY
  * @brief Usb device library.
  * @{
  */

/** @addtogroup USBD_CDC_IF
  * @{
  */

/** @defgroup USBD_CDC_IF_Private_TypesDefinitions USBD_CDC_IF_Private_TypesDefinitions
  * @brief Private types.
  * @{
  */

/* USER CODE BEGIN PRIVATE_TYPES */

/* USER CODE END PRIVATE_TYPES */

/**
  * @}
  */

/** @defgroup USBD_CDC_IF_Private_Defines USBD_CDC_IF_Private_Defines
  * @brief Private defines.
  * @{
  */

/* USER CODE BEGIN PRIVATE_DEFINES */
/* USER CODE END PRIVATE_DEFINES */

/**
  * @}
  */

/** @defgroup USBD_CDC_IF_Private_Macros USBD_CDC_IF_Private_Macros
  * @brief Private macros.
  * @{
  */

/* USER CODE BEGIN PRIVATE_MACRO */
bool b_Get_flagrun()
{
	return xUSBDownloadVar.flag_run;
}

void v_Clear_flagrun()
{
	xUSBDownloadVar.flag_run = false;
}

uint16_t ui_Get_testindex()
{
	return xUSBDownloadVar.test_index;
}

uint16_t ui_Get_testsize()
{
	return xUSBDownloadVar.test_size;
}

uint8_t ui_Get_flagmodel()
{
	return xUSBDownloadVar.flag_model;
}
/* USER CODE END PRIVATE_MACRO */

/**
  * @}
  */

/** @defgroup USBD_CDC_IF_Private_Variables USBD_CDC_IF_Private_Variables
  * @brief Private variables.
  * @{
  */
/* Create buffer for reception and transmission           */
/* It's up to user to redefine and/or remove those define */
/** Received data over USB are stored in this buffer      */
uint8_t UserRxBufferFS[APP_RX_DATA_SIZE];

/** Data to send over USB CDC are stored in this buffer   */
uint8_t UserTxBufferFS[APP_TX_DATA_SIZE];

/* USER CODE BEGIN PRIVATE_VARIABLES */

/* USER CODE END PRIVATE_VARIABLES */

/**
  * @}
  */

/** @defgroup USBD_CDC_IF_Exported_Variables USBD_CDC_IF_Exported_Variables
  * @brief Public variables.
  * @{
  */

extern USBD_HandleTypeDef hUsbDeviceFS;

/* USER CODE BEGIN EXPORTED_VARIABLES */

/* USER CODE END EXPORTED_VARIABLES */

/**
  * @}
  */

/** @defgroup USBD_CDC_IF_Private_FunctionPrototypes USBD_CDC_IF_Private_FunctionPrototypes
  * @brief Private functions declaration.
  * @{
  */

static int8_t CDC_Init_FS(void);
static int8_t CDC_DeInit_FS(void);
static int8_t CDC_Control_FS(uint8_t cmd, uint8_t* pbuf, uint16_t length);
static int8_t CDC_Receive_FS(uint8_t* pbuf, uint32_t *Len);
static int8_t CDC_TransmitCplt_FS(uint8_t *pbuf, uint32_t *Len, uint8_t epnum);

/* USER CODE BEGIN PRIVATE_FUNCTIONS_DECLARATION */

/* USER CODE END PRIVATE_FUNCTIONS_DECLARATION */

/**
  * @}
  */

USBD_CDC_ItfTypeDef USBD_Interface_fops_FS =
{
  CDC_Init_FS,
  CDC_DeInit_FS,
  CDC_Control_FS,
  CDC_Receive_FS,
  CDC_TransmitCplt_FS
};

/* Private functions ---------------------------------------------------------*/
/**
  * @brief  Initializes the CDC media low layer over the FS USB IP
  * @retval USBD_OK if all operations are OK else USBD_FAIL
  */
static int8_t CDC_Init_FS(void)
{
  /* USER CODE BEGIN 3 */
  /* Set Application Buffers */
  USBD_CDC_SetTxBuffer(&hUsbDeviceFS, UserTxBufferFS, 0);
  USBD_CDC_SetRxBuffer(&hUsbDeviceFS, UserRxBufferFS);
  return (USBD_OK);
  /* USER CODE END 3 */
}

/**
  * @brief  DeInitializes the CDC media low layer
  * @retval USBD_OK if all operations are OK else USBD_FAIL
  */
static int8_t CDC_DeInit_FS(void)
{
  /* USER CODE BEGIN 4 */
  return (USBD_OK);
  /* USER CODE END 4 */
}

/**
  * @brief  Manage the CDC class requests
  * @param  cmd: Command code
  * @param  pbuf: Buffer containing command data (request parameters)
  * @param  length: Number of data to be sent (in bytes)
  * @retval Result of the operation: USBD_OK if all operations are OK else USBD_FAIL
  */
static int8_t CDC_Control_FS(uint8_t cmd, uint8_t* pbuf, uint16_t length)
{
  /* USER CODE BEGIN 5 */
  switch(cmd)
  {
    case CDC_SEND_ENCAPSULATED_COMMAND:

    break;

    case CDC_GET_ENCAPSULATED_RESPONSE:

    break;

    case CDC_SET_COMM_FEATURE:

    break;

    case CDC_GET_COMM_FEATURE:

    break;

    case CDC_CLEAR_COMM_FEATURE:

    break;

  /*******************************************************************************/
  /* Line Coding Structure                                                       */
  /*-----------------------------------------------------------------------------*/
  /* Offset | Field       | Size | Value  | Description                          */
  /* 0      | dwDTERate   |   4  | Number |Data terminal rate, in bits per second*/
  /* 4      | bCharFormat |   1  | Number | Stop bits                            */
  /*                                        0 - 1 Stop bit                       */
  /*                                        1 - 1.5 Stop bits                    */
  /*                                        2 - 2 Stop bits                      */
  /* 5      | bParityType |  1   | Number | Parity                               */
  /*                                        0 - None                             */
  /*                                        1 - Odd                              */
  /*                                        2 - Even                             */
  /*                                        3 - Mark                             */
  /*                                        4 - Space                            */
  /* 6      | bDataBits  |   1   | Number Data bits (5, 6, 7, 8 or 16).          */
  /*******************************************************************************/
    case CDC_SET_LINE_CODING:

    break;

    case CDC_GET_LINE_CODING:

    break;

    case CDC_SET_CONTROL_LINE_STATE:

    break;

    case CDC_SEND_BREAK:

    break;

  default:
    break;
  }

  return (USBD_OK);
  /* USER CODE END 5 */
}

/**
  * @brief  Data received over USB OUT endpoint are sent over CDC interface
  *         through this function.
  *
  *         @note
  *         This function will issue a NAK packet on any OUT packet received on
  *         USB endpoint until exiting this function. If you exit this function
  *         before transfer is complete on CDC interface (ie. using DMA controller)
  *         it will result in receiving more data while previous ones are still
  *         not sent.
  *
  * @param  Buf: Buffer of data to be received
  * @param  Len: Number of data received (in bytes)
  * @retval Result of the operation: USBD_OK if all operations are OK else USBD_FAIL
  */
static int8_t CDC_Receive_FS(uint8_t* Buf, uint32_t *Len)
{
  /* USER CODE BEGIN 6 */
	uint8_t	Tx_Buffer[64];
	uint8_t	Aux_Buffer[16];
	uint8_t dig_count;
	uint16_t num;
	static	uint16_t	PacketCount = 0;
	static uint32_t		Flash_BaseAddr;
	int i = 0;
	static uint32_t		flash_cont = 0;

	USBD_CDC_SetRxBuffer(&hUsbDeviceFS, &Buf[0]);
	USBD_CDC_ReceivePacket(&hUsbDeviceFS);

	if(xUSBDownloadVar.RxMode == true)
	{
		USBD_CDC_SetRxBuffer(&hUsbDeviceFS, &Buf[0]);
		USBD_CDC_ReceivePacket(&hUsbDeviceFS);
		for(i = 0; i<(*Len); i++)
		{
			HAL_FLASH_Unlock();
			HAL_FLASH_Program(FLASH_TYPEPROGRAM_BYTE, (flash_cont + i), Buf[i]);
			HAL_FLASH_Lock();
			Buf[i]=0;
		}
		flash_cont += *Len;

		PacketCount++;
		//if(PacketCount >= ((xUSBDownloadVar.Size+(PACKET_SIZE-1))/PACKET_SIZE))
		if((flash_cont - Flash_BaseAddr) >= xUSBDownloadVar.Size)
		{
			xUSBDownloadVar.RxMode	=	false;
			strncpy((char*)Tx_Buffer, "File Received.\0", 15);
			CDC_Transmit_FS((uint8_t*)Tx_Buffer, strlen((char*)Tx_Buffer));
			switch(xUSBDownloadVar.DataType)
			{
			case DATATYPE_DATASET:
				break;
			case DATATYPE_FLOATMODEL:
				break;
			case DATATYPE_INTMODEL:
				break;
			default:
				break;
			}
		}
	}
	else
	{
		switch(Buf[0])
		{
			case	USB_MODE_TRANSFER_SIZE:
				USBD_CDC_SetRxBuffer(&hUsbDeviceFS, &Buf[0]);
				USBD_CDC_ReceivePacket(&hUsbDeviceFS);

				free(xUSBDownloadVar.DataBuffer);
				xUSBDownloadVar.Size = atoi((char*)&Buf[POS_TRANSFER_SIZE]);

				strncpy((char*)Tx_Buffer, "Size Received for Data Transfer: ", 36);
				itoa(xUSBDownloadVar.Size, (char*)Aux_Buffer,10);
				strcat((char*)Tx_Buffer, (char*)Aux_Buffer);
				strcat((char*)Tx_Buffer, "\0");

				CDC_Transmit_FS((uint8_t*)Tx_Buffer, strlen((char*)Tx_Buffer));
			break;

			case	USB_MODE_TRANSFER_DATATYPE:
				if(strncmp((char*)&Buf[POS_TRANSFER_DATATYPE], "dataset", 7) == 0)
				{
					xUSBDownloadVar.DataType = DATATYPE_DATASET;
					strncpy((char*)Tx_Buffer, "Data Type: dataset\0", 20);
				}
				else if(strncmp((char*)&Buf[POS_TRANSFER_DATATYPE], "float model", 11) == 0)
				{
					xUSBDownloadVar.DataType = DATATYPE_FLOATMODEL;
					strncpy((char*)Tx_Buffer, "Data Type: float model\0", 33);
				}
				else if(strncmp((char*)&Buf[POS_TRANSFER_DATATYPE], "int model", 9) == 0)
				{
					xUSBDownloadVar.DataType = DATATYPE_INTMODEL;
					strncpy((char*)Tx_Buffer, "Data Type: int model\0", 33);
				}
				else
				{
					xUSBDownloadVar.DataType = 0xff;
					strncpy((char*)Tx_Buffer, "Data Type: Error\0", 33);
				}
				USBD_CDC_SetRxBuffer(&hUsbDeviceFS, &Buf[0]);
				USBD_CDC_ReceivePacket(&hUsbDeviceFS);

				CDC_Transmit_FS((uint8_t*)Tx_Buffer, strlen((char*)Tx_Buffer));
			break;

			case	USB_MODE_INITTRANSFER:;
				/* Flash Erase Variables */
				uint32_t SectorError;
				FLASH_EraseInitTypeDef xFlash_Erase;
				xFlash_Erase.TypeErase = FLASH_TYPEERASE_SECTORS;
				xFlash_Erase.NbSectors = 1;
				xFlash_Erase.VoltageRange = FLASH_VOLTAGE_RANGE_3;

				/* Flash Sector Erase */
				HAL_FLASH_Unlock();
				switch(xUSBDownloadVar.DataType)
				{
				case DATATYPE_DATASET	:
					xFlash_Erase.Sector = FLASH_SECTOR_6;
					HAL_FLASHEx_Erase(&xFlash_Erase, &SectorError);
					xFlash_Erase.Sector = FLASH_SECTOR_7;
					HAL_FLASHEx_Erase(&xFlash_Erase, &SectorError);
					xFlash_Erase.Sector = FLASH_SECTOR_8;
					HAL_FLASHEx_Erase(&xFlash_Erase, &SectorError);
					xFlash_Erase.Sector = FLASH_SECTOR_9;
					HAL_FLASHEx_Erase(&xFlash_Erase, &SectorError);
					Flash_BaseAddr = BASEADDR_DATASET;
					break;
				case DATATYPE_FLOATMODEL:
					xFlash_Erase.Sector = FLASH_SECTOR_10;
					HAL_FLASHEx_Erase(&xFlash_Erase, &SectorError);
					Flash_BaseAddr = BASEADDR_FLOATMODEL;
					break;
				case DATATYPE_INTMODEL:
					xFlash_Erase.Sector = FLASH_SECTOR_11;
					HAL_FLASHEx_Erase(&xFlash_Erase, &SectorError);
					Flash_BaseAddr = BASEADDR_INTMODEL;
					break;
				}
				HAL_FLASH_Lock();

				USBD_CDC_SetRxBuffer(&hUsbDeviceFS, &Buf[0]);
				USBD_CDC_ReceivePacket(&hUsbDeviceFS);
				PacketCount					=	0;
				xUSBDownloadVar.RxMode		=	true;
				strncpy((char*)Tx_Buffer, "FLASH Sections Erased. \0", 24);
				CDC_Transmit_FS((uint8_t*)Tx_Buffer, strlen((char*)Tx_Buffer));
				flash_cont = Flash_BaseAddr;
			break;

			case USB_MODE_SELMODEL:
				if(strncmp((char*)&Buf[POS_TRANSFER_DATATYPE], "float", 5) == 0)
					xUSBDownloadVar.flag_model = FLAGMODEL_FLOAT;
				else if(strncmp((char*)&Buf[POS_TRANSFER_DATATYPE], "int 4", 5) == 0)
					xUSBDownloadVar.flag_model = FLAGMODEL_INT4;
				else if(strncmp((char*)&Buf[POS_TRANSFER_DATATYPE], "int 8", 5) == 0)
					xUSBDownloadVar.flag_model = FLAGMODEL_INT8;
				else if(strncmp((char*)&Buf[POS_TRANSFER_DATATYPE], "int 16", 6) == 0)
					xUSBDownloadVar.flag_model = FLAGMODEL_INT16;
				else if(strncmp((char*)&Buf[POS_TRANSFER_DATATYPE], "int 32", 6) == 0)
					xUSBDownloadVar.flag_model = FLAGMODEL_INT32;
				break;

			case USB_MODE_RUN:
				xUSBDownloadVar.flag_run = true;
				xUSBDownloadVar.test_index = atoi((char*)&Buf[POS_TRANSFER_SIZE]);
				num = xUSBDownloadVar.test_index;
				dig_count = 0;
				do
				{
					dig_count++;
					num /= 10;
				} while(num != 0);

				xUSBDownloadVar.test_size = atoi((char*)&Buf[POS_TRANSFER_SIZE + dig_count + 1]);
				break;

			default:
				break;
		}
	}
	return (USBD_OK);
  /* USER CODE END 6 */
}

/**
  * @brief  CDC_Transmit_FS
  *         Data to send over USB IN endpoint are sent over CDC interface
  *         through this function.
  *         @note
  *
  *
  * @param  Buf: Buffer of data to be sent
  * @param  Len: Number of data to be sent (in bytes)
  * @retval USBD_OK if all operations are OK else USBD_FAIL or USBD_BUSY
  */
uint8_t CDC_Transmit_FS(uint8_t* Buf, uint16_t Len)
{
  uint8_t result = USBD_OK;
  /* USER CODE BEGIN 7 */
  USBD_CDC_HandleTypeDef *hcdc = (USBD_CDC_HandleTypeDef*)hUsbDeviceFS.pClassData;
  if (hcdc->TxState != 0){
    return USBD_BUSY;
  }
  USBD_CDC_SetTxBuffer(&hUsbDeviceFS, Buf, Len);
  result = USBD_CDC_TransmitPacket(&hUsbDeviceFS);
  /* USER CODE END 7 */
  return result;
}

/**
  * @brief  CDC_TransmitCplt_FS
  *         Data transmited callback
  *
  *         @note
  *         This function is IN transfer complete callback used to inform user that
  *         the submitted Data is successfully sent over USB.
  *
  * @param  Buf: Buffer of data to be received
  * @param  Len: Number of data received (in bytes)
  * @retval Result of the operation: USBD_OK if all operations are OK else USBD_FAIL
  */
static int8_t CDC_TransmitCplt_FS(uint8_t *Buf, uint32_t *Len, uint8_t epnum)
{
  uint8_t result = USBD_OK;
  /* USER CODE BEGIN 13 */
  UNUSED(Buf);
  UNUSED(Len);
  UNUSED(epnum);
  /* USER CODE END 13 */
  return result;
}

/* USER CODE BEGIN PRIVATE_FUNCTIONS_IMPLEMENTATION */

/* USER CODE END PRIVATE_FUNCTIONS_IMPLEMENTATION */

/**
  * @}
  */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
