/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2021 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                        opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "usb_device.h"
#include "usbd_cdc_if.h"
#include "DataTransfer.h"
#include "fp32_FeedForwardNN.h"
#include "int_FeedForwardNN.h"
#include "sys_FeedForwardNN.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
RTC_HandleTypeDef hrtc;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_RTC_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_RTC_Init();
  MX_USB_DEVICE_Init();

  /* USER CODE BEGIN 2 */
  int i;
  int j;
  uint32_t time_old = 0;
  uint32_t process_time = 0;

  netparam_t xNetParam;

  DataSet_t xDataSet = {.x_test = NULL, .y_test = NULL};
  float* WBPtr = NULL;

  distribution_t xDistribution;
  quantizedval_t xQuantizedVal;


  float* f_ZPtr = NULL;
  float* f_FuncPtr = NULL;

  void* int_FuncPtr = NULL;
  void* int_ZPtr = NULL;

  uint16_t test_index;
  uint16_t test_size;
  uint8_t flag_model;



  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
	  if(b_Get_flagrun() == true)
	  {
		  v_Clear_flagrun();

		  test_index = ui_Get_testindex();
		  test_size = ui_Get_testsize();
		  flag_model = ui_Get_flagmodel();


		  switch(flag_model)
		  {
		  case FLAGMODEL_FLOAT:
			  v_LoadFloatModel(&xNetParam, &WBPtr);
			  v_FloatSetParameters(xNetParam);
			  v_float_SetPtr(WBPtr);
			  v_DynamicAllocForwardProp(&f_ZPtr, &f_FuncPtr);
			  time_old = HAL_GetTick();
			  for(j = 0; j<10; j++)
			  {
				  for(i = 0; i < test_size; i++)
				  {
					  v_LoadTestDataNum(&xDataSet, xNetParam, (test_index+i));
					  v_ProcessForwardPropNN_FP32(f_ZPtr, f_FuncPtr, xDataSet.x_test);
				  }
			  }
			  process_time = HAL_GetTick() - time_old;
			  break;
		  case FLAGMODEL_INT4:
			  v_DynamicAllocForwardProp_int(&int_ZPtr, &int_FuncPtr);
			  v_LoadIntModel(&xNetParam, &xDistribution, &xQuantizedVal);
			  xNetParam.xVarPrecision.precision = 4;
			  v_SetQuantNetParameters(xNetParam, xQuantizedVal, xDistribution);
			  time_old = HAL_GetTick();
			  for(j = 0; j<100; j++)
			  {
				  for(i = 0; i < test_size; i++)
				  {
					  v_LoadTestDataNum(&xDataSet, xNetParam, (test_index+i));
					  v_QuantizeIntputs_int(&((int8_t*)int_FuncPtr)[0], (float*)xDataSet.x_test);
					  v_ProcessForwardPropNN_int(int_ZPtr, int_FuncPtr, xDataSet.x_test);
				  }
			  }
			  process_time = HAL_GetTick() - time_old;
			  break;
		  case FLAGMODEL_INT8:
			  v_DynamicAllocForwardProp_int(&int_ZPtr, &int_FuncPtr);
			  v_LoadIntModel(&xNetParam, &xDistribution, &xQuantizedVal);
			  xNetParam.xVarPrecision.precision = 8;
			  v_SetQuantNetParameters(xNetParam, xQuantizedVal, xDistribution);
			  time_old = HAL_GetTick();
			  for(j = 0; j<100; j++)
			  {
				  for(i = 0; i < test_size; i++)
				  {
					  v_LoadTestDataNum(&xDataSet, xNetParam, (test_index+i));
					  v_QuantizeIntputs_int(&((int8_t*)int_FuncPtr)[0], (float*)xDataSet.x_test);
					  v_ProcessForwardPropNN_int(int_ZPtr, int_FuncPtr, xDataSet.x_test);
				  }
			  }
			  process_time = HAL_GetTick() - time_old;
			  break;
		  case FLAGMODEL_INT16:
			  v_DynamicAllocForwardProp_int(&int_ZPtr, &int_FuncPtr);
			  v_LoadIntModel(&xNetParam, &xDistribution, &xQuantizedVal);
			  xNetParam.xVarPrecision.precision = 16;
			  v_SetQuantNetParameters(xNetParam, xQuantizedVal, xDistribution);
			  time_old = HAL_GetTick();
			  for(j = 0; j<100; j++)
			  {
				  for(i = 0; i < test_size; i++)
				  {
					  v_LoadTestDataNum(&xDataSet, xNetParam, (test_index+i));
					  v_QuantizeIntputs_int(&((int16_t*)int_FuncPtr)[0], (float*)xDataSet.x_test);
					  v_ProcessForwardPropNN_int(int_ZPtr, int_FuncPtr, xDataSet.x_test);
				  }
			  }
			  process_time = HAL_GetTick() - time_old;
			  break;
		  case FLAGMODEL_INT32:
			v_DynamicAllocForwardProp_int(&int_ZPtr, &int_FuncPtr);
			v_LoadIntModel(&xNetParam, &xDistribution, &xQuantizedVal);
			xNetParam.xVarPrecision.precision = 32;
			v_SetQuantNetParameters(xNetParam, xQuantizedVal, xDistribution);
			time_old = HAL_GetTick();
			for(j = 0; j<10; j++)
			{
				  for(i = 0; i < test_size; i++)
				  {
					  v_LoadTestDataNum(&xDataSet, xNetParam, (test_index+i));
					  v_QuantizeIntputs_int(&((int32_t*)int_FuncPtr)[0], (float*)xDataSet.x_test);
					  v_ProcessForwardPropNN_int(int_ZPtr, int_FuncPtr, xDataSet.x_test);
				  }
			}
			process_time = HAL_GetTick() - time_old;
			break;
		  }


	  }
    /* USER CODE END WHILE */
    /* USER CODE BEGIN 3 */

  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_LSI|RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.LSIState = RCC_LSI_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 168;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 7;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
  PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_RTC;
  PeriphClkInitStruct.RTCClockSelection = RCC_RTCCLKSOURCE_LSI;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
  /** Enables the Clock Security System
  */
  HAL_RCC_EnableCSS();
}

/**
  * @brief RTC Initialization Function
  * @param None
  * @retval None
  */
static void MX_RTC_Init(void)
{

  /* USER CODE BEGIN RTC_Init 0 */

  /* USER CODE END RTC_Init 0 */

  /* USER CODE BEGIN RTC_Init 1 */

  /* USER CODE END RTC_Init 1 */
  /** Initialize RTC Only
  */
  hrtc.Instance = RTC;
  hrtc.Init.HourFormat = RTC_HOURFORMAT_24;
  hrtc.Init.AsynchPrediv = 127;
  hrtc.Init.SynchPrediv = 255;
  hrtc.Init.OutPut = RTC_OUTPUT_DISABLE;
  hrtc.Init.OutPutPolarity = RTC_OUTPUT_POLARITY_HIGH;
  hrtc.Init.OutPutType = RTC_OUTPUT_TYPE_OPENDRAIN;
  if (HAL_RTC_Init(&hrtc) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN RTC_Init 2 */

  /* USER CODE END RTC_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();

}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */

  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     tex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
