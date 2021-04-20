#include	"sys_FeedForwardNN.h"
#include	"int_FeedForwardNN.h"
#include	"fp32_FeedForwardNN.h"
#include 	"usbd_cdc_if.h"

#ifndef DATATRANSFER
#define DATATRANSFER

void v_LoadFloatModel(netparam_t* xNetParam, float** WBPtr);

//void v_SaveFloatModel(netparam_t xNetParam, float* WBPtr);

void v_LoadIntModel(netparam_t* xNetParam, distribution_t* xDistribution, quantizedval_t* xQuantizedVal);

//void v_SaveIntModel(netparam_t xNetParam, distribution_t xDistribution, quantizedval_t xQuantizedVal);

void v_LoadTestDataSet(uint8_t* DataPtr, DataSet_t* xDataSet, netparam_t	xNetParam);

//void v_SaveTestDataSet(DataSet_t xDataSet, netparam_t	xNetParam);

void v_LoadTestDataNum(DataSet_t* xDataSet, netparam_t	xNetParam, uint16_t num);
#endif
