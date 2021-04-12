#include	<stdio.h>
#include	<stdlib.h>
#include	<stdint.h>
#include	<math.h>
#include	<stdbool.h>
#include	<string.h>
#include	"sys_FeedForwardNN.h"

#ifndef FP32_FEEDFORWARDNN
#define FP32_FEEDFORWARDNN

#define	SEL_BIAS	false
#define	SEL_WEIGHT	true

void		v_FloatSetParameters(netparam_t	input_xNetParam);
void		v_float_SetPtr(float* input_WBPtr);
void		v_DynamicAllocForwardProp(float** ZPtrPtr, float** FuncPtrPtr);
void		v_DynamicAlloc_NeuronVar(float** WBPtrPtr);
void		v_DynamicAllocBackProp(float** PropErrorPtr);
void		v_ProcessForwardPropNN_FP32(float* ZPtr, float* FuncPtr, float* Input);
void		v_ProcessBackPropNN_FP32(float* PropErrorPtr, float* ZPtr, float* FuncPtr, float* Input, float* YPtr);
void		v_OptimizeWB_FP32(float* PropErrorPtr, float* FuncPtr);
void		v_TrainNN_FP32(DataSet_t xDataSet, bool verbose);

//void		v_ProcessOutputError_FP32(netparam_t	xNetParam, float* PropErrorPtr, float* FuncPtr, float* ZPtr, float* YPtr);
//void		v_MACForwardProp_FP32(	float* res, netparam_t	xNetParam, uint8_t	layer, float* FuncPtr, float* WBPtr);
//void		v_ReLUActiv_FP32(netparam_t xNetParam, float* ZPtr, uint16_t layer, float* FuncPtr);
//uint32_t	u32_GetIndex(netparam_t	xNetParam, uint8_t	layer, uint16_t row, bool	weightbias);

#endif