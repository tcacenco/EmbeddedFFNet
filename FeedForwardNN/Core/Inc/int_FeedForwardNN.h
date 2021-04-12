#include	"sys_FeedForwardNN.h"

void v_DynamicAlloc_quant(quantizedval_t* xQuantizedVal, distribution_t* xDistribution, netparam_t	xNetParam);

void v_FreeAlloc_quant(quantizedval_t	xQuantizedVal, distribution_t xDistribution);

void v_SetQuantNetParameters(netparam_t	input_xnetparam, quantizedval_t input_xquantizedval, distribution_t input_xDistribution);

void v_DynamicAllocForwardProp_int(float** ZPtrPtr, float** FuncPtrPtr);

void v_MacForwardProp_int(void* res, uint8_t	layer, void* Funcptr, void* WBptr);

void v_QuantizeIntputs_int(void* quant_input, float* input);

void v_ProcessForwardPropNN_int(void* ZPtr, void* FuncPtr, void* input);
