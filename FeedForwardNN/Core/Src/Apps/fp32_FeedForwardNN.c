#include "fp32_FeedForwardNN.h"

netparam_t	xNetParam;
float* WBPtr;
float* ZPtr;
float* FuncPtr;

//************************************************************************************************
//	SET STRUCTS
//************************************************************************************************
void v_FloatSetParameters(netparam_t	input_xNetParam)
{
	xNetParam = input_xNetParam;
}

void v_float_SetPtr(float* input_WBPtr)
{
	WBPtr = input_WBPtr;
}


//************************************************************************************************
//	MEMORY FUNCTIONS
//************************************************************************************************

//	ALLOCATE DYNAMIC MEMORY
//**************************************************/**
//	IN		xNetParam			(NETWORK HYPERPARAMETERS STRUCT)
//	OUT		*ptr				(pointer to void)
//**************************************************
void v_DynamicAllocForwardProp(float** ZPtrPtr, float** FuncPtrPtr)
{
	uint16_t	i;
	uint32_t	neurons = 0;
	for (i = 0; i < xNetParam.Layers; i++)
	{
		neurons += xNetParam.NonLayer[i];
	}

	*FuncPtrPtr = malloc(ceil((float)(neurons * xNetParam.xVarPrecision.precision) / 8.0));
	*ZPtrPtr = malloc(ceil((float)(neurons*xNetParam.xVarPrecision.precision) / 8.0));
}

//	ALLOCATE DYNAMIC MEMORY
//**************************************************/**
//	IN		xNetParam			(NETWORK HYPERPARAMETERS STRUCT)
//	OUT		*ptr				(pointer to void)
//**************************************************
void v_DynamicAlloc_NeuronVar(float** WBPtrPtr)
{
	uint16_t	i;
	uint32_t	values = 0;
	for (i = 0; i < (xNetParam.Layers - 1); i++)
	{
		values += ((1 + xNetParam.NonLayer[i]) * xNetParam.NonLayer[i + 1]);
	}
	*WBPtrPtr = malloc(ceil((float)(values * xNetParam.xVarPrecision.precision) / 8.0));
}

//	ALLOCATE DYNAMIC MEMORY FOR BACKPROPAGATION
//**************************************************/**
//	IN		xNetParam			(NETWORK HYPERPARAMETERS STRUCT)
//	OUT		*ptr				(pointer to void)
//**************************************************
void v_DynamicAllocBackProp(float** PropErrorPtr)
{
	uint16_t	i;
	uint32_t	neurons = 0;
	for (i = 0; i < xNetParam.Layers; i++)
	{
		neurons += xNetParam.NonLayer[i];
	}

	*PropErrorPtr	= malloc(ceil((float)(neurons * xNetParam.xVarPrecision.precision) / 8.0));
}





//************************************************************************************************
//	ALGEBRAIC FUNCTIONS (Forward Propagation)
//************************************************************************************************

	//	Calculate for a given layer the following operation:	Z = b + W*f		(equivalent to	RESULT	=	VEC + MATRIX*VEC)
	//**************************************************
	//	IN		res, xNetParam, layer, WBPtr, FuncPtr			(FLOAT POINTER, NETWORK HYPERPARAMETERS STRUCT, layer, FLOAT POINTER, FLOAT POINTER)
	//	OUT		NONE
	//	obs.:	to adapt the function, change float to int....
	//**************************************************
void v_MACForwardProp_FP32(float* res, uint16_t	layer, float* FuncPtr)
{
	float		aux;
	uint32_t	index_w, index_b, rows, cols;

	index_w = u32_GetIndex(xNetParam, layer, 0, false);
	index_b = index_w + (xNetParam.NonLayer[layer - 1] * xNetParam.NonLayer[layer]);
	rows = xNetParam.NonLayer[layer];
	cols = xNetParam.NonLayer[layer - 1];

	for (size_t i = 0; i < rows; i++)
	{
		aux = WBPtr[index_b + i];

		for (size_t j = 0; j < cols; j++)
		{
			aux += WBPtr[index_w + (i * cols) + j] * FuncPtr[j];
		}
		res[i] = aux;
	}
}

//	Calculate Activation and Store at FuncPtr
//**************************************************
//	IN		ZPtr, xNetParam, layer, FuncPtr
//	OUT		NONE
//	obs.:	to adapt the function, change float to int....
//**************************************************
void v_ReLUActiv_FP32(float* ZPtr, uint16_t layer, float* FuncPtr)
{
	for (size_t i = 0; i < xNetParam.NonLayer[layer]; i++)
	{
		if (ZPtr[i] < 0)
		{
			FuncPtr[i] = (xNetParam.LeakyReLU_rate * ZPtr[i]) / UINT16_MAX;
		}
		else
		{
			FuncPtr[i] = ZPtr[i];
		}
	}
}

void v_LinearActiv_FP32(float* ZPtr, uint16_t layer, float* FuncPtr)
{
	for (size_t i = 0; i < xNetParam.NonLayer[layer]; i++)
	{
		FuncPtr[i] = ZPtr[i];
	}
}

//	Process all Algebraic Operations related to the FeedForward Algorithm and return NN output value (pointer)
//**************************************************
//	IN		result, xNetParam, Input, WBPtr			(result, NETWORK HYPERPARAMETERS STRUCT,FLOAT POINTER, FLOAT POINTER)
//	OUT		NONE
//	obs.:	to adapt the function, change float to int....
//**************************************************
void v_ProcessForwardPropNN_FP32(float* ZPtr, float* FuncPtr, float* Input)
{
	uint16_t	i;
	uint32_t	neurons = 0;
	uint32_t	aux_index = 0;

	for (i = 0; i < xNetParam.Layers; i++)
	{
		neurons += xNetParam.NonLayer[i];
	}
	memcpy(FuncPtr, Input, (xNetParam.NonLayer[0] * FP32_SIZE));

	neurons = 0;
	for (i = 1; i < (xNetParam.Layers-1); i++)
	{
		aux_index = neurons;
		neurons += xNetParam.NonLayer[i - 1];
		v_MACForwardProp_FP32(&ZPtr[neurons], i, &FuncPtr[aux_index]);		//	CALCULATE [Z = W*F + b]
		v_ReLUActiv_FP32(&ZPtr[neurons], i, &FuncPtr[neurons]);				//	CALCULATE [F = ACTIV(Z)]
	}
	aux_index = neurons;
	neurons += xNetParam.NonLayer[xNetParam.Layers - 2];
	v_MACForwardProp_FP32(&ZPtr[neurons], (xNetParam.Layers - 1), &FuncPtr[aux_index]);		//	CALCULATE [Z = W*F + b]
	v_LinearActiv_FP32(&ZPtr[neurons], (xNetParam.Layers - 1), &FuncPtr[neurons]);										//	CALCULATE [F = ACTIV(Z)]
}





//************************************************************************************************
//	ALGEBRAIC FUNCTIONS (Backpropagation and Optimization)
//************************************************************************************************

//**************************************************/**
//	@brief
//
//	@param[in]			
//	@return				None			None
//	-	obs.:
//**************************************************
void v_ProcessOutputError_FP32(float* PropErrorPtr, float* FuncPtr, float* ZPtr, float* YPtr)
{
	uint32_t	index = 0;

	for (size_t i = 0; i < (xNetParam.Layers - 1); i++)
	{
		index += xNetParam.NonLayer[i];
	}

	for (size_t i = 0; i < xNetParam.NonLayer[xNetParam.Layers-1]; i++)
	{
		PropErrorPtr[index + i] = (FuncPtr[index + i] - YPtr[i]) / xNetParam.NonLayer[xNetParam.Layers - 1];
	}
}

//**************************************************/**
//	@brief Calculate for a given layer the following operation:	PropError[L-1] = W*PropError[L] \hadamard g'ativ(Z[L])		(equivalent to	RESULT	=	VEC \hadamard MATRIX*VEC)
//
//	@param[in]			netparam_t		xNetParam
//	@param[in]			uint8_t			layer				-> layer to which error will propagate
//	@param[in, out]		float*			PropErrorPtr		-> pointer addr of first value on (layer+1)
//	@param[in]			float*			ZPtr				-> pointer addr of first value on (layer+1)
//	@param[in]			float*			WBPtr
//	@return				None			None
//	-	to adapt the function, change float to int....
//**************************************************
void v_MACBackProp_FP32(uint8_t	layer, float* PropErrorPtr, float* ZPtr)
{
	float		aux;
	uint32_t	index_w, rows, cols;

	index_w = u32_GetIndex(xNetParam, (layer+1), 0, false);
	rows = xNetParam.NonLayer[layer + 1];
	cols = xNetParam.NonLayer[layer];

	for (size_t i = 0; i < cols; i++)
	{
		aux = 0;

		for (size_t j = 0; j < rows; j++)
		{
			aux += WBPtr[index_w + i + (j * cols)] * PropErrorPtr[j+cols];
		}
		//MULTIPLY AUX BY g'ativ
		if (ZPtr[i] < 0)
		{
			PropErrorPtr[i] = aux * xNetParam.LeakyReLU_rate / UINT16_MAX;
		}
		else
		{
			PropErrorPtr[i] = aux;
		}
	}
}

//**************************************************/**
//	@brief
//
//	@param[in]			
//	@return				None			None
//	-	obs.:
//**************************************************
void v_ProcessBackPropNN_FP32(float* PropErrorPtr, float* ZPtr, float* FuncPtr, float* Input, float* YPtr)
{
	uint16_t neurons = 0;

	for (size_t i = 0; i < (xNetParam.Layers-2); i++)
	{
		neurons += xNetParam.NonLayer[i];
	}

	v_ProcessForwardPropNN_FP32(ZPtr, FuncPtr, Input);
	v_ProcessOutputError_FP32(PropErrorPtr, FuncPtr, ZPtr, YPtr);

	for (uint8_t i = (xNetParam.Layers-2); i > 0; i--)
	{
		v_MACBackProp_FP32(i, &PropErrorPtr[neurons], &ZPtr[neurons]);
		neurons -= xNetParam.NonLayer[i];
	}
}

//**************************************************/**
//	@brief
//
//	@param[in]			
//	@return				None			None
//	-	obs.:
//**************************************************
void v_OptimizeWB_FP32(float* PropErrorPtr, float* FuncPtr)
{
	uint32_t	index_new, index_old, index_w, index_b, rows, cols;

	index_new = 0;

	for (uint16_t l = 1; l < (xNetParam.Layers-1); l++)
	{
		index_old = index_new;
		index_new += xNetParam.NonLayer[l-1];

		index_w = u32_GetIndex(xNetParam, l, 0, false);
		index_b = index_w + (xNetParam.NonLayer[l - 1] * xNetParam.NonLayer[l]);
		rows = xNetParam.NonLayer[l];
		cols = xNetParam.NonLayer[l - 1];

		for (size_t i = 0; i < rows; i++)
		{
			WBPtr[index_b + i] -= xNetParam.Learn_rate * PropErrorPtr[index_new + i] / UINT16_MAX;

			for (size_t j = 0; j < cols; j++)
			{
				WBPtr[index_w + (i * cols) + j] -= xNetParam.Learn_rate * (PropErrorPtr[index_new + i] * FuncPtr[index_old + j])/ UINT16_MAX;
			}
		}
	}
}

//**************************************************/**
//	@brief
//
//	@param[in]			
//	@return				None			None
//	-	obs.:
//**************************************************
float f_MeasureLoss_FP32(float* ZPtr, float* FuncPtr, DataSet_t xDataSet)
{
	float		Loss = 0;
	uint16_t	neurons = 0;

	for (size_t i = 0; i < (xNetParam.Layers-1); i++)
	{
		neurons += xNetParam.NonLayer[i];
	}

	for (size_t i = 0; i < xDataSet.test_size; i++)
	{
		v_ProcessForwardPropNN_FP32(ZPtr, FuncPtr, &xDataSet.x_test[i*xNetParam.NonLayer[0]]);
		for (size_t j = 0; j < xNetParam.NonLayer[xNetParam.Layers-1]; j++)
		{
			Loss += powf((FuncPtr[neurons + j] - xDataSet.y_test[i * xNetParam.NonLayer[xNetParam.Layers - 1] + j]), 2.0)/(2 * xNetParam.NonLayer[xNetParam.Layers - 1]);
		}
	}
	Loss = Loss / xDataSet.test_size;

	return Loss;
}

//**************************************************/**
//	@brief
//
//	@param[in]			
//	@return				None			None
//	-	obs.:
//**************************************************
void v_TrainNN_FP32(DataSet_t xDataSet, bool verbose)
{
	uint16_t	Batches = 0;
	uint16_t	Batch_size = 0;
	uint16_t	aux = 0;
	float*		ZPtr = NULL;
	float*		FuncPtr = NULL;
	float*		PropErrorPtr = NULL;
	float*		aux_PropErrorPtr = NULL;
	v_DynamicAllocForwardProp(&ZPtr, &FuncPtr);
	v_DynamicAllocBackProp(&PropErrorPtr);
	v_DynamicAllocBackProp(&aux_PropErrorPtr);

	for (size_t i = 0; i < xNetParam.EPOCHS; i++)
	{
		Batches = ((xDataSet.train_size + xNetParam.Batch_size - 1) / xNetParam.Batch_size);
		Batch_size = xNetParam.Batch_size;
		for (size_t j = 0; j < Batches; j++)
		{
			if (j == (Batches-1))
			{
				Batch_size = (xDataSet.train_size % xNetParam.Batch_size);
			}
			for (size_t k = 0; k < Batch_size; k++)
			{

				v_ProcessBackPropNN_FP32(PropErrorPtr, ZPtr, FuncPtr, &xDataSet.x_train[(j*xNetParam.Batch_size + k) * xNetParam.NonLayer[0]], &xDataSet.y_train[(j * xNetParam.Batch_size + k) * xNetParam.NonLayer[xNetParam.Layers-1]]);
				
				aux = 0;
				if (k == 0)
				{
					for (size_t l = 0; l < xNetParam.Layers; l++)
					{
						for (size_t m = 0; m < xNetParam.NonLayer[l]; m++)
						{
							aux_PropErrorPtr[(aux + m)] = PropErrorPtr[(aux + m)];
						}
						aux += xNetParam.NonLayer[l];
					}
				}
				else
				{
					for (size_t l = 0; l < xNetParam.Layers; l++)
					{
						for (size_t m = 0; m < xNetParam.NonLayer[l]; m++)
						{
							aux_PropErrorPtr[(aux + m)] += PropErrorPtr[(aux + m)];
						}
						aux += xNetParam.NonLayer[l];
					}
				}
			}

			aux = 0;
			for (size_t l = 0; l < (xNetParam.Layers-1); l++)
			{
				aux += xNetParam.NonLayer[l];
				for (size_t m = 0; m < xNetParam.NonLayer[l+1]; m++)
				{
					aux_PropErrorPtr[(aux + m)] = aux_PropErrorPtr[(aux + m)]/xNetParam.Batch_size;
				}
			}

			v_OptimizeWB_FP32(aux_PropErrorPtr, FuncPtr);
		}
		if (verbose == true)
		{
			//printf("Validation_Loss = %f\n", f_MeasureLoss_FP32(ZPtr, FuncPtr, xDataSet));
		}
	}
	if(ZPtr)
		free(ZPtr);
	if(FuncPtr)
		free(FuncPtr);
	if(PropErrorPtr)
		free(PropErrorPtr);
	if(aux_PropErrorPtr)
		free(aux_PropErrorPtr);
}
