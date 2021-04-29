#include	"int_FeedForwardNN.h"

static volatile netparam_t	xNetParam;
static volatile quantizedval_t xQuantizedVal;
static volatile distribution_t xDistribution;

//************************************************************************************************
//	MEMORY FUNCTIONS
//************************************************************************************************
void v_DynamicAlloc_quant(quantizedval_t* xQuantizedVal, distribution_t* xDistribution, netparam_t	xNetParam)
{
	uint16_t	i;
	uint32_t	values = 0;
	uint16_t	neurons = 0;
	for (i = 0; i < (xNetParam.Layers - 1); i++)
	{
		neurons += xNetParam.NonLayer[i];
		values += ((1 + xNetParam.NonLayer[i]) * xNetParam.NonLayer[i + 1]);
	}
	neurons += xNetParam.NonLayer[xNetParam.Layers - 1];

	xQuantizedVal->WBPtr = (void*)malloc(ceil((float)(values * xNetParam.xVarPrecision.precision) / 8.0));

	xQuantizedVal->Zy = (float*)malloc(xNetParam.Layers * FP32_SIZE);
	xQuantizedVal->Sy = (float*)malloc(xNetParam.Layers * FP32_SIZE);
	xQuantizedVal->Sz = (float*)malloc(xNetParam.Layers * FP32_SIZE);
	xQuantizedVal->Sw = (float*)malloc(xNetParam.Layers * FP32_SIZE);
	xQuantizedVal->Sb = (float*)malloc(xNetParam.Layers * FP32_SIZE);
	xQuantizedVal->Ssumcomp = (float*)malloc(xNetParam.Layers * FP32_SIZE);
	xQuantizedVal->Sbiascomp = (float*)malloc(xNetParam.Layers * FP32_SIZE);
	xQuantizedVal->ZWsum = (float*)malloc(xNetParam.Layers * FP32_SIZE);
	xDistribution->W_min = (float*)malloc(xNetParam.Layers * FP32_SIZE);
	xDistribution->W_max = (float*)malloc(xNetParam.Layers * FP32_SIZE);
	xDistribution->b_min = (float*)malloc(xNetParam.Layers * FP32_SIZE);
	xDistribution->b_max = (float*)malloc(xNetParam.Layers * FP32_SIZE);
	xDistribution->func_min = (float*)malloc(xNetParam.Layers * FP32_SIZE);
	xDistribution->func_max = (float*)malloc(xNetParam.Layers * FP32_SIZE);
}

void v_FreeAlloc_quant(quantizedval_t	xQuantizedVal, distribution_t xDistribution)
{
	free(xQuantizedVal.WBPtr);
	free(xQuantizedVal.Zy);
	free(xQuantizedVal.Sy);
	free(xQuantizedVal.Sz);
	free(xQuantizedVal.Ssumcomp);
	free(xQuantizedVal.Sbiascomp);
	free(xQuantizedVal.ZWsum);
	free(xDistribution.W_min);
	free(xDistribution.W_max);
	free(xDistribution.b_min);
	free(xDistribution.b_max);
	free(xDistribution.func_min);
	free(xDistribution.func_max);
}

void v_SetQuantNetParameters(netparam_t	input_xnetparam, quantizedval_t input_xquantizedval, distribution_t input_xDistribution)
{
	xNetParam = input_xnetparam;
	xQuantizedVal = input_xquantizedval;
	xDistribution = input_xDistribution;
}

void v_DynamicAllocForwardProp_int(void** ZPtrPtr, void** FuncPtrPtr)
{
	uint16_t	i;
	uint32_t	neurons = 0;
	for (i = 0; i < (xNetParam.Layers - 1); i++)
	{
		neurons += xNetParam.NonLayer[i];
	}
	neurons += xNetParam.NonLayer[xNetParam.Layers - 1];

	*FuncPtrPtr = malloc(ceil((float)(neurons * xNetParam.xVarPrecision.precision) / 8.0));
	*ZPtrPtr = malloc(ceil((float)(neurons * xNetParam.xVarPrecision.precision) / 8.0));
}

//	calculate activation and store at funcptr
//**************************************************
//	in		zptr, xnetparam, layer, funcptr
//	out		none
//	obs.:	to adapt the function, change float to int....
//**************************************************
void v_MacForwardProp_int(void* res, uint8_t	layer, void* Funcptr, void* WBptr)
{
	uint32_t	index_w, index_b, rows, cols;
	long long int	long_aux = 0;
	int cont;

	index_w = u32_GetIndex(xNetParam, layer, 0, false);
	index_b = index_w + (xNetParam.NonLayer[layer - 1] * xNetParam.NonLayer[layer]);
	rows = xNetParam.NonLayer[layer];
	cols = xNetParam.NonLayer[layer - 1];

	switch (xNetParam.xVarPrecision.precision)
	{
	case 32:
		for (size_t i = 0; i < rows; i++)
		{
			long_aux = 0;
			for (size_t j = 0; j < cols; j++)
			{
				//long_aux += ((long long int)((int32_t*)WBptr)[index_w + (i * cols) + j] * (long long int)((int32_t*)Funcptr)[j]) & 0xffffffff00000000;	//mac
				long_aux = __SMMLA(((int32_t*)WBptr)[index_w + (i * cols) + j], ((int32_t*)Funcptr)[j], long_aux);
			}

			//long_aux = (round(xQuantizedVal.Ssumcomp[layer] * long_aux) + (xNetParam.NonLayer[layer] * ((int32_t*)WBptr)[index_b + i]));
			long_aux = (round(xQuantizedVal.Ssumcomp[layer] * (long_aux << 32)) + (((int32_t*)WBptr)[index_b + i]));

			// OVERFLOW and UNDERFLOW PROTECTION
			if (long_aux >= INT32_MAX)
			{
				long_aux = INT32_MAX;
			}
			else if (long_aux <= INT32_MIN)
			{
				long_aux = INT32_MIN;
			}
			((int32_t*)res)[i] = (int32_t)long_aux;
		}
		break;

	case 16:
		cont = (cols/2 + cols%2);
		for (size_t i = 0; i < rows; i++)
		{
			long_aux = 0;
			for (size_t j = 0; j < cont; j++)
			{
				//long_aux += ((int16_t*)WBptr)[index_w + (i * cols) + j] * ((int16_t*)Funcptr)[j];	//mac
				if((j < (cont-1)) | (cols%2 == 0))
					long_aux = (int32_t)__SMLAD((((((int16_t*)WBptr)[index_w + (i * cols) + 2*j] << 16) & 0xffff0000) | ((((int16_t*)WBptr)[index_w + (i * cols) + 2*j + 1]) & 0x0000ffff)), (((((int16_t*)Funcptr)[2*j] << 16) & 0xffff0000) | ((((int16_t*)Funcptr)[2*j + 1]) & 0x0000ffff)), long_aux);
				else
					long_aux = (int32_t)__SMLAD((((int16_t*)WBptr)[index_w + (i * cols) + 2*j] & 0x0000ffff), (((int16_t*)Funcptr)[2*j] & 0x0000ffff), long_aux);
			}

			//long_aux = (round(xQuantizedVal.Ssumcomp[layer] * long_aux) + (xNetParam.NonLayer[layer] * ((int16_t*)WBptr)[index_b + i]));
			long_aux = (round(xQuantizedVal.Ssumcomp[layer] * long_aux) + (((int16_t*)WBptr)[index_b + i]));

			// OVERFLOW and UNDERFLOW PROTECTION
			if (long_aux >= INT16_MAX)
			{
				long_aux = INT16_MAX;
			}
			else if (long_aux <= INT16_MIN)
			{
				long_aux = INT16_MIN;
			}
			((int16_t*)res)[i] = (int16_t)long_aux;
		}
		break;

	case 8:
		cont = (cols/2 + cols%2);
		for (size_t i = 0; i < rows; i++)
		{
			long_aux = 0;
			for (size_t j = 0; j < cont; j++)
			{
				//long_aux += ((int8_t*)WBptr)[index_w + (i * cols) + j] * ((int8_t*)Funcptr)[j];	//mac
				if((j < (cont-1)) | (cols%2 == 0))
					long_aux = (int32_t)__SMLAD((((((int8_t*)WBptr)[index_w + (i * cols) + 2*j] << 16) & 0xffff0000) | ((((int8_t*)WBptr)[index_w + (i * cols) + 2*j + 1]) & 0x0000ffff)), (((((int8_t*)Funcptr)[2*j] << 16) & 0xffff0000) | ((((int8_t*)Funcptr)[2*j + 1]) & 0x0000ffff)), long_aux);
				else
					long_aux = (int32_t)__SMLAD((((int8_t*)WBptr)[index_w + (i * cols) + 2*j] & 0x0000ffff), (((int8_t*)Funcptr)[2*j] & 0x0000ffff), long_aux);
			}

			//long_aux = (round(xQuantizedVal.Ssumcomp[layer] * long_aux) + (xNetParam.NonLayer[layer] * ((int8_t*)WBptr)[index_b + i]));
			long_aux = (round(xQuantizedVal.Ssumcomp[layer] * long_aux) + (((int8_t*)WBptr)[index_b + i]));

			// OVERFLOW and UNDERFLOW PROTECTION
			if (long_aux >= INT8_MAX)
			{
				long_aux = INT8_MAX;
			}
			else if (long_aux <= INT8_MIN)
			{
				long_aux = INT8_MIN;
			}
			((int8_t*)res)[i] = (int8_t)long_aux;
		}
		break;

	case 4:;
		int16_t EvOdd_Lm1 = 0;
		int16_t EvOdd_L = 0;
		for (size_t i = 0; i < (layer - 1); i++)
		{
			EvOdd_Lm1 += xNetParam.NonLayer[i];
		}
		EvOdd_L = (EvOdd_Lm1 + xNetParam.NonLayer[layer - 1]) % 2;
		EvOdd_Lm1 = EvOdd_Lm1 % 2;

		cont = (cols/2 + cols%2);
		for (size_t i = 0; i < rows; i++)
		{
			long_aux = 0;
			for (size_t j = 0; j < cont; j++)
			{
				//long_aux += ((int8_t*)WBptr)[index_w + (i * cols) + j] * ((int8_t*)Funcptr)[j];	//mac
				//long_aux += v_GetByte_4bits(((int8_t*)WBptr)[(index_w + (i * cols) + j) / 2], (index_w + (i * cols) + j) % 2) * v_GetByte_4bits(((int8_t*)Funcptr)[(j + EvOdd_Lm1) / 2], (j + EvOdd_Lm1) % 2);

				if((j < (cont-1)) | (cols%2 == 0))
					long_aux = (int32_t)__SMLAD((((v_GetByte_4bits(((int8_t*)WBptr)[(index_w + (i * cols) + 2*j) / 2], (index_w + (i * cols) + 2*j) % 2) << 16) & 0xffff0000) | ((v_GetByte_4bits(((int8_t*)WBptr)[(index_w + (i * cols) + 2*j + 1) / 2], (index_w + (i * cols) + 2*j + 1) % 2)) & 0x0000ffff)), (((v_GetByte_4bits(((int8_t*)Funcptr)[(2*j + EvOdd_Lm1) / 2], (2*j + EvOdd_Lm1) % 2) << 16) & 0xffff0000) | ((v_GetByte_4bits(((int8_t*)Funcptr)[(2*j + 1 + EvOdd_Lm1) / 2], (2*j + 1 + EvOdd_Lm1) % 2)) & 0x0000ffff)), long_aux);
				else
					long_aux = (int32_t)__SMLAD((((int8_t*)WBptr)[index_w + (i * cols) + 2*j] & 0x0000ffff), (((int8_t*)Funcptr)[2*j] & 0x0000ffff), long_aux);
			}

			//long_aux = (round(xQuantizedVal.Ssumcomp[layer] * long_aux) + (xNetParam.NonLayer[layer] * ((int8_t*)WBptr)[index_b + i]));
			//long_aux = round(xQuantizedVal.Ssumcomp[layer] * long_aux) + (xNetParam.NonLayer[layer] * v_GetByte_4bits(((int8_t*)WBptr)[(index_b + i) / 2], (index_b + i) % 2));
			long_aux = round(xQuantizedVal.Ssumcomp[layer] * long_aux) + (v_GetByte_4bits(((int8_t*)WBptr)[(index_b + i) / 2], (index_b + i) % 2));

			// OVERFLOW and UNDERFLOW PROTECTION
			if (long_aux >= INT4_MAX)
			{
				long_aux = INT4_MAX;
			}
			else if (long_aux <= INT4_MIN)
			{
				long_aux = INT4_MIN;
			}
			//((int8_t*)res)[i] = (int8_t)long_aux;
			v_MountByte_4bits(&((int8_t*)res)[(EvOdd_L + i) / 2], (int8_t)long_aux, (EvOdd_L + i) % 2);
		}
		break;

	default:
		break;
	}
}

//	calculate activation and store at funcptr
//**************************************************
//	in		zptr, xnetparam, layer, funcptr
//	out		none
//	obs.:	to adapt the function, change float to int....
//**************************************************
void v_ReLUActiv_int(void* ZPtr, uint16_t layer, void* funcptr)
{
	long long int long_aux;
	switch (xNetParam.xVarPrecision.precision)
	{
	case 32:
		for (size_t i = 0; i < xNetParam.NonLayer[layer]; i++)
		{
			if (((int32_t*)ZPtr)[i] < 0)
			{
				long_aux = (((long long int)xNetParam.LeakyReLU_rate * ((int32_t*)ZPtr)[i]) / UINT16_MAX) + xQuantizedVal.Zy[layer];
			}
			else
			{
				long_aux = ((int32_t*)ZPtr)[i] + xQuantizedVal.Zy[layer];
			}
			if (long_aux >= INT32_MAX)
			{
				long_aux = INT32_MAX;
			}
			else if (long_aux <= INT32_MIN)
			{
				long_aux = INT32_MIN;
			}
			((int32_t*)funcptr)[i] = (int32_t)long_aux;
		}
		break;

	case 16:
		for (size_t i = 0; i < xNetParam.NonLayer[layer]; i++)
		{
			if (((int16_t*)ZPtr)[i] < 0)
			{
				long_aux = (((long long int)xNetParam.LeakyReLU_rate * ((int16_t*)ZPtr)[i]) / UINT16_MAX) + xQuantizedVal.Zy[layer];
			}
			else
			{
				long_aux = ((int16_t*)ZPtr)[i] + xQuantizedVal.Zy[layer];
			}
			if (long_aux >= INT16_MAX)
			{
				long_aux = INT16_MAX;
			}
			else if (long_aux <= INT16_MIN)
			{
				long_aux = INT16_MIN;
			}
			((int16_t*)funcptr)[i] = (int16_t)long_aux;
		}
		break;

	case 8:
		for (size_t i = 0; i < xNetParam.NonLayer[layer]; i++)
		{
			if (((int8_t*)ZPtr)[i] < 0)
			{
				long_aux = (((long long int)xNetParam.LeakyReLU_rate * ((int8_t*)ZPtr)[i]) / UINT16_MAX) + xQuantizedVal.Zy[layer];
			}
			else
			{
				long_aux = ((int8_t*)ZPtr)[i] + xQuantizedVal.Zy[layer];
			}
			if (long_aux >= INT8_MAX)
			{
				long_aux = INT8_MAX;
			}
			else if (long_aux <= INT8_MIN)
			{
				long_aux = INT8_MIN;
			}
			((int8_t*)funcptr)[i] = (int8_t)long_aux;
		}
		break;

	case 4:;
		int16_t EvOdd_L = 0;
		for (size_t i = 0; i < layer; i++)
		{
			EvOdd_L += xNetParam.NonLayer[i];
		}
		EvOdd_L = EvOdd_L % 2;

		for (size_t i = 0; i < xNetParam.NonLayer[layer]; i++)
		{
			long_aux = v_GetByte_4bits(((int8_t*)ZPtr)[(i + EvOdd_L) / 2], (i + EvOdd_L) % 2);
			if (long_aux < 0)
			{
				long_aux = (((long long int)xNetParam.LeakyReLU_rate * long_aux) / UINT16_MAX) + xQuantizedVal.Zy[layer];
			}
			else
			{
				long_aux = long_aux + xQuantizedVal.Zy[layer];
			}
			if (long_aux >= INT4_MAX)
			{
				long_aux = INT4_MAX;
			}
			else if (long_aux <= INT4_MIN)
			{
				long_aux = INT4_MIN;
			}
			v_MountByte_4bits(&((int8_t*)funcptr)[(i + EvOdd_L) / 2], (int8_t)long_aux, (i + EvOdd_L) % 2);
			//((int8_t*)funcptr)[i] = (int8_t)long_aux;
		}
		break;

	default:
		break;
	}
}

//	process all algebraic operations related to the feedforward algorithm and return nn output value (pointer)
//**************************************************
//	in		result, xnetparam, input, wbptr			(result, network hyperparameters struct,float pointer, float pointer)
//	out		none
//	obs.:	to adapt the function, change float to int....
//**************************************************
void v_LinearActiv_int(void* ZPtr, uint16_t layer, void* FuncPtr)
{
	long long int long_aux;
	switch (xNetParam.xVarPrecision.precision)
	{
	case 32:
		for (size_t i = 0; i < xNetParam.NonLayer[layer]; i++)
		{
			long_aux = (((int32_t*)ZPtr)[i] + xQuantizedVal.Zy[layer]);

			if (long_aux >= INT32_MAX)
			{
				long_aux = INT32_MAX;
			}
			else if (long_aux <= INT32_MIN)
			{
				long_aux = INT32_MIN;
			}
			((int32_t*)FuncPtr)[i] = long_aux;
		}
		break;

	case 16:
		for (size_t i = 0; i < xNetParam.NonLayer[layer]; i++)
		{
			long_aux = (((int16_t*)ZPtr)[i] + xQuantizedVal.Zy[layer]);

			if (long_aux >= INT16_MAX)
			{
				long_aux = INT16_MAX;
			}
			else if (long_aux <= INT16_MIN)
			{
				long_aux = INT16_MIN;
			}
			((int16_t*)FuncPtr)[i] = long_aux;
		}
		break;

	case 8:
		for (size_t i = 0; i < xNetParam.NonLayer[layer]; i++)
		{
			long_aux = (((int8_t*)ZPtr)[i] + xQuantizedVal.Zy[layer]);

			if (long_aux >= INT8_MAX)
			{
				long_aux = INT8_MAX;
			}
			else if (long_aux <= INT8_MIN)
			{
				long_aux = INT8_MIN;
			}
			((int8_t*)FuncPtr)[i] = long_aux;
		}
		break;

	case 4:;
		int16_t EvOdd_L = 0;
		for (size_t i = 0; i < layer; i++)
		{
			EvOdd_L += xNetParam.NonLayer[i];
		}
		EvOdd_L = EvOdd_L % 2;

		for (size_t i = 0; i < xNetParam.NonLayer[layer]; i++)
		{
			long_aux = v_GetByte_4bits(((int8_t*)ZPtr)[(i + EvOdd_L) / 2], (i + EvOdd_L) % 2) + xQuantizedVal.Zy[layer];

			if (long_aux >= INT4_MAX)
			{
				long_aux = INT4_MAX;
			}
			else if (long_aux <= INT4_MIN)
			{
				long_aux = INT4_MIN;
			}
			v_MountByte_4bits(&((int8_t*)FuncPtr)[(i + EvOdd_L) / 2], (int8_t)long_aux, (i + EvOdd_L) % 2);
		}
		break;

	default:
		break;
	}
}

//	process all algebraic operations related to the feedforward algorithm and return nn output value (pointer)
//**************************************************
//	in		result, xnetparam, input, wbptr			(result, network hyperparameters struct,float pointer, float pointer)
//	out		none
//	obs.:	to adapt the function, change float to int....
//**************************************************
void v_QuantizeIntputs_int(void* quant_input, float* input)
{
	switch (xNetParam.xVarPrecision.precision)
	{
	case 32:
		for (size_t i = 0; i < xNetParam.NonLayer[0]; i++)
		{
			// Saturation Protection
			if (input[i] >= xDistribution.func_max[0])
			{
				((int32_t*)quant_input)[i] = (int32_t)round((xDistribution.func_max[0] / xQuantizedVal.Sy[0]) + xQuantizedVal.Zy[0]);
			}
			else if (input[i] <= xDistribution.func_min[0])
			{
				((int32_t*)quant_input)[i] = (int32_t)round((xDistribution.func_min[0] / xQuantizedVal.Sy[0]) + xQuantizedVal.Zy[0]);
			}
			else
			{
				((int32_t*)quant_input)[i] = (int32_t)round((input[i] / xQuantizedVal.Sy[0]) + xQuantizedVal.Zy[0]);
			}
		}
		break;

	case 16:
		for (size_t i = 0; i < xNetParam.NonLayer[0]; i++)
		{

			// Saturation Protection
			if (input[i] >= xDistribution.func_max[0])
			{
				((int16_t*)quant_input)[i] = (int16_t)round((xDistribution.func_max[0] / xQuantizedVal.Sy[0]) + xQuantizedVal.Zy[0]);
			}
			else if (input[i] <= xDistribution.func_min[0])
			{
				((int16_t*)quant_input)[i] = (int16_t)round((xDistribution.func_min[0] / xQuantizedVal.Sy[0]) + xQuantizedVal.Zy[0]);
			}
			else
			{
				((int16_t*)quant_input)[i] = (int16_t)round((input[i] / xQuantizedVal.Sy[0]) + xQuantizedVal.Zy[0]);
			}
		}
		break;

	case 8:
		for (size_t i = 0; i < xNetParam.NonLayer[0]; i++)
		{

			// Saturation Protection
			if (input[i] >= xDistribution.func_max[0])
			{
				((int8_t*)quant_input)[i] = (int8_t)round((xDistribution.func_max[0] / xQuantizedVal.Sy[0]) + xQuantizedVal.Zy[0]);
			}
			else if (input[i] <= xDistribution.func_min[0])
			{
				((int8_t*)quant_input)[i] = (int8_t)round((xDistribution.func_min[0] / xQuantizedVal.Sy[0]) + xQuantizedVal.Zy[0]);
			}
			else
			{
				((int8_t*)quant_input)[i] = (int8_t)round((input[i] / xQuantizedVal.Sy[0]) + xQuantizedVal.Zy[0]);
			}
		}
		break;

	case 4:
		for (size_t i = 0; i < xNetParam.NonLayer[0]; i++)
		{
			// Saturation Protection
			if (input[i] >= xDistribution.func_max[0])
			{
				v_MountByte_4bits(&((int8_t*)quant_input)[i / 2], (int8_t)round((xDistribution.func_max[0] / xQuantizedVal.Sy[0]) + xQuantizedVal.Zy[0]), i % 2);
			}
			else if (input[i] <= xDistribution.func_min[0])
			{
				v_MountByte_4bits(&((int8_t*)quant_input)[i / 2], (int8_t)round((xDistribution.func_min[0] / xQuantizedVal.Sy[0]) + xQuantizedVal.Zy[0]), i % 2);
			}
			else
			{
				v_MountByte_4bits(&((int8_t*)quant_input)[i / 2], (int8_t)round((input[i] / xQuantizedVal.Sy[0]) + xQuantizedVal.Zy[0]), i % 2);
			}
		}
		break;

	default:
		break;
	}
}

//	process all algebraic operations related to the feedforward algorithm and return nn output value (pointer)
//**************************************************
//	in		result, xnetparam, input, wbptr			(result, network hyperparameters struct,float pointer, float pointer)
//	out		none
//	obs.:	to adapt the function, change float to int....
//**************************************************
void v_ProcessForwardPropNN_int(void* ZPtr, void* FuncPtr, void* input)
{
	uint16_t	i;
	uint32_t	neurons = 0;
	uint32_t	aux_index = 0;

	switch (xNetParam.xVarPrecision.precision)
	{
	case 32:
		v_QuantizeIntputs_int(&((int32_t*)FuncPtr)[0], (float*)input);
		for (i = 1; i < (xNetParam.Layers - 1); i++)
		{
			aux_index = neurons;
			neurons += xNetParam.NonLayer[i - 1];
			v_MacForwardProp_int(&((int32_t*)ZPtr)[neurons], i, &((int32_t*)FuncPtr)[aux_index], xQuantizedVal.WBPtr);		//	calculate [z = w*f + b]
			v_ReLUActiv_int(&((int32_t*)ZPtr)[neurons], i, &((int32_t*)FuncPtr)[neurons]);									//	calculate [f = activ(z)]
		}
		aux_index = neurons;
		neurons += xNetParam.NonLayer[xNetParam.Layers - 2];
		v_MacForwardProp_int(&((int32_t*)ZPtr)[neurons], i, &((int32_t*)FuncPtr)[aux_index], xQuantizedVal.WBPtr);			//	calculate [z = w*f + b]
		v_LinearActiv_int(&((int32_t*)ZPtr)[neurons], i, &((int32_t*)FuncPtr)[neurons]);									//	calculate [f = activ(z)]
		break;

	case 16:
		v_QuantizeIntputs_int(&((int16_t*)FuncPtr)[0], (float*)input);
		for (i = 1; i < (xNetParam.Layers - 1); i++)
		{
			aux_index = neurons;
			neurons += xNetParam.NonLayer[i - 1];
			v_MacForwardProp_int(&((int16_t*)ZPtr)[neurons], i, &((int16_t*)FuncPtr)[aux_index], xQuantizedVal.WBPtr);		//	calculate [z = w*f + b]
			v_ReLUActiv_int(&((int16_t*)ZPtr)[neurons], i, &((int16_t*)FuncPtr)[neurons]);									//	calculate [f = activ(z)]
		}
		aux_index = neurons;
		neurons += xNetParam.NonLayer[xNetParam.Layers - 2];
		v_MacForwardProp_int(&((int16_t*)ZPtr)[neurons], i, &((int16_t*)FuncPtr)[aux_index], xQuantizedVal.WBPtr);			//	calculate [z = w*f + b]
		v_LinearActiv_int(&((int16_t*)ZPtr)[neurons], i, &((int16_t*)FuncPtr)[neurons]);									//	calculate [f = activ(z)]
		break;

	case 8:
		v_QuantizeIntputs_int(&((int8_t*)FuncPtr)[0], (float*)input);
		for (i = 1; i < (xNetParam.Layers - 1); i++)
		{
			aux_index = neurons;
			neurons += xNetParam.NonLayer[i - 1];
			v_MacForwardProp_int(&((int8_t*)ZPtr)[neurons], i, &((int8_t*)FuncPtr)[aux_index], xQuantizedVal.WBPtr);		//	calculate [z = w*f + b]
			v_ReLUActiv_int(&((int8_t*)ZPtr)[neurons], i, &((int8_t*)FuncPtr)[neurons]);									//	calculate [f = activ(z)]
		}
		aux_index = neurons;
		neurons += xNetParam.NonLayer[xNetParam.Layers - 2];
		v_MacForwardProp_int(&((int8_t*)ZPtr)[neurons], i, &((int8_t*)FuncPtr)[aux_index], xQuantizedVal.WBPtr);			//	calculate [z = w*f + b]
		v_LinearActiv_int(&((int8_t*)ZPtr)[neurons], i, &((int8_t*)FuncPtr)[neurons]);									//	calculate [f = activ(z)]

		break;

	case 4:
		v_QuantizeIntputs_int(&((int8_t*)FuncPtr)[0], (float*)input);
		for (i = 1; i < (xNetParam.Layers - 1); i++)
		{
			aux_index = neurons;
			neurons += xNetParam.NonLayer[i - 1];
			v_MacForwardProp_int(&((int8_t*)ZPtr)[neurons / 2], i, &((int8_t*)FuncPtr)[aux_index / 2], xQuantizedVal.WBPtr);		//	calculate [z = w*f + b]
			v_ReLUActiv_int(&((int8_t*)ZPtr)[neurons / 2], i, &((int8_t*)FuncPtr)[neurons / 2]);									//	calculate [f = activ(z)]
		}
		aux_index = neurons;
		neurons += xNetParam.NonLayer[xNetParam.Layers - 2];
		v_MacForwardProp_int(&((int8_t*)ZPtr)[neurons / 2], i, &((int8_t*)FuncPtr)[aux_index / 2], xQuantizedVal.WBPtr);			//	calculate [z = w*f + b]
		v_LinearActiv_int(&((int8_t*)ZPtr)[neurons / 2], i, &((int8_t*)FuncPtr)[neurons / 2]);									//	calculate [f = activ(z)]

		break;

	default:
		break;
	}
}
