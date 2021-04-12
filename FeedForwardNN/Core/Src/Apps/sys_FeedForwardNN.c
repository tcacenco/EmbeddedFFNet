#include	"sys_FeedForwardNN.h"

//	FIND INDEX OF DYNAMICALLY ALLOCATED NETWORK VARIABLE
//**************************************************/**
//	@brief	FIND INDEX OF DYNAMICALLY ALLOCATED NETWORK VARIABLE
//
//	@param[in]		netparam_t	xNetParam		-> (NETWORK HYPERPARAMETERS STRUCT)
//	@param[in]		uint8_t		layer			-> (Layer)
//	@param[in]		bool		weightbias		-> (bool to determine whether weights or bias will be chosen: weight = false; bias = true;)
//	@param[out]		uint32_t	index			-> (Index)
//**************************************************
uint32_t u32_GetIndex(netparam_t	xNetParam, uint16_t	layer, uint16_t row, bool	weightbias)	//weight	=	false; bias	=	true;
{
	uint32_t	index = 0;

	for (size_t i = 1; i < layer; i++)
	{
		index += (1 + xNetParam.NonLayer[i - 1]) * xNetParam.NonLayer[i];
	}
	if (weightbias == true)
	{
		index += row + (1 + xNetParam.NonLayer[layer - 1]) * xNetParam.NonLayer[layer];
	}
	else
	{
		index += row * (1 + xNetParam.NonLayer[layer - 1]);
	}
	return index;
}

void v_MountByte_4bits(int8_t* OldVar_8bits, int8_t NewVar_4bits, int8_t LowHigh)
{
	if (LowHigh == 0)
	{
		*OldVar_8bits = (*OldVar_8bits & (0xf0)) | (NewVar_4bits & (0x0f));
	}
	else
	{
		*OldVar_8bits = ((NewVar_4bits & (0x0f)) << 4) | (*OldVar_8bits & (0x0f));
	}
}

int8_t v_GetByte_4bits(int8_t Var_8bits, int8_t LowHigh)
{
	if (LowHigh == 0)
	{
		// 2nd Complement Adjustment for 4bit to 8bit transposition
		if ((Var_8bits & (1 << 3)))
		{
			return ((0xf0) | (Var_8bits & (0x0f)));
		}
		else
		{
			return (Var_8bits & (0x0f));
		}
	}
	else
	{
		// 2nd Complement Adjustment for 4bit to 8bit transposition
		if ((Var_8bits & (1 << 7)))
		{
			return ((0xf0) | ((Var_8bits & (0xf0)) >> 4));
		}
		else
		{
			return ((Var_8bits & (0xf0)) >> 4);
		}
	}
}