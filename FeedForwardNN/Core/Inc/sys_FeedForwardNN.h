#include	<stdio.h>
#include	<stdlib.h>
#include	<stdint.h>
#include	<math.h>
#include	<stdbool.h>
#include	<string.h>

#ifndef SYS_FEEDFORWARDNN
#define SYS_FEEDFORWARDNN

#define	FP32_SIZE	4

#define INT4_MAX	15
#define INT4_MIN	-16

//************************************************************************************************
//	STRUCT TYPEDEF
//************************************************************************************************
typedef	struct DATASET
{
	float* x_train;
	float* y_train;
	uint16_t	train_size;					//<! Number os input/output sets to train (lines)

	float* x_test;
	float* y_test;
	uint16_t	test_size;					//<! Number os input/output sets to train (lines)
}DataSet_t;

typedef struct QUANTIZEDVAL
{
	float* Sy;
	float* Sz;
	float* Sw;
	float* Sb;
	float* Ssumcomp;
	float* Sbiascomp;
	float* ZWsum;
	float* Zy;
	void* WBPtr;
}quantizedval_t;

typedef struct DISTRIBUTION
{
	float* W_min;
	float* W_max;
	float* b_min;
	float* b_max;
	float* func_min;
	float* func_max;
}distribution_t;

typedef	struct VARPRECISION {
	char	type;		//	float =	 'f';		int	=	'i'
	uint8_t	precision;
}varprecision_t;

typedef	struct NETPARAM {
	varprecision_t	xVarPrecision;
	uint16_t		Layers;
	uint16_t		NonLayer[32];
	uint16_t		LeakyReLU_rate;	//	value = LeakyReLU_rate / 65535
	uint16_t		Learn_rate;		//	value = Learn_rate / 65535
	uint16_t		Batch_size;
	uint16_t		EPOCHS;
	char			GativOutput[8];	//	"linear", "sigmoid", "tanh"...
}netparam_t;


uint32_t u32_GetIndex(netparam_t	xNetParam, uint16_t	layer, uint16_t row, bool	weightbias);

void v_MountByte_4bits(int8_t* OldVar_8bits, int8_t NewVar_4bits, int8_t LowHigh);

int8_t v_GetByte_4bits(int8_t Var_8bits, int8_t LowHigh);

#endif