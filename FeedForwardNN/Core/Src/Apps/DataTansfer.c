#include	"DataTransfer.h"

// TODO: TESTAR TUDO NOVAMENTE

void v_LoadFloatModel(netparam_t* xNetParam, float** WBPtr)
{
	uint16_t	i;
	uint32_t	values = 0;
	uint8_t* 	DataPtr;

	DataPtr = (uint8_t*)BASEADDR_FLOATMODEL;

	memcpy(xNetParam, DataPtr, sizeof(*xNetParam));

	v_FloatSetParameters(*xNetParam);
	v_DynamicAlloc_NeuronVar(WBPtr);

	for (i = 0; i < (xNetParam->Layers - 1); i++)
	{
		values += ((1 + xNetParam->NonLayer[i]) * xNetParam->NonLayer[i + 1]);
	}

	memcpy(*WBPtr, &DataPtr[sizeof(*xNetParam)], (values * sizeof(float)));
}

//void v_SaveFloatModel(netparam_t xNetParam, float *WBPtr)
//{
//	FILE* FPtr = NULL;
//	uint16_t	i;
//	uint32_t	values = 0;
//
//	for (i = 0; i < (xNetParam.Layers - 1); i++)
//	{
//		values += ((1 + xNetParam.NonLayer[i]) * xNetParam.NonLayer[i + 1]);
//	}
//
//	fopen_s(&FPtr, "modelfloat.bin", "wb");
//
//	fwrite(&xNetParam, sizeof(xNetParam), 1, FPtr);
//	fwrite(WBPtr, sizeof(float), values, FPtr);
//	fclose(FPtr);
//}

void v_LoadIntModel(netparam_t *xNetParam, distribution_t *xDistribution, quantizedval_t *xQuantizedVal)
{
	uint16_t	i;
	uint32_t	values = 0;
	uint8_t*	DataPtr;

	DataPtr = (uint8_t*)BASEADDR_INTMODEL;

	memcpy(xNetParam, DataPtr, sizeof(*xNetParam));

	v_DynamicAlloc_quant(xQuantizedVal, xDistribution, *xNetParam);

	for (i = 0; i < (xNetParam->Layers - 1); i++)
	{
		values += ((1 + xNetParam->NonLayer[i]) * xNetParam->NonLayer[i + 1]);
	}

	memcpy(xDistribution->W_min,		&DataPtr[sizeof(*xNetParam)], (sizeof(float) * xNetParam->Layers));
	memcpy(xDistribution->W_max,		&DataPtr[sizeof(*xNetParam)	+ 1*(sizeof(float) * xNetParam->Layers)], (sizeof(float) * xNetParam->Layers));
	memcpy(xDistribution->b_min,		&DataPtr[sizeof(*xNetParam)	+ 2*(sizeof(float) * xNetParam->Layers)], (sizeof(float) * xNetParam->Layers));
	memcpy(xDistribution->b_max,		&DataPtr[sizeof(*xNetParam)	+ 3*(sizeof(float) * xNetParam->Layers)], (sizeof(float) * xNetParam->Layers));
	memcpy(xDistribution->func_min,		&DataPtr[sizeof(*xNetParam) + 4*(sizeof(float) * xNetParam->Layers)], (sizeof(float) * xNetParam->Layers));
	memcpy(xDistribution->func_max,		&DataPtr[sizeof(*xNetParam) + 5*(sizeof(float) * xNetParam->Layers)], (sizeof(float) * xNetParam->Layers));
	memcpy(xQuantizedVal->Sy,			&DataPtr[sizeof(*xNetParam)	+ 6 * (sizeof(float) * xNetParam->Layers)], (sizeof(float) * xNetParam->Layers));
	memcpy(xQuantizedVal->Sz,			&DataPtr[sizeof(*xNetParam)	+ 7 * (sizeof(float) * xNetParam->Layers)], (sizeof(float) * xNetParam->Layers));
	memcpy(xQuantizedVal->Sw,			&DataPtr[sizeof(*xNetParam)	+ 8 * (sizeof(float) * xNetParam->Layers)], (sizeof(float) * xNetParam->Layers));
	memcpy(xQuantizedVal->Sb,			&DataPtr[sizeof(*xNetParam)	+ 9 * (sizeof(float) * xNetParam->Layers)], (sizeof(float) * xNetParam->Layers));
	memcpy(xQuantizedVal->Ssumcomp,		&DataPtr[sizeof(*xNetParam) + 10 * (sizeof(float) * xNetParam->Layers)], (sizeof(float) * xNetParam->Layers));
	memcpy(xQuantizedVal->Sbiascomp,	&DataPtr[sizeof(*xNetParam) + 11 * (sizeof(float) * xNetParam->Layers)], (sizeof(float) * xNetParam->Layers));
	memcpy(xQuantizedVal->ZWsum,		&DataPtr[sizeof(*xNetParam) + 12 * (sizeof(float) * xNetParam->Layers)], (sizeof(float) * xNetParam->Layers));
	memcpy(xQuantizedVal->Zy,			&DataPtr[sizeof(*xNetParam) + 13 * (sizeof(float) * xNetParam->Layers)], (sizeof(float) * xNetParam->Layers));

	memcpy(((uint8_t*)xQuantizedVal->WBPtr), &DataPtr[sizeof(*xNetParam) + 14 * (sizeof(float) * xNetParam->Layers)], ceil((values * xNetParam->xVarPrecision.precision) / 8.0));
}

//void v_SaveIntModel(netparam_t xNetParam, distribution_t xDistribution, quantizedval_t xQuantizedVal)
//{
//	FILE* FPtr = NULL;
//	uint16_t	i;
//	uint32_t	values = 0;
//
//	for (i = 0; i < (xNetParam.Layers - 1); i++)
//	{
//		values += ((1 + xNetParam.NonLayer[i]) * xNetParam.NonLayer[i + 1]);
//	}
//
//	fopen_s(&FPtr, "modelint.bin", "wb");
//
//	fwrite(&xNetParam, sizeof(xNetParam), 1, FPtr);
//
//	fwrite(xDistribution.W_min, sizeof(float), xNetParam.Layers, FPtr);
//	fwrite(xDistribution.W_max, sizeof(float), xNetParam.Layers, FPtr);
//	fwrite(xDistribution.b_min, sizeof(float), xNetParam.Layers, FPtr);
//	fwrite(xDistribution.b_max, sizeof(float), xNetParam.Layers, FPtr);
//	fwrite(xDistribution.func_min, sizeof(float), xNetParam.Layers, FPtr);
//	fwrite(xDistribution.func_max, sizeof(float), xNetParam.Layers, FPtr);
//
//	fwrite(xQuantizedVal.Sy, sizeof(float), xNetParam.Layers, FPtr);
//	fwrite(xQuantizedVal.Sz, sizeof(float), xNetParam.Layers, FPtr);
//	fwrite(xQuantizedVal.Sw, sizeof(float), xNetParam.Layers, FPtr);
//	fwrite(xQuantizedVal.Sb, sizeof(float), xNetParam.Layers, FPtr);
//	fwrite(xQuantizedVal.Ssumcomp, sizeof(float), xNetParam.Layers, FPtr);
//	fwrite(xQuantizedVal.Sbiascomp, sizeof(float), xNetParam.Layers, FPtr);
//	fwrite(xQuantizedVal.ZWsum, sizeof(float), xNetParam.Layers, FPtr);
//	fwrite(xQuantizedVal.Zy, sizeof(float), xNetParam.Layers, FPtr);
//
//	fwrite(((uint8_t*)xQuantizedVal.WBPtr), sizeof(uint8_t), ((values * xNetParam.xVarPrecision.precision) / 8.0), FPtr);
//
//	fclose(FPtr);
//}

// Must be called after loading Model
void v_LoadTestDataSet(uint8_t * DataPtr, DataSet_t * xDataSet, netparam_t	xNetParam)
{
	memcpy(&(xDataSet->test_size), &DataPtr[0], sizeof(uint16_t));

	if (xDataSet->x_test != NULL)
	{
		free(xDataSet->x_test);
	}
	if (xDataSet->y_test != NULL)
	{
		free(xDataSet->y_test);
	}

	xDataSet->x_test = malloc(sizeof(float) * xDataSet->test_size * xNetParam.NonLayer[0]);
	xDataSet->y_test = malloc(sizeof(float) * xDataSet->test_size * xNetParam.NonLayer[xNetParam.Layers - 1]);

	memcpy(xDataSet->x_test, &DataPtr[sizeof(uint16_t)], (xDataSet->test_size * sizeof(float) * xNetParam.NonLayer[0]));
	memcpy(xDataSet->y_test, &DataPtr[sizeof(uint16_t) + (xDataSet->test_size * sizeof(float) * xNetParam.NonLayer[0])], (xDataSet->test_size * sizeof(float) * xNetParam.NonLayer[xNetParam.Layers-1]));
}

//void v_SaveTestDataSet(DataSet_t xDataSet, netparam_t	xNetParam)
//{
//	FILE* FPtr = NULL;
//
//	fopen_s(&FPtr, "dataset.bin", "wb");
//
//	fwrite(&xDataSet.test_size, sizeof(uint16_t), 1, FPtr);
//	fwrite(xDataSet.x_test, sizeof(float), (xDataSet.test_size * xNetParam.NonLayer[0]), FPtr);
//	fwrite(xDataSet.y_test, sizeof(float), (xDataSet.test_size * xNetParam.NonLayer[xNetParam.Layers-1]), FPtr);
//
//	fclose(FPtr);
//}

void v_LoadTestDataNum(DataSet_t* xDataSet, netparam_t	xNetParam, uint16_t num)
{
	uint8_t* DataPtr;

	DataPtr = (uint8_t*)BASEADDR_DATASET;

	memcpy(&(xDataSet->test_size), &DataPtr[0], sizeof(uint16_t));

	if (xDataSet->x_test != NULL)
	{
		free(xDataSet->x_test);
	}
	if (xDataSet->y_test != NULL)
	{
		free(xDataSet->y_test);
	}

	xDataSet->x_test = malloc(sizeof(float) * 40);
	xDataSet->y_test = malloc(sizeof(float));

	memcpy(xDataSet->x_test, &DataPtr[sizeof(uint16_t)+(num*(xNetParam.NonLayer[0])*sizeof(float))], (sizeof(float) * xNetParam.NonLayer[0]));
	memcpy(xDataSet->y_test, &DataPtr[sizeof(uint16_t) + (xDataSet->test_size * sizeof(float) * xNetParam.NonLayer[0]) + (num * (xNetParam.NonLayer[xNetParam.Layers - 1]) * sizeof(float))], (sizeof(float) * xNetParam.NonLayer[xNetParam.Layers - 1]));
}
