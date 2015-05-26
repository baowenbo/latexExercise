#include "YUV.SJTU_headers.h"
//#include "upscale.h"

#define  HF				(8.0f)
#define  a				(-0.50000000f)
__inline__ __device__ float cubic(float x){
	float abs_x = fabsf(x);

	if (abs_x <= 1.0f){
		return (a + 2)*abs_x *abs_x *abs_x - (a + 3)*abs_x *abs_x + 1;
	}
	else if (abs_x < 2.0f){
		return a * abs_x *abs_x *abs_x - 5 * a *abs_x *abs_x + 8 * a *abs_x - 4 * a;
	}
	else {
		return 0.0f;
	}
}
__inline__ __device__ float linear(float x){
	return x;
}


__global__ void DownSample(unsigned char * In, int iw, int ih,
	float * downbicu, int dw, int dh,
	float downscale_x, float downscale_y
	) {
	const int dstBaseX = blockIdx.x * blockDim.x + threadIdx.x;
	const int dstBaseY = blockIdx.y * blockDim.y + threadIdx.y;

	float downbicu_value = 0.0f;
	//float downbili_value = 0.0f;

	float srcY, srcX;
	int srcY0, srcX0; int srcY1, srcX1; int srcY2, srcX2; int srcY3, srcX3;


	float cubic_X[4], cubic_Y[4];
	//float linear_X[2], linear_Y[2];

	srcX = (dstBaseX + 1) / downscale_x + 0.5f*(1.0f - 1.0f / downscale_x) - 1;
	srcY = (dstBaseY + 1) / downscale_y + 0.5f*(1.0f - 1.0f / downscale_y) - 1;

	srcX0 = floorf(srcX) - 1;
	srcY0 = floorf(srcY) - 1;
	srcX1 = srcX0 + 1;
	srcY1 = srcY0 + 1;
	srcX2 = srcX0 + 2;
	srcY2 = srcY0 + 2;
	srcX3 = srcX0 + 3;
	srcY3 = srcY0 + 3;

	cubic_X[0] = cubic(srcX - srcX0);
	cubic_X[1] = cubic(srcX - srcX1);
	cubic_X[2] = cubic(srcX - srcX2);
	cubic_X[3] = cubic(srcX - srcX3);

	cubic_Y[0] = cubic(srcY - srcY0);
	cubic_Y[1] = cubic(srcY - srcY1);
	cubic_Y[2] = cubic(srcY - srcY2);
	cubic_Y[3] = cubic(srcY - srcY3);

	//linear_X[0] = linear(srcX2 - srcX);
	//linear_X[1] = linear(srcX - srcX1);

	//linear_Y[0] = linear(srcY2 - srcY);
	//linear_Y[1] = linear(srcY - srcY1);


	srcX0 = min(max(srcX0, 0), iw - 1);// (srcX0 < 0) ? 0 : srcX0;
	srcY0 = min(max(srcY0, 0), ih - 1);// (srcY0 < 0) ? 0 : srcY0;
	srcX1 = min(max(srcX1, 0), iw - 1);// (srcX1 < 0) ? 0 : srcX1;
	srcY1 = min(max(srcY1, 0), ih - 1);//(srcY1 < 0) ? 0 : srcY1;
	srcX2 = min(max(srcX2, 0), iw - 1);// (srcX2 > validW - 1) ? validW - 1 : srcX2;
	srcY2 = min(max(srcY2, 0), ih - 1);//(srcY2 > validH - 1) ? validH - 1 : srcY2;
	srcX3 = min(max(srcX3, 0), iw - 1);// (srcX3 > validW - 1) ? validW - 1 : srcX3;
	srcY3 = min(max(srcY3, 0), ih - 1);//(srcY3 > validH - 1) ? validH - 1 : srcY3;

	downbicu_value =
		cubic_X[0] * cubic_Y[0] * In[srcY0 * iw + srcX0] +
		cubic_X[0] * cubic_Y[1] * In[srcY1 * iw + srcX0] +
		cubic_X[0] * cubic_Y[2] * In[srcY2 * iw + srcX0] +
		cubic_X[0] * cubic_Y[3] * In[srcY3 * iw + srcX0] +
		cubic_X[1] * cubic_Y[0] * In[srcY0 * iw + srcX1] +
		cubic_X[1] * cubic_Y[1] * In[srcY1 * iw + srcX1] +
		cubic_X[1] * cubic_Y[2] * In[srcY2 * iw + srcX1] +
		cubic_X[1] * cubic_Y[3] * In[srcY3 * iw + srcX1] +
		cubic_X[2] * cubic_Y[0] * In[srcY0 * iw + srcX2] +
		cubic_X[2] * cubic_Y[1] * In[srcY1 * iw + srcX2] +
		cubic_X[2] * cubic_Y[2] * In[srcY2 * iw + srcX2] +
		cubic_X[2] * cubic_Y[3] * In[srcY3 * iw + srcX2] +
		cubic_X[3] * cubic_Y[0] * In[srcY0 * iw + srcX3] +
		cubic_X[3] * cubic_Y[1] * In[srcY1 * iw + srcX3] +
		cubic_X[3] * cubic_Y[2] * In[srcY2 * iw + srcX3] +
		cubic_X[3] * cubic_Y[3] * In[srcY3 * iw + srcX3];

	//downbili_value =
	//	linear_X[0] * linear_Y[0] * In[srcY1 * iw + srcX1] +
	//	linear_X[0] * linear_Y[1] * In[srcY2 * iw + srcX1] +
	//	linear_X[1] * linear_Y[0] * In[srcY1 * iw + srcX2] +
	//	linear_X[1] * linear_Y[1] * In[srcY2 * iw + srcX2];

	if (dstBaseY < dh && dstBaseX < dw){
		downbicu[dstBaseY * dw + dstBaseX] = downbicu_value;
		//downbili[dstBaseY * dw + dstBaseX] = downbili_value;
	}
}

__global__ void UpEnhanceLL(
	float * downbicu, int dw, int dh,
	unsigned char *In, unsigned char * Out, int ow, int oh,
	float upscale_x, float upscale_y
	){
	const int dstBaseX = blockIdx.x * blockDim.x + threadIdx.x;
	const int dstBaseY = blockIdx.y * blockDim.y + threadIdx.y;

	float upbicu_value = 0.0f;
	//float upbili_value = 0.0f;


	float srcY, srcX;

	int srcY0, srcX0; int srcY1, srcX1; int srcY2, srcX2; int srcY3, srcX3;

	float cubic_X[4], cubic_Y[4];
	//float linear_X[2], linear_Y[2];

	if (dstBaseX < ow / 2 - 1){
	// i just use the local coordinate instead of global coordinate....this may cause some block effect ....let's see
	srcX = (dstBaseX + 1) / upscale_x + 0.5f*(1.0f - 1.0f / upscale_x) - 1;
	srcY = (dstBaseY + 1) / upscale_y + 0.5f*(1.0f - 1.0f / upscale_y) - 1;

	srcX0 = floorf(srcX) - 1;
	srcY0 = floorf(srcY) - 1;
	srcX1 = srcX0 + 1;
	srcY1 = srcY0 + 1;
	srcX2 = srcX0 + 2;
	srcY2 = srcY0 + 2;
	srcX3 = srcX0 + 3;
	srcY3 = srcY0 + 3;

	cubic_X[0] = cubic(srcX - srcX0);
	cubic_X[1] = cubic(srcX - srcX1);
	cubic_X[2] = cubic(srcX - srcX2);
	cubic_X[3] = cubic(srcX - srcX3);

	cubic_Y[0] = cubic(srcY - srcY0);
	cubic_Y[1] = cubic(srcY - srcY1);
	cubic_Y[2] = cubic(srcY - srcY2);
	cubic_Y[3] = cubic(srcY - srcY3);

	//linear_X[0] = linear(srcX2 - srcX);
	//linear_X[1] = linear(srcX - srcX1);

	//linear_Y[0] = linear(srcY2 - srcY);
	//linear_Y[1] = linear(srcY - srcY1);

	srcX0 = min(max(srcX0, 0), dw - 1);// (srcX0 < 0) ? 0 : srcX0;
	srcY0 = min(max(srcY0, 0), dh - 1);// (srcY0 < 0) ? 0 : srcY0;
	srcX1 = min(max(srcX1, 0), dw - 1);// (srcX1 < 0) ? 0 : srcX1;
	srcY1 = min(max(srcY1, 0), dh - 1);//(srcY1 < 0) ? 0 : srcY1;
	srcX2 = min(max(srcX2, 0), dw - 1);// (srcX2 > validW - 1) ? validW - 1 : srcX2;
	srcY2 = min(max(srcY2, 0), dh - 1);//(srcY2 > validH - 1) ? validH - 1 : srcY2;
	srcX3 = min(max(srcX3, 0), dw - 1);// (srcX3 > validW - 1) ? validW - 1 : srcX3;
	srcY3 = min(max(srcY3, 0), dh - 1);//(srcY3 > validH - 1) ? validH - 1 : srcY3;


	upbicu_value =
		cubic_X[0] * cubic_Y[0] * downbicu[srcY0 * dw + srcX0] +
		cubic_X[0] * cubic_Y[1] * downbicu[srcY1 * dw + srcX0] +
		cubic_X[0] * cubic_Y[2] * downbicu[srcY2 * dw + srcX0] +
		cubic_X[0] * cubic_Y[3] * downbicu[srcY3 * dw + srcX0] +
		cubic_X[1] * cubic_Y[0] * downbicu[srcY0 * dw + srcX1] +
		cubic_X[1] * cubic_Y[1] * downbicu[srcY1 * dw + srcX1] +
		cubic_X[1] * cubic_Y[2] * downbicu[srcY2 * dw + srcX1] +
		cubic_X[1] * cubic_Y[3] * downbicu[srcY3 * dw + srcX1] +
		cubic_X[2] * cubic_Y[0] * downbicu[srcY0 * dw + srcX2] +
		cubic_X[2] * cubic_Y[1] * downbicu[srcY1 * dw + srcX2] +
		cubic_X[2] * cubic_Y[2] * downbicu[srcY2 * dw + srcX2] +
		cubic_X[2] * cubic_Y[3] * downbicu[srcY3 * dw + srcX2] +
		cubic_X[3] * cubic_Y[0] * downbicu[srcY0 * dw + srcX3] +
		cubic_X[3] * cubic_Y[1] * downbicu[srcY1 * dw + srcX3] +
		cubic_X[3] * cubic_Y[2] * downbicu[srcY2 * dw + srcX3] +
		cubic_X[3] * cubic_Y[3] * downbicu[srcY3 * dw + srcX3];

	//upbili_value =
	//	linear_X[0] * linear_Y[0] * downbili[srcY1 * dw + srcX1] +
	//	linear_X[0] * linear_Y[1] * downbili[srcY2 * dw + srcX1] +
	//	linear_X[1] * linear_Y[0] * downbili[srcY1 * dw + srcX2] +
	//	linear_X[1] * linear_Y[1] * downbili[srcY2 * dw + srcX2];


	upbicu_value = In[dstBaseY * ow + dstBaseX] + HF*(In[dstBaseY * ow + dstBaseX] - upbicu_value);
	}
	else if (dstBaseX == ow / 2 - 1 || dstBaseX == ow / 2){
		upbicu_value = 255;
	}
	else{
		upbicu_value = In[dstBaseY * ow + dstBaseX - ow / 2];
	}

	if (dstBaseY < oh && dstBaseX < ow){
		Out[dstBaseY * ow + dstBaseX] = min(max(unsigned int(upbicu_value), 0), 255);
	}
}

void OptBicu_BicuexeckernelLL(unsigned char * In, int iw, int ih,
	float * downbicu, int dw, int dh, float downscale_x, float downscale_y,
	unsigned char * Out, int ow, int oh, float upscale_x, float upscale_y
	){

	static dim3 grid;
	static dim3 block;

	block.x = 32;
	block.y = 32;
	block.z = 1;
	grid.x = (dw + block.x - 1) / block.x;
	grid.y = (dh + block.y - 1) / block.y;
	grid.z = 1;

	DownSample << <grid, block >> >(In, iw, ih, downbicu, dw, dh, downscale_x, downscale_y);
	
	grid.x = (ow + block.x - 1) / block.x;
	grid.y = (oh + block.y - 1) / block.y;
	grid.z = 1;


	//Sheme Four:
		//left side ==>  I + 4 *(I - upbicu(downbicu(I)))
		//right side ==> I
	UpEnhanceLL << <grid, block >> >(downbicu, dw, dh,In, Out, ow, oh, upscale_x, upscale_y);


}
 


__global__ void UpEnhanceLR(
	float * downbicu, int dw, int dh,
	unsigned char *In, unsigned char * Out, int ow, int oh,
	float upscale_x, float upscale_y
	){
	const int dstBaseX = blockIdx.x * blockDim.x + threadIdx.x;
	const int dstBaseY = blockIdx.y * blockDim.y + threadIdx.y;

	float upbicu_value = 0.0f;
	//float upbili_value = 0.0f;


	float srcY, srcX;

	int srcY0, srcX0; int srcY1, srcX1; int srcY2, srcX2; int srcY3, srcX3;

	float cubic_X[4], cubic_Y[4];
	//float linear_X[2], linear_Y[2];


	// i just use the local coordinate instead of global coordinate....this may cause some block effect ....let's see
	srcX = (dstBaseX + 1) / upscale_x + 0.5f*(1.0f - 1.0f / upscale_x) - 1;
	srcY = (dstBaseY + 1) / upscale_y + 0.5f*(1.0f - 1.0f / upscale_y) - 1;

	srcX0 = floorf(srcX) - 1;
	srcY0 = floorf(srcY) - 1;
	srcX1 = srcX0 + 1;
	srcY1 = srcY0 + 1;
	srcX2 = srcX0 + 2;
	srcY2 = srcY0 + 2;
	srcX3 = srcX0 + 3;
	srcY3 = srcY0 + 3;

	cubic_X[0] = cubic(srcX - srcX0);
	cubic_X[1] = cubic(srcX - srcX1);
	cubic_X[2] = cubic(srcX - srcX2);
	cubic_X[3] = cubic(srcX - srcX3);

	cubic_Y[0] = cubic(srcY - srcY0);
	cubic_Y[1] = cubic(srcY - srcY1);
	cubic_Y[2] = cubic(srcY - srcY2);
	cubic_Y[3] = cubic(srcY - srcY3);

	//linear_X[0] = linear(srcX2 - srcX);
	//linear_X[1] = linear(srcX - srcX1);

	//linear_Y[0] = linear(srcY2 - srcY);
	//linear_Y[1] = linear(srcY - srcY1);

	srcX0 = min(max(srcX0, 0), dw - 1);// (srcX0 < 0) ? 0 : srcX0;
	srcY0 = min(max(srcY0, 0), dh - 1);// (srcY0 < 0) ? 0 : srcY0;
	srcX1 = min(max(srcX1, 0), dw - 1);// (srcX1 < 0) ? 0 : srcX1;
	srcY1 = min(max(srcY1, 0), dh - 1);//(srcY1 < 0) ? 0 : srcY1;
	srcX2 = min(max(srcX2, 0), dw - 1);// (srcX2 > validW - 1) ? validW - 1 : srcX2;
	srcY2 = min(max(srcY2, 0), dh - 1);//(srcY2 > validH - 1) ? validH - 1 : srcY2;
	srcX3 = min(max(srcX3, 0), dw - 1);// (srcX3 > validW - 1) ? validW - 1 : srcX3;
	srcY3 = min(max(srcY3, 0), dh - 1);//(srcY3 > validH - 1) ? validH - 1 : srcY3;


	upbicu_value =
		cubic_X[0] * cubic_Y[0] * downbicu[srcY0 * dw + srcX0] +
		cubic_X[0] * cubic_Y[1] * downbicu[srcY1 * dw + srcX0] +
		cubic_X[0] * cubic_Y[2] * downbicu[srcY2 * dw + srcX0] +
		cubic_X[0] * cubic_Y[3] * downbicu[srcY3 * dw + srcX0] +
		cubic_X[1] * cubic_Y[0] * downbicu[srcY0 * dw + srcX1] +
		cubic_X[1] * cubic_Y[1] * downbicu[srcY1 * dw + srcX1] +
		cubic_X[1] * cubic_Y[2] * downbicu[srcY2 * dw + srcX1] +
		cubic_X[1] * cubic_Y[3] * downbicu[srcY3 * dw + srcX1] +
		cubic_X[2] * cubic_Y[0] * downbicu[srcY0 * dw + srcX2] +
		cubic_X[2] * cubic_Y[1] * downbicu[srcY1 * dw + srcX2] +
		cubic_X[2] * cubic_Y[2] * downbicu[srcY2 * dw + srcX2] +
		cubic_X[2] * cubic_Y[3] * downbicu[srcY3 * dw + srcX2] +
		cubic_X[3] * cubic_Y[0] * downbicu[srcY0 * dw + srcX3] +
		cubic_X[3] * cubic_Y[1] * downbicu[srcY1 * dw + srcX3] +
		cubic_X[3] * cubic_Y[2] * downbicu[srcY2 * dw + srcX3] +
		cubic_X[3] * cubic_Y[3] * downbicu[srcY3 * dw + srcX3];

	//upbili_value =
	//	linear_X[0] * linear_Y[0] * downbili[srcY1 * dw + srcX1] +
	//	linear_X[0] * linear_Y[1] * downbili[srcY2 * dw + srcX1] +
	//	linear_X[1] * linear_Y[0] * downbili[srcY1 * dw + srcX2] +
	//	linear_X[1] * linear_Y[1] * downbili[srcY2 * dw + srcX2];

	if (dstBaseX < ow / 2 - 1){
		upbicu_value = In[dstBaseY * ow + dstBaseX] + HF*(In[dstBaseY * ow + dstBaseX] - upbicu_value);
	}
	else if (dstBaseX == ow / 2 - 1 || dstBaseX == ow / 2){
		upbicu_value = 255;
	}
	else{
		upbicu_value = In[dstBaseY * ow + dstBaseX];
	}

	if (dstBaseY < oh && dstBaseX < ow){
		Out[dstBaseY * ow + dstBaseX] = min(max(unsigned int(upbicu_value), 0), 255);
	}
}


void OptBicu_BicuexeckernelLR(unsigned char * In, int iw, int ih,
	float * downbicu, int dw, int dh, float downscale_x, float downscale_y,
	unsigned char * Out, int ow, int oh, float upscale_x, float upscale_y
	){

	static dim3 grid;
	static dim3 block;

	block.x = 32;
	block.y = 32;
	block.z = 1;
	grid.x = (dw + block.x - 1) / block.x;
	grid.y = (dh + block.y - 1) / block.y;
	grid.z = 1;

	DownSample << <grid, block >> >(In, iw, ih, downbicu, dw, dh, downscale_x, downscale_y);

	grid.x = (ow + block.x - 1) / block.x;
	grid.y = (oh + block.y - 1) / block.y;
	grid.z = 1;


	//Sheme Four:
	//left side ==>  I + 4 *(I - upbicu(downbicu(I)))
	//right side ==> I
	UpEnhanceLR << <grid, block >> >(downbicu, dw, dh, In, Out, ow, oh, upscale_x, upscale_y);


}

__global__ void UpEnhance(
	float * downbicu, int dw, int dh,
	unsigned char *In, unsigned char * Out, int ow, int oh,
	float upscale_x, float upscale_y
	){
	const int dstBaseX = blockIdx.x * blockDim.x + threadIdx.x;
	const int dstBaseY = blockIdx.y * blockDim.y + threadIdx.y;

	float upbicu_value = 0.0f;
	//float upbili_value = 0.0f;


	float srcY, srcX;

	int srcY0, srcX0; int srcY1, srcX1; int srcY2, srcX2; int srcY3, srcX3;

	float cubic_X[4], cubic_Y[4];
	//float linear_X[2], linear_Y[2];


	// i just use the local coordinate instead of global coordinate....this may cause some block effect ....let's see
	srcX = (dstBaseX + 1) / upscale_x + 0.5f*(1.0f - 1.0f / upscale_x) - 1;
	srcY = (dstBaseY + 1) / upscale_y + 0.5f*(1.0f - 1.0f / upscale_y) - 1;

	srcX0 = floorf(srcX) - 1;
	srcY0 = floorf(srcY) - 1;
	srcX1 = srcX0 + 1;
	srcY1 = srcY0 + 1;
	srcX2 = srcX0 + 2;
	srcY2 = srcY0 + 2;
	srcX3 = srcX0 + 3;
	srcY3 = srcY0 + 3;

	cubic_X[0] = cubic(srcX - srcX0);
	cubic_X[1] = cubic(srcX - srcX1);
	cubic_X[2] = cubic(srcX - srcX2);
	cubic_X[3] = cubic(srcX - srcX3);

	cubic_Y[0] = cubic(srcY - srcY0);
	cubic_Y[1] = cubic(srcY - srcY1);
	cubic_Y[2] = cubic(srcY - srcY2);
	cubic_Y[3] = cubic(srcY - srcY3);

	//linear_X[0] = linear(srcX2 - srcX);
	//linear_X[1] = linear(srcX - srcX1);

	//linear_Y[0] = linear(srcY2 - srcY);
	//linear_Y[1] = linear(srcY - srcY1);

	srcX0 = min(max(srcX0, 0), dw - 1);// (srcX0 < 0) ? 0 : srcX0;
	srcY0 = min(max(srcY0, 0), dh - 1);// (srcY0 < 0) ? 0 : srcY0;
	srcX1 = min(max(srcX1, 0), dw - 1);// (srcX1 < 0) ? 0 : srcX1;
	srcY1 = min(max(srcY1, 0), dh - 1);//(srcY1 < 0) ? 0 : srcY1;
	srcX2 = min(max(srcX2, 0), dw - 1);// (srcX2 > validW - 1) ? validW - 1 : srcX2;
	srcY2 = min(max(srcY2, 0), dh - 1);//(srcY2 > validH - 1) ? validH - 1 : srcY2;
	srcX3 = min(max(srcX3, 0), dw - 1);// (srcX3 > validW - 1) ? validW - 1 : srcX3;
	srcY3 = min(max(srcY3, 0), dh - 1);//(srcY3 > validH - 1) ? validH - 1 : srcY3;


	upbicu_value =
		cubic_X[0] * cubic_Y[0] * downbicu[srcY0 * dw + srcX0] +
		cubic_X[0] * cubic_Y[1] * downbicu[srcY1 * dw + srcX0] +
		cubic_X[0] * cubic_Y[2] * downbicu[srcY2 * dw + srcX0] +
		cubic_X[0] * cubic_Y[3] * downbicu[srcY3 * dw + srcX0] +
		cubic_X[1] * cubic_Y[0] * downbicu[srcY0 * dw + srcX1] +
		cubic_X[1] * cubic_Y[1] * downbicu[srcY1 * dw + srcX1] +
		cubic_X[1] * cubic_Y[2] * downbicu[srcY2 * dw + srcX1] +
		cubic_X[1] * cubic_Y[3] * downbicu[srcY3 * dw + srcX1] +
		cubic_X[2] * cubic_Y[0] * downbicu[srcY0 * dw + srcX2] +
		cubic_X[2] * cubic_Y[1] * downbicu[srcY1 * dw + srcX2] +
		cubic_X[2] * cubic_Y[2] * downbicu[srcY2 * dw + srcX2] +
		cubic_X[2] * cubic_Y[3] * downbicu[srcY3 * dw + srcX2] +
		cubic_X[3] * cubic_Y[0] * downbicu[srcY0 * dw + srcX3] +
		cubic_X[3] * cubic_Y[1] * downbicu[srcY1 * dw + srcX3] +
		cubic_X[3] * cubic_Y[2] * downbicu[srcY2 * dw + srcX3] +
		cubic_X[3] * cubic_Y[3] * downbicu[srcY3 * dw + srcX3];

	//upbili_value =
	//	linear_X[0] * linear_Y[0] * downbili[srcY1 * dw + srcX1] +
	//	linear_X[0] * linear_Y[1] * downbili[srcY2 * dw + srcX1] +
	//	linear_X[1] * linear_Y[0] * downbili[srcY1 * dw + srcX2] +
	//	linear_X[1] * linear_Y[1] * downbili[srcY2 * dw + srcX2];

	upbicu_value = In[dstBaseY * ow + dstBaseX] + HF*(In[dstBaseY * ow + dstBaseX] - upbicu_value);

	if (dstBaseY < oh && dstBaseX < ow){
		Out[dstBaseY * ow + dstBaseX] = min(max(unsigned int(upbicu_value), 0), 255);
	}
}


void OptBicu_Bicuexeckernel(unsigned char * In, int iw, int ih,
	float * downbicu, int dw, int dh, float downscale_x, float downscale_y,
	unsigned char * Out, int ow, int oh, float upscale_x, float upscale_y
	){

	static dim3 grid;
	static dim3 block;

	block.x = 32;
	block.y = 32;
	block.z = 1;
	grid.x = (dw + block.x - 1) / block.x;
	grid.y = (dh + block.y - 1) / block.y;
	grid.z = 1;

	DownSample << <grid, block >> >(In, iw, ih, downbicu, dw, dh, downscale_x, downscale_y);

	grid.x = (ow + block.x - 1) / block.x;
	grid.y = (oh + block.y - 1) / block.y;
	grid.z = 1;


	//Sheme Four:
	//left side ==>  I + 4 *(I - upbicu(downbicu(I)))
	//right side ==> I
	UpEnhance << <grid, block >> >(downbicu, dw, dh, In, Out, ow, oh, upscale_x, upscale_y);
	printf("Here I anm\n\year");

}
