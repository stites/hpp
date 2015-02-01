#include    <wb.h>

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_Width 5
#define Mask_Radius Mask_Width/2
#define Tile_Width 16
#define Output_Width Tile_Width - Mask_Width + 1
#define Channels 3

//@@ INSERT CODE HERE
__device__ float clamp(float x, float start, float end){
  return min(max(x, start), end);
}

__global__ void convolution_2D(float * inputImageData,
                               float * outputImageData,
                               const float * __restrict__ Mask,
                               int imageWidth,
                               int imageHeight) {
  // add short hand notation - maybe remove this later
  int bx  = blockIdx.x;
  int by  = blockIdx.y;
  int bdx = blockDim.x;
  int bdy = blockDim.y;
  int tx  = threadIdx.x;
  int ty  = threadIdx.y;
  int rowIdx_o = by * Output_Width + ty;
  int colIdx_o = bx * Output_Width + tx;
  int rowIdx_i = rowIdx_o - Mask_Radius;
  int colIdx_i = colIdx_o - Mask_Radius;

  float output[Channels] = {0.0f, 0.0f, 0.0f};

  // allocate shared memory
  __shared__ float ds_input[Tile_Width][Tile_Width][Channels];

  // load the cache - check halo conditions
  if((rowIdx_i >= 0) && (rowIdx_i < imageHeight) && (colIdx_i >= 0) && (colIdx_i < imageWidth) ) {
    for (int k = 0; k < Channels; k++)
      ds_input[ty][tx][k] = inputImageData[(rowIdx_i * imageWidth + colIdx_i) * Channels + k];
  } else{
    for (int k = 0; k < Channels; k++)
      ds_input[ty][tx][k] = 0.0f;
  }

  // sync before moving on
  __syncthreads();

  // begin convolution logic:
  if((ty < Output_Width) && (tx < Output_Width)){
    for(int i = 0; i < Mask_Width; i++) {
      for(int j = 0; j < Mask_Width; j++) {
        for (int k = 0; k < Channels; k++)
          output[k] += Mask[i * Mask_Width + j] * ds_input[i+ty][j+tx][k];
      }
    }

    __syncthreads();

    if(rowIdx_o < imageHeight && colIdx_o < imageWidth){
      for (int k = 0; k < Channels; k++)
        outputImageData[(rowIdx_o * imageWidth + colIdx_o) * Channels + k] = clamp(output[k], 0, 1);
    }
  }
}

int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
    dim3 dimGrid((imageWidth-1)/Output_Width + 1, (imageHeight-1)/Output_Width + 1, 1);
    dim3 dimBlock(Tile_Width, Tile_Width, 1);

    convolution_2D<<<dimGrid, dimBlock>>>(deviceInputImageData,
                                          deviceOutputImageData,
                                          deviceMaskData,
                                          imageWidth,
                                          imageHeight);

    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
