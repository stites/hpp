#include    <wb.h>

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2
#define TILE_WIDTH 16

//@@ INSERT CODE HERE
__global__ void convolution_2d (float *N, float *M, float *P,
                                int Channels, int Mask_Width, int Width){
  int bx  = blockIdx.x;
  int by  = blockIdx.y;
  int bdx = blockDim.x;
  int bdy = blockDim.y;
  int tx  = threadIdx.x;
  int ty  = threadIdx.y;
  int n   = Mask_Width / 2;
  int rIdx = by * bdx + ty;
  int cIdx = bx * bdy + tx;

  int halo_index_left = (bx - 1) * bdx + tx;
  int inputWidth      = TILE_WIDTH + Mask_width - 1;

  float pValue = 0.0;

  // share memory
  __shared__ float N_ds[TILE_WIDTH][TILE_WIDTH];
  if (tx >= bdx - n) {
    N_ds[rIdx][cIdx] = N[i];
  }
  __syncthreads();

  /*
  def clamp(x, start, end)
    return min(max(x, start), end)
  end
  for i from 0 to height do
    for j from 0 to width do
      for k from 0 to channels
        acc := 0
        for y from -maskRadius to maskRadius do
          for x from -maskRadius to maskRadius do
            xOffset := j + x
            yOffset := i + y
            if xOffset >= 0 && xOffset < width &&
               yOffset >= 0 && yOffset < height then
              imagePixel := I[(yOffset * width + xOffset) * channels + k]
              maskValue := K[(y+maskRadius)*maskWidth+x+maskRadius]
              acc += imagePixel * maskValue
            end
          end
        end
        # pixels are in the range of 0 to 1
        P[(i * width + j)*channels + k] = clamp(acc, 0, 1)
      end
    end
  */
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
    dim3 dimGrid(imageWidth/TILE_WIDTH, imageHeight/TILE_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    convolution_2D <<<dimGrid, dimBlock>>> (deviceInputImageData,
        deviceMaskData,
        deviceOutputImageData,
        imageChannels,
        imageWidth,
        imageHeight);
    // init kernel
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
