#include <stdio.h>
#include <cuda_runtime.h>

#define N 10
#define M 10
#define THREADS_PER_BLOCK 10
#define BLOCKS  (THREADS_PER_BLOCK + (N*M-1))/THREADS_PER_BLOCK

__device__ void do_operations(int r, int c, int *d_input, int *d_filter, int *d_conv, int input_rows, int input_cols, int filter_size, int conv_rows, int conv_cols) {
   int conv_idx = r * conv_rows + c;
   d_conv[conv_idx] = 0;
   for(int i = 0; i < filter_size; i++) {
      for(int j = 0; j < filter_size; j++) {
         d_conv[conv_idx] +=
            d_input[(r + i) * input_rows + (c + j)] *
            d_filter[(i * filter_size) + j];
      }
   }
}

__global__ void convolution(int *d_input, int *d_filter, int *d_conv, int input_rows, int input_cols, int filter_size, int conv_rows, int conv_cols) {
   // Get the coordinates based on the threads status
   int conv_col = blockIdx.x * blockDim.x + threadIdx.x;
   int conv_row = blockIdx.y * blockDim.y + threadIdx.y;

   if(conv_col < conv_cols && conv_row < conv_rows) {
      do_operations(conv_row, conv_col, d_input, d_filter, d_conv, input_rows, input_cols, filter_size, conv_rows, conv_cols);
   }
}

void fill_matrix(int *m, int r, int c) {
   int idx;
   for(int i = 0; i < r; i++) {
      for(int j = 0; j < c; j++) {
         idx = i * r + j;
         m[idx] = idx;
      }
   }
}

void print_matrix(int *m, int r, int c) {
   int idx;
   for(int i = 0; i < r; i++) {
      for(int j = 0; j < c; j++) {
         idx = i * r + j;
         printf("%d\t", m[idx]);
      }
      printf("\n");
   }
}

int main() {
   // Matrices in the host
   int *input, *filter, *conv;
   // Matrices in the device
   int *d_input, *d_filter, *d_conv;

   // Size of the matrices
   int input_rows = N;
   int input_cols = M;
   int filter_size = 5;
   int conv_rows = input_rows - (filter_size/2)*2;
   int conv_cols = input_cols - (filter_size/2)*2;

   // Allocate memory on the host
   input = (int*) malloc(input_rows * input_cols * sizeof(int));
   filter = (int*) malloc(filter_size * filter_size * sizeof(int));
   conv = (int*) malloc(conv_rows * conv_cols * sizeof(int));

   // Fill the input and filter matrices
   fill_matrix(input, input_rows, input_cols);
   fill_matrix(filter, filter_size, filter_size);

   // Allocate memory on the device
   cudaMalloc((void**)&d_input, input_rows * input_cols * sizeof(int));
   cudaMalloc((void**)&d_filter, filter_size * filter_size * sizeof(int));
   cudaMalloc((void**)&d_conv, conv_rows * conv_cols * sizeof(int));

   // Copy data from host to device
   cudaMemcpy(d_input, input, input_rows * input_cols * sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(d_filter, filter, filter_size * filter_size * sizeof(int), cudaMemcpyHostToDevice);

   // Create the grid of threads that will be used
   dim3 Threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
   dim3 Blocks(BLOCKS, BLOCKS);

   // Invoke the function to be executed on the device
   convolution <<<Blocks, Threads>>>(d_input, d_filter, d_conv, input_rows, input_cols, filter_size, conv_rows, conv_cols);

   // Copy the result from device to host
   cudaMemcpy(conv, d_conv, conv_rows * conv_cols * sizeof(int), cudaMemcpyDeviceToHost);

   // Display the result
   printf("Input matrix\n");
   print_matrix(input, input_rows, input_cols);
   printf("Filter\n");
   print_matrix(filter, filter_size, filter_size);
   printf("Convolutioned matrix\n");
   print_matrix(conv, conv_rows, conv_cols);

   // Free the memory
   cudaFree(d_input);
   cudaFree(d_filter);
   cudaFree(d_conv);

   free(input);
   free(filter);
   free(conv);

   return 0;
}