#include <mpi.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

#include <iostream>
#include <sstream>

__global__ void update(float* out)
{
	size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
	out[xid] = 69.0f;
};

int main ( int argc, char** argv )
{
	MPI_Init(NULL, NULL);

	int rankCount;
    MPI_Comm_size(MPI_COMM_WORLD, &rankCount);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    char processor_name[200];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    
    int gpuCount;
    cudaGetDeviceCount(&gpuCount);

    cudaSetDevice(4 - 4*rank);
    void* d_sendPtr;
	checkCudaErrors(cudaMalloc(&d_sendPtr, 4));
    void* d_recvPtr;
	checkCudaErrors(cudaMalloc(&d_recvPtr, 4));
    
    int32_t value = rank + 1;
    checkCudaErrors(cudaMemcpy(d_sendPtr, &value, 4, cudaMemcpyHostToDevice)); 
	
    cudaSetDevice(0);

	MPI_Send(d_sendPtr, 1, MPI_INT, (rank + 1) % 2, 0, MPI_COMM_WORLD);
	MPI_Recv(d_recvPtr, 1, MPI_INT, (rank + 1) % 2, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

	MPI_Send(d_sendPtr, 1, MPI_INT, (rank + 1) % 2, 0, MPI_COMM_WORLD);
	MPI_Recv(d_recvPtr, 1, MPI_INT, (rank + 1) % 2, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

    int32_t recvValue = 0;
    checkCudaErrors(cudaMemcpy(&recvValue, d_recvPtr, 4, cudaMemcpyDeviceToHost));
	
	std::cout << "Rank " << rank + 1 << "/" << rankCount << " is running on processor " << processor_name << " and it has " << gpuCount << " GPUs and reveived a message " << recvValue << std::endl;
	
	MPI_Finalize();

	return 0;
}
