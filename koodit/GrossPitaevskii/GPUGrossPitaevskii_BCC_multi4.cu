#include <cuda_runtime.h>
#include "helper_cuda.h"

#include "VortexState.hpp"
#include "../../jukan_koodit/Output/Picture.hpp"
#include "../../jukan_koodit/Output/Text.hpp"
#include "../../jukan_koodit/Types/Complex.hpp"
#include "../../jukan_koodit/Types/Random.hpp"
#include "../../jukan_koodit/Mesh/DelaunayMesh.hpp"
#include <iostream>
#include <sstream>

ddouble RATIO = 0.1;
ddouble KAPPA = 20;
ddouble G = 5000;

#define LOAD_STATE_FROM_DISK 1
#define SAVE_PICTURE 0
#define SAVE_VOLUME 1

#define THREAD_BLOCK_X 8
#define THREAD_BLOCK_Y 8
#define THREAD_BLOCK_Z 1
#define FACE_COUNT 4 // Primary faces
#define VALUES_IN_BLOCK 12 // Dual nodes
#define INDICES_PER_BLOCK 48

ddouble potentialRZ(const ddouble r, const ddouble z)
{
	return 0.5 * (r * r + RATIO * RATIO * z * z);
}

ddouble potentialV3(const Vector3 &p)
{
	return 0.5 * (p.x * p.x + p.y * p.y + RATIO * RATIO * p.z * p.z);
}

bool saveVolumeMap(const std::string &path, const Buffer<ushort> &vol, const uint xsize, const uint ysize, const uint zsize, const Vector3 &h)
{
	Text rawpath;
	rawpath << path << ".raw";

	// save raw
	std::ofstream fs(rawpath.str().c_str(), std::ios_base::binary | std::ios::trunc);
	if(fs.fail()) return false;
	fs.write((char*)&vol[0], 2 * xsize * ysize * zsize);
	fs.close();

	// save header
	Text text;

	text <<	"ObjectType              = Image" << std::endl;
	text <<	"NDims                   = 3" << std::endl;
	text <<	"BinaryData              = True" << std::endl;
	text <<	"CompressedData          = False" << std::endl;
	text <<	"BinaryDataByteOrderMSB  = False" << std::endl;
	text <<	"TransformMatrix         = 1 0 0 0 1 0 0 0 1" << std::endl;
	text <<	"Offset                  = " << -0.5 * xsize * h.x << " " << -0.5 * ysize * h.y << " " << -0.5 * zsize * h.z << std::endl;
	text <<	"CenterOfRotation        = 0 0 0" << std::endl;
	text <<	"DimSize                 = " << xsize << " " << ysize << " " << zsize << std::endl;
	text <<	"ElementSpacing          = " << h.x << " " << h.y << " " << h.z << std::endl;
	text <<	"ElementNumberOfChannels = 1" << std::endl;
	text <<	"ElementType             = MET_USHORT" << std::endl;
	text <<	"ElementDataFile         = " << rawpath.str() << std::endl;
	text.save(path);
	return true;
}

// bcc grid
const Vector3 BLOCK_WIDTH = sqrt(8.0) * Vector3(1, 1, 1); // dimensions of unit block
const ddouble VOLUME = sqrt(32.0 / 9.0); // volume of body elements
const bool IS_3D = true; // 3-dimensional
void getPositions(Buffer<Vector3> &pos)
{
	pos.resize(12);
	const ddouble SQ1_8 = 1.0 / sqrt(8.0);
	pos[0] = SQ1_8 * Vector3(1, 3, 7);
	pos[1] = SQ1_8 * Vector3(7, 1, 3);
	pos[2] = SQ1_8 * Vector3(7, 3, 1);
	pos[3] = SQ1_8 * Vector3(7, 3, 5);
	pos[4] = SQ1_8 * Vector3(7, 5, 3);
	pos[5] = SQ1_8 * Vector3(1, 7, 3);
	pos[6] = SQ1_8 * Vector3(3, 7, 1);
	pos[7] = SQ1_8 * Vector3(3, 7, 5);
	pos[8] = SQ1_8 * Vector3(5, 7, 3);
	pos[9] = SQ1_8 * Vector3(3, 1, 7);
	pos[10] = SQ1_8 * Vector3(3, 5, 7);
	pos[11] = SQ1_8 * Vector3(5, 3, 7);
}
ddouble getLaplacian(Buffer<int2> &ind, const int nx, const int ny, const int nz) // nx, ny, nz in bytes
{
	ind.resize(48);
    // Primary faces of the 1. dual node
	ind[0] = make_int2(0, 9);
	ind[1] = make_int2(0, 10);
	ind[2] = make_int2(nz - nx, 2);
	ind[3] = make_int2(-nx, 3);

    // Primary faces of the 2. dual node
	ind[4] = make_int2(0, 2);
	ind[5] = make_int2(0, 3);
	ind[6] = make_int2(nx - ny, 5);
	ind[7] = make_int2(-ny, 8);

    // Primary faces of the 3. dual node
	ind[8] = make_int2(0, 1);
	ind[9] = make_int2(0, 4);
	ind[10] = make_int2(-nz + nx, 0);
	ind[11] = make_int2(-nz, 11);

	// Primary faces of the 4. dual node
    ind[12] = make_int2(0, 1);
	ind[13] = make_int2(0, 4);
	ind[14] = make_int2(nx, 0);
	ind[15] = make_int2(0, 11);

    // Primary faces of the 5. dual node
	ind[16] = make_int2(0, 2);
	ind[17] = make_int2(0, 3);
	ind[18] = make_int2(nx, 5);
	ind[19] = make_int2(0, 8);

    // Primary faces of the 6. dual node
	ind[20] = make_int2(0, 6);
	ind[21] = make_int2(0, 7);
	ind[22] = make_int2(-nx + ny, 1);
	ind[23] = make_int2(-nx, 4);

    // Primary faces of the 7. dual node
	ind[24] = make_int2(0, 5);
	ind[25] = make_int2(0, 8);
	ind[26] = make_int2(-nz + ny, 9);
	ind[27] = make_int2(-nz, 10);

    // Primary faces of the 8. dual node
	ind[28] = make_int2(0, 5);
	ind[29] = make_int2(0, 8);
	ind[30] = make_int2(ny, 9);
	ind[31] = make_int2(0, 10);

    // Primary faces of the 9. dual node
	ind[32] = make_int2(0, 6);
	ind[33] = make_int2(0, 7);
	ind[34] = make_int2(ny, 1);
	ind[35] = make_int2(0, 4);

    // Primary faces of the 10. dual node
	ind[36] = make_int2(0, 0);
	ind[37] = make_int2(0, 11);
	ind[38] = make_int2(nz - ny, 6);
	ind[39] = make_int2(-ny, 7);

    // Primary faces of the 11. dual node
	ind[40] = make_int2(0, 0);
	ind[41] = make_int2(0, 11);
	ind[42] = make_int2(nz, 6);
	ind[43] = make_int2(0, 7);

    // Primary faces of the 12. dual node
	ind[44] = make_int2(0, 9);
	ind[45] = make_int2(0, 10);
	ind[46] = make_int2(nz, 2);
	ind[47] = make_int2(0, 3);

	return 1.5;
}

struct BlockPsis
{
	double2 values[VALUES_IN_BLOCK]; 
};

struct BlockPots
{
	double values[VALUES_IN_BLOCK];
};

struct PitchedPtr
{
	char* ptr;
	size_t pitch;
	size_t slicePitch;
};

// Arithmetic operators for cuda vector types
inline __host__ __device__ double2 operator+(double2 a, double2 b)
{
    return make_double2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(double2 &a, double2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ double2 operator*(double b, double2 a)
{
    return make_double2(b * a.x, b * a.y);
}

__global__ void update(PitchedPtr nextStep, PitchedPtr prevStep, PitchedPtr potentials, int2* lapInd, double2 lapfacs, double g, uint3 dimensions)
{
	size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t yid = blockIdx.y * blockDim.y + threadIdx.y;
	size_t zid = blockIdx.z * blockDim.z + threadIdx.z;

	// Load Laplacian indices into LDS
	__shared__ int2 ldsLapInd[INDICES_PER_BLOCK];
	size_t threadIdxInBlock = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
	if (threadIdxInBlock < INDICES_PER_BLOCK)
	{
		ldsLapInd[threadIdxInBlock] = lapInd[threadIdxInBlock];
	}
	__syncthreads();

	size_t dataZid = zid / VALUES_IN_BLOCK; // One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on z-axis)

	// Exit leftover threads
	if (xid >= dimensions.x || yid >= dimensions.y || dataZid >= dimensions.z)
	{
		return;
	}

	// Calculate the pointers for this block
	char* prevPsi = prevStep.ptr + prevStep.slicePitch * dataZid + prevStep.pitch * yid + sizeof(BlockPsis) * xid;
	BlockPsis* nextPsi = (BlockPsis*)(nextStep.ptr + nextStep.slicePitch * dataZid + nextStep.pitch * yid) + xid;
	BlockPots* pot = (BlockPots*)(potentials.ptr + potentials.slicePitch * dataZid + potentials.pitch * yid) + xid;

	// Update psi
	size_t dualNodeId = zid % VALUES_IN_BLOCK; // Dual node id. One thread per every dual node so VALUES_IN_BLOCK threads per mesh block (on z-axis)

    // 4 primary faces
    uint face = dualNodeId * FACE_COUNT;
	double2 sum =  (*(BlockPsis*)(prevPsi + ldsLapInd[face].x)).values[ldsLapInd[face++].y];
	        sum += (*(BlockPsis*)(prevPsi + ldsLapInd[face].x)).values[ldsLapInd[face++].y];
	        sum += (*(BlockPsis*)(prevPsi + ldsLapInd[face].x)).values[ldsLapInd[face++].y];
	        sum += (*(BlockPsis*)(prevPsi + ldsLapInd[face].x)).values[ldsLapInd[face++].y];

	double2 prev = (*(BlockPsis*)prevPsi).values[dualNodeId];
	double normsq = prev.x * prev.x + prev.y * prev.y;
	sum = lapfacs.x * sum + (lapfacs.y + (*pot).values[dualNodeId] + g * normsq) * prev;

	(*nextPsi).values[dualNodeId] += make_double2(sum.y, -sum.x);
};

uint integrateInTime(const VortexState &state, const ddouble block_scale, const Vector3 &minp, const Vector3 &maxp, const ddouble iteration_period, const uint number_of_iterations)
{	
	uint i, j, k, l;

	// find dimensions
	const Vector3 domain = maxp - minp;
	const uint xsize = uint(domain.x / (block_scale * BLOCK_WIDTH.x)) + 1;
	const uint ysize = uint(domain.y / (block_scale * BLOCK_WIDTH.y)) + 1;
	const uint zsize = uint(domain.z / (block_scale * BLOCK_WIDTH.z)) + 1;
	const Vector3 p0 = 0.5 * (minp + maxp - block_scale * Vector3(BLOCK_WIDTH.x * xsize, BLOCK_WIDTH.y * ysize, BLOCK_WIDTH.z * zsize));

	const uint zsize_first_half = zsize / 2 + zsize % 2;
	const uint zsize_second_half = zsize / 2;
	const uint zsize_d0 = zsize_first_half / 2 + zsize_first_half % 2;
	const uint zsize_d1 = zsize_first_half / 2;
	const uint zsize_d2 = zsize_second_half / 2 + zsize_second_half % 2;
	const uint zsize_d3 = zsize_second_half / 2;

	//std::cout << xsize << ", " << ysize << ", " << zsize << std::endl;

	// find relative circumcenters for each body element
	Buffer<Vector3> bpos;
	getPositions(bpos);

	// compute discrete dimensions
	const uint bsize = bpos.size(); // number of values inside a block
	const uint bxsize = (xsize + 1) * bsize; // number of values on x-row
	const uint bxysize = (ysize + 1) * bxsize; // number of values on xy-plane
	const uint ii0 = (IS_3D ? bxysize : 0) + bxsize + bsize; // reserved zeros in the beginning of value table
	const uint vsize = ii0 + (IS_3D ? zsize + 1 : zsize) * bxysize; // total number of values

	std::cout << "bodies = " << xsize * ysize * zsize * bsize << std::endl;

	// initialize stationary state
	Buffer<Complex> Psi0(vsize, Complex(0,0)); // initial discrete wave function
	Buffer<ddouble> pot(vsize, 0.0); // discrete potential multiplied by time step size
	ddouble g = state.getG(); // effective interaction strength
	ddouble maxpot = 0.0; // maximal value of potential
	for(k=0; k<zsize; k++)
	{
		for(j=0; j<ysize; j++)
		{
			for(i=0; i<xsize; i++)
			{
				for(l=0; l<bsize; l++)
				{
					const uint ii = ii0 + k * bxysize + j * bxsize + i * bsize + l;
					const Vector3 p(p0.x + block_scale * (i * BLOCK_WIDTH.x + bpos[l].x), p0.y + block_scale * (j * BLOCK_WIDTH.y + bpos[l].y), p0.z + block_scale * (k * BLOCK_WIDTH.z + bpos[l].z)); // position
					Psi0[ii] = state.getPsi(p);
					pot[ii] = potentialV3(p);
					const ddouble poti = pot[ii] + g * Psi0[ii].normsq();
					if(poti > maxpot) maxpot = poti;
				}
			}
		}
	}

	// Initialize device memory
	size_t dxsize = xsize + 2; // One element buffer to both ends
	size_t dysize = ysize + 2;
	size_t dzsize_d0 = zsize_d0 + 2;
	size_t dzsize_d1 = zsize_d1 + 2;
	size_t dzsize_d2 = zsize_d2 + 2;
	size_t dzsize_d3 = zsize_d3 + 2;

	cudaExtent psiExtent_d0 = make_cudaExtent(dxsize * sizeof(BlockPsis), dysize, dzsize_d0);
	cudaExtent potExtent_d0 = make_cudaExtent(dxsize * sizeof(BlockPots), dysize, dzsize_d0);
	cudaExtent psiExtent_d1 = make_cudaExtent(dxsize * sizeof(BlockPsis), dysize, dzsize_d1);
	cudaExtent potExtent_d1 = make_cudaExtent(dxsize * sizeof(BlockPots), dysize, dzsize_d1);
	cudaExtent psiExtent_d2 = make_cudaExtent(dxsize * sizeof(BlockPsis), dysize, dzsize_d2);
	cudaExtent potExtent_d2 = make_cudaExtent(dxsize * sizeof(BlockPots), dysize, dzsize_d2);
	cudaExtent psiExtent_d3 = make_cudaExtent(dxsize * sizeof(BlockPsis), dysize, dzsize_d3);
	cudaExtent potExtent_d3 = make_cudaExtent(dxsize * sizeof(BlockPots), dysize, dzsize_d3);

	cudaPitchedPtr d_cudaEvenPsi_d0;
	cudaPitchedPtr d_cudaOddPsi_d0;
	cudaPitchedPtr d_cudaPot_d0;

	cudaPitchedPtr d_cudaEvenPsi_d1;
	cudaPitchedPtr d_cudaOddPsi_d1;
	cudaPitchedPtr d_cudaPot_d1;
	
	cudaPitchedPtr d_cudaEvenPsi_d2;
	cudaPitchedPtr d_cudaOddPsi_d2;
	cudaPitchedPtr d_cudaPot_d2;

	cudaPitchedPtr d_cudaEvenPsi_d3;
	cudaPitchedPtr d_cudaOddPsi_d3;
	cudaPitchedPtr d_cudaPot_d3;

	cudaSetDevice(0);
	cudaDeviceEnablePeerAccess(1, 0);
	checkCudaErrors(cudaMalloc3D(&d_cudaEvenPsi_d0, psiExtent_d0));
	checkCudaErrors(cudaMalloc3D(&d_cudaOddPsi_d0, psiExtent_d0));
	checkCudaErrors(cudaMalloc3D(&d_cudaPot_d0, potExtent_d0));

	cudaSetDevice(1);
	cudaDeviceEnablePeerAccess(0, 0);
	checkCudaErrors(cudaMalloc3D(&d_cudaEvenPsi_d1, psiExtent_d1));
	checkCudaErrors(cudaMalloc3D(&d_cudaOddPsi_d1, psiExtent_d1));
	checkCudaErrors(cudaMalloc3D(&d_cudaPot_d1, potExtent_d1));
	
	cudaSetDevice(2);
	cudaDeviceEnablePeerAccess(3, 0);
	checkCudaErrors(cudaMalloc3D(&d_cudaEvenPsi_d2, psiExtent_d2));
	checkCudaErrors(cudaMalloc3D(&d_cudaOddPsi_d2, psiExtent_d2));
	checkCudaErrors(cudaMalloc3D(&d_cudaPot_d2, potExtent_d2));

	cudaSetDevice(3);
	cudaDeviceEnablePeerAccess(2, 0);
	checkCudaErrors(cudaMalloc3D(&d_cudaEvenPsi_d3, psiExtent_d3));
	checkCudaErrors(cudaMalloc3D(&d_cudaOddPsi_d3, psiExtent_d3));
	checkCudaErrors(cudaMalloc3D(&d_cudaPot_d3, potExtent_d3));

    // Offsets are for the zero valued padding on the edges, offset = z + y + x in bytes
	size_t offset_d0 = d_cudaEvenPsi_d0.pitch * dysize + d_cudaEvenPsi_d0.pitch + sizeof(BlockPsis);
	size_t potOffset_d0 = d_cudaPot_d0.pitch * dysize + d_cudaPot_d0.pitch + sizeof(BlockPots);
	PitchedPtr d_evenPsi_d0 = {(char*)d_cudaEvenPsi_d0.ptr + offset_d0, d_cudaEvenPsi_d0.pitch, d_cudaEvenPsi_d0.pitch * dysize};
	PitchedPtr d_oddPsi_d0 = {(char*)d_cudaOddPsi_d0.ptr + offset_d0, d_cudaOddPsi_d0.pitch, d_cudaOddPsi_d0.pitch * dysize};
	PitchedPtr d_pot_d0 = {(char*)d_cudaPot_d0.ptr + potOffset_d0, d_cudaPot_d0.pitch, d_cudaPot_d0.pitch * dysize};

	size_t offset_d1 = d_cudaEvenPsi_d1.pitch * dysize + d_cudaEvenPsi_d1.pitch + sizeof(BlockPsis);
	size_t potOffset_d1 = d_cudaPot_d1.pitch * dysize + d_cudaPot_d1.pitch + sizeof(BlockPots);
	PitchedPtr d_evenPsi_d1 = {(char*)d_cudaEvenPsi_d1.ptr + offset_d1, d_cudaEvenPsi_d1.pitch, d_cudaEvenPsi_d1.pitch * dysize};
	PitchedPtr d_oddPsi_d1 = {(char*)d_cudaOddPsi_d1.ptr + offset_d1, d_cudaOddPsi_d1.pitch, d_cudaOddPsi_d1.pitch * dysize};
	PitchedPtr d_pot_d1 = {(char*)d_cudaPot_d1.ptr + potOffset_d1, d_cudaPot_d1.pitch, d_cudaPot_d1.pitch * dysize};
	
	size_t offset_d2 = d_cudaEvenPsi_d2.pitch * dysize + d_cudaEvenPsi_d2.pitch + sizeof(BlockPsis);
	size_t potOffset_d2 = d_cudaPot_d2.pitch * dysize + d_cudaPot_d2.pitch + sizeof(BlockPots);
	PitchedPtr d_evenPsi_d2 = {(char*)d_cudaEvenPsi_d2.ptr + offset_d2, d_cudaEvenPsi_d2.pitch, d_cudaEvenPsi_d2.pitch * dysize};
	PitchedPtr d_oddPsi_d2 = {(char*)d_cudaOddPsi_d2.ptr + offset_d2, d_cudaOddPsi_d2.pitch, d_cudaOddPsi_d2.pitch * dysize};
	PitchedPtr d_pot_d2 = {(char*)d_cudaPot_d2.ptr + potOffset_d2, d_cudaPot_d2.pitch, d_cudaPot_d2.pitch * dysize};

	size_t offset_d3 = d_cudaEvenPsi_d3.pitch * dysize + d_cudaEvenPsi_d3.pitch + sizeof(BlockPsis);
	size_t potOffset_d3 = d_cudaPot_d3.pitch * dysize + d_cudaPot_d3.pitch + sizeof(BlockPots);
	PitchedPtr d_evenPsi_d3 = {(char*)d_cudaEvenPsi_d3.ptr + offset_d3, d_cudaEvenPsi_d3.pitch, d_cudaEvenPsi_d3.pitch * dysize};
	PitchedPtr d_oddPsi_d3 = {(char*)d_cudaOddPsi_d3.ptr + offset_d3, d_cudaOddPsi_d3.pitch, d_cudaOddPsi_d3.pitch * dysize};
	PitchedPtr d_pot_d3 = {(char*)d_cudaPot_d3.ptr + potOffset_d3, d_cudaPot_d3.pitch, d_cudaPot_d3.pitch * dysize};

	// find terms for laplacian
	Buffer<int2> lapind_d0;
	ddouble lapfac = -0.5 * getLaplacian(lapind_d0, sizeof(BlockPsis), d_evenPsi_d0.pitch, d_evenPsi_d0.slicePitch) / (block_scale * block_scale);
	const uint lapsize = lapind_d0.size() / bsize;
	ddouble lapfac0 = lapsize * (-lapfac);

	Buffer<int2> lapind_d1;
	getLaplacian(lapind_d1, sizeof(BlockPsis), d_evenPsi_d1.pitch, d_evenPsi_d1.slicePitch);
	
	Buffer<int2> lapind_d2;
	getLaplacian(lapind_d2, sizeof(BlockPsis), d_evenPsi_d2.pitch, d_evenPsi_d2.slicePitch);
	
	Buffer<int2> lapind_d3;
	getLaplacian(lapind_d3, sizeof(BlockPsis), d_evenPsi_d3.pitch, d_evenPsi_d3.slicePitch);

	// compute time step size
	const uint steps_per_iteration = uint(iteration_period * (maxpot + lapfac0)) + 1; // number of time steps per iteration period
	const ddouble time_step_size = iteration_period / ddouble(steps_per_iteration); // time step in time units

	std::cout << "steps_per_iteration = " << steps_per_iteration << std::endl;

	// multiply terms with time_step_size
	g *= time_step_size;
	lapfac *= time_step_size;
	lapfac0 *= time_step_size;
	for(i=0; i<vsize; i++) pot[i] *= time_step_size;

	int2* d_lapind_d0;
	cudaSetDevice(0);
	checkCudaErrors(cudaMalloc(&d_lapind_d0, lapind_d0.size() * sizeof(int2)));

	int2* d_lapind_d1;
	cudaSetDevice(1);
	checkCudaErrors(cudaMalloc(&d_lapind_d1, lapind_d1.size() * sizeof(int2)));
	
	int2* d_lapind_d2;
	cudaSetDevice(2);
	checkCudaErrors(cudaMalloc(&d_lapind_d2, lapind_d2.size() * sizeof(int2)));

	int2* d_lapind_d3;
	cudaSetDevice(3);
	checkCudaErrors(cudaMalloc(&d_lapind_d3, lapind_d3.size() * sizeof(int2)));

    // Initialize host memory
	size_t hostSize = dxsize * dysize * (zsize + 2);
	BlockPsis* h_evenPsi;// = new BlockPsis[dxsize * dysize * (zsize + 2)];
	BlockPsis* h_oddPsi;// = new BlockPsis[dxsize * dysize * (zsize + 2)];
	BlockPots* h_pot;// = new BlockPots[dxsize * dysize * (zsize + 2)];
	checkCudaErrors(cudaMallocHost(&h_evenPsi, hostSize * sizeof(BlockPsis)));
	checkCudaErrors(cudaMallocHost(&h_oddPsi, hostSize * sizeof(BlockPsis)));
	checkCudaErrors(cudaMallocHost(&h_pot, hostSize * sizeof(BlockPots)));
	memset(h_evenPsi, 0, hostSize * sizeof(BlockPsis));
	memset(h_oddPsi, 0, hostSize * sizeof(BlockPsis));
	memset(h_pot, 0, hostSize * sizeof(BlockPots));

	// initialize discrete field
	const Complex oddPhase = state.getPhase(-0.5 * time_step_size);
	Random rnd(54363);
	for(k=0; k<zsize; k++)
	{
		for(j=0; j<ysize; j++)
		{
			for(i=0; i<xsize; i++)
			{
				for(l=0; l<bsize; l++)
				{		
					const uint srcI = ii0 + k * bxysize + j * bxsize + i * bsize + l;
					const uint dstI = (k+1) * dxsize*dysize + (j+1) * dxsize + (i+1);
					const Vector2 c = 0.01 * rnd.getUniformCircle();
					const Complex noise(c.x + 1.0, c.y);
					const Complex noisedPsi = Psi0[srcI] * noise;
					double2 even = make_double2(noisedPsi.r, noisedPsi.i);
					h_evenPsi[dstI].values[l] = even;
					h_oddPsi[dstI].values[l] = make_double2(oddPhase.r * even.x - oddPhase.i * even.y,
															oddPhase.i * even.x + oddPhase.r * even.y);
					h_pot[dstI].values[l] = pot[srcI];
				}
			}
		}
	}

	cudaPitchedPtr h_cudaEvenPsi_d0 = {0};
	cudaPitchedPtr h_cudaOddPsi_d0 = {0};
	cudaPitchedPtr h_cudaPot_d0 = {0};
	cudaPitchedPtr h_cudaEvenPsi_d1 = {0};
	cudaPitchedPtr h_cudaOddPsi_d1 = {0};
	cudaPitchedPtr h_cudaPot_d1 = {0};
	cudaPitchedPtr h_cudaEvenPsi_d2 = {0};
	cudaPitchedPtr h_cudaOddPsi_d2 = {0};
	cudaPitchedPtr h_cudaPot_d2 = {0};
	cudaPitchedPtr h_cudaEvenPsi_d3 = {0};
	cudaPitchedPtr h_cudaOddPsi_d3 = {0};
	cudaPitchedPtr h_cudaPot_d3 = {0};

	h_cudaEvenPsi_d0.ptr = h_evenPsi;
	h_cudaEvenPsi_d0.pitch = dxsize * sizeof(BlockPsis);
	h_cudaEvenPsi_d0.xsize = d_cudaEvenPsi_d0.xsize;
	h_cudaEvenPsi_d0.ysize = d_cudaEvenPsi_d0.ysize;
	h_cudaEvenPsi_d1.ptr = ((BlockPsis*)h_cudaEvenPsi_d0.ptr) + dxsize * dysize * (dzsize_d0 - 2);
	h_cudaEvenPsi_d1.pitch = dxsize * sizeof(BlockPsis);
	h_cudaEvenPsi_d1.xsize = d_cudaEvenPsi_d1.xsize;
	h_cudaEvenPsi_d1.ysize = d_cudaEvenPsi_d1.ysize;
	h_cudaEvenPsi_d2.ptr = ((BlockPsis*)h_cudaEvenPsi_d1.ptr) + dxsize * dysize * (dzsize_d1 - 2);
	h_cudaEvenPsi_d2.pitch = dxsize * sizeof(BlockPsis);
	h_cudaEvenPsi_d2.xsize = d_cudaEvenPsi_d2.xsize;
	h_cudaEvenPsi_d2.ysize = d_cudaEvenPsi_d2.ysize;
	h_cudaEvenPsi_d3.ptr = ((BlockPsis*)h_cudaEvenPsi_d2.ptr) + dxsize * dysize * (dzsize_d2 - 2);
	h_cudaEvenPsi_d3.pitch = dxsize * sizeof(BlockPsis);
	h_cudaEvenPsi_d3.xsize = d_cudaEvenPsi_d3.xsize;
	h_cudaEvenPsi_d3.ysize = d_cudaEvenPsi_d3.ysize;

	h_cudaOddPsi_d0.ptr = h_oddPsi;
	h_cudaOddPsi_d0.pitch = dxsize * sizeof(BlockPsis);
	h_cudaOddPsi_d0.xsize = d_cudaOddPsi_d0.xsize;
	h_cudaOddPsi_d0.ysize = d_cudaOddPsi_d0.ysize;
	h_cudaOddPsi_d1.ptr = ((BlockPsis*)h_cudaOddPsi_d0.ptr) + dxsize * dysize * (dzsize_d0 - 2);
	h_cudaOddPsi_d1.pitch = dxsize * sizeof(BlockPsis);
	h_cudaOddPsi_d1.xsize = d_cudaOddPsi_d1.xsize;
	h_cudaOddPsi_d1.ysize = d_cudaOddPsi_d1.ysize;
	h_cudaOddPsi_d2.ptr = ((BlockPsis*)h_cudaOddPsi_d1.ptr) + dxsize * dysize * (dzsize_d1 - 2);
	h_cudaOddPsi_d2.pitch = dxsize * sizeof(BlockPsis);
	h_cudaOddPsi_d2.xsize = d_cudaOddPsi_d2.xsize;
	h_cudaOddPsi_d2.ysize = d_cudaOddPsi_d2.ysize;
	h_cudaOddPsi_d3.ptr = ((BlockPsis*)h_cudaOddPsi_d2.ptr) + dxsize * dysize * (dzsize_d2 - 2);
	h_cudaOddPsi_d3.pitch = dxsize * sizeof(BlockPsis);
	h_cudaOddPsi_d3.xsize = d_cudaOddPsi_d3.xsize;
	h_cudaOddPsi_d3.ysize = d_cudaOddPsi_d3.ysize;

	h_cudaPot_d0.ptr = h_pot;
	h_cudaPot_d0.pitch = dxsize * sizeof(BlockPots);
	h_cudaPot_d0.xsize = d_cudaPot_d0.xsize;
	h_cudaPot_d0.ysize = d_cudaPot_d0.ysize;
	h_cudaPot_d1.ptr = ((BlockPots*)h_cudaPot_d0.ptr) + dxsize * dysize * (dzsize_d0 - 2);
	h_cudaPot_d1.pitch = dxsize * sizeof(BlockPots);
	h_cudaPot_d1.xsize = d_cudaPot_d1.xsize;
	h_cudaPot_d1.ysize = d_cudaPot_d1.ysize;
	h_cudaPot_d2.ptr = ((BlockPots*)h_cudaPot_d1.ptr) + dxsize * dysize * (dzsize_d1 - 2);
	h_cudaPot_d2.pitch = dxsize * sizeof(BlockPots);
	h_cudaPot_d2.xsize = d_cudaPot_d2.xsize;
	h_cudaPot_d2.ysize = d_cudaPot_d2.ysize;
	h_cudaPot_d3.ptr = ((BlockPots*)h_cudaPot_d2.ptr) + dxsize * dysize * (dzsize_d2 - 2);
	h_cudaPot_d3.pitch = dxsize * sizeof(BlockPots);
	h_cudaPot_d3.xsize = d_cudaPot_d3.xsize;
	h_cudaPot_d3.ysize = d_cudaPot_d3.ysize;

	// Copy from host memory to device memory
	cudaMemcpy3DParms evenPsiParams_d0 = {0};
	cudaMemcpy3DParms oddPsiParams_d0 = {0};
	cudaMemcpy3DParms potParams_d0 = {0};
	cudaMemcpy3DParms evenPsiParams_d1 = {0};
	cudaMemcpy3DParms oddPsiParams_d1 = {0};
	cudaMemcpy3DParms potParams_d1 = {0};
	cudaMemcpy3DParms evenPsiParams_d2 = {0};
	cudaMemcpy3DParms oddPsiParams_d2 = {0};
	cudaMemcpy3DParms potParams_d2 = {0};
	cudaMemcpy3DParms evenPsiParams_d3 = {0};
	cudaMemcpy3DParms oddPsiParams_d3 = {0};
	cudaMemcpy3DParms potParams_d3 = {0};

	evenPsiParams_d0.srcPtr = h_cudaEvenPsi_d0;
	evenPsiParams_d0.dstPtr = d_cudaEvenPsi_d0;
	evenPsiParams_d0.extent = psiExtent_d0;
	evenPsiParams_d0.kind = cudaMemcpyHostToDevice;
	evenPsiParams_d1.srcPtr = h_cudaEvenPsi_d1;
	evenPsiParams_d1.dstPtr = d_cudaEvenPsi_d1;
	evenPsiParams_d1.extent = psiExtent_d1;
	evenPsiParams_d1.kind = cudaMemcpyHostToDevice;
	evenPsiParams_d2.srcPtr = h_cudaEvenPsi_d2;
	evenPsiParams_d2.dstPtr = d_cudaEvenPsi_d2;
	evenPsiParams_d2.extent = psiExtent_d2;
	evenPsiParams_d2.kind = cudaMemcpyHostToDevice;
	evenPsiParams_d3.srcPtr = h_cudaEvenPsi_d3;
	evenPsiParams_d3.dstPtr = d_cudaEvenPsi_d3;
	evenPsiParams_d3.extent = psiExtent_d3;
	evenPsiParams_d3.kind = cudaMemcpyHostToDevice;

	oddPsiParams_d0.srcPtr = h_cudaOddPsi_d0;
	oddPsiParams_d0.dstPtr = d_cudaOddPsi_d0;
	oddPsiParams_d0.extent = psiExtent_d0;
	oddPsiParams_d0.kind = cudaMemcpyHostToDevice;
	oddPsiParams_d1.srcPtr = h_cudaOddPsi_d1;
	oddPsiParams_d1.dstPtr = d_cudaOddPsi_d1;
	oddPsiParams_d1.extent = psiExtent_d1;
	oddPsiParams_d1.kind = cudaMemcpyHostToDevice;
	oddPsiParams_d2.srcPtr = h_cudaOddPsi_d2;
	oddPsiParams_d2.dstPtr = d_cudaOddPsi_d2;
	oddPsiParams_d2.extent = psiExtent_d2;
	oddPsiParams_d2.kind = cudaMemcpyHostToDevice;
	oddPsiParams_d3.srcPtr = h_cudaOddPsi_d3;
	oddPsiParams_d3.dstPtr = d_cudaOddPsi_d3;
	oddPsiParams_d3.extent = psiExtent_d3;
	oddPsiParams_d3.kind = cudaMemcpyHostToDevice;

	potParams_d0.srcPtr = h_cudaPot_d0;
	potParams_d0.dstPtr = d_cudaPot_d0;
	potParams_d0.extent = potExtent_d0;
	potParams_d0.kind = cudaMemcpyHostToDevice;
	potParams_d1.srcPtr = h_cudaPot_d1;
	potParams_d1.dstPtr = d_cudaPot_d1;
	potParams_d1.extent = potExtent_d1;
	potParams_d1.kind = cudaMemcpyHostToDevice;
	potParams_d2.srcPtr = h_cudaPot_d2;
	potParams_d2.dstPtr = d_cudaPot_d2;
	potParams_d2.extent = potExtent_d2;
	potParams_d2.kind = cudaMemcpyHostToDevice;
	potParams_d3.srcPtr = h_cudaPot_d3;
	potParams_d3.dstPtr = d_cudaPot_d3;
	potParams_d3.extent = potExtent_d3;
	potParams_d3.kind = cudaMemcpyHostToDevice;

	cudaSetDevice(0);
	checkCudaErrors(cudaMemcpy3DAsync(&evenPsiParams_d0));
	checkCudaErrors(cudaMemcpy3DAsync(&oddPsiParams_d0));
	checkCudaErrors(cudaMemcpy3DAsync(&potParams_d0));
	checkCudaErrors(cudaMemcpyAsync(d_lapind_d0, &lapind_d0[0], lapind_d0.size() * sizeof(int2), cudaMemcpyHostToDevice));

	cudaSetDevice(1);
	checkCudaErrors(cudaMemcpy3DAsync(&evenPsiParams_d1));
	checkCudaErrors(cudaMemcpy3DAsync(&oddPsiParams_d1));
	checkCudaErrors(cudaMemcpy3DAsync(&potParams_d1));
	checkCudaErrors(cudaMemcpyAsync(d_lapind_d1, &lapind_d1[0], lapind_d1.size() * sizeof(int2), cudaMemcpyHostToDevice));
	
	cudaSetDevice(2);
	checkCudaErrors(cudaMemcpy3DAsync(&evenPsiParams_d2));
	checkCudaErrors(cudaMemcpy3DAsync(&oddPsiParams_d2));
	checkCudaErrors(cudaMemcpy3DAsync(&potParams_d2));
	checkCudaErrors(cudaMemcpyAsync(d_lapind_d2, &lapind_d2[0], lapind_d2.size() * sizeof(int2), cudaMemcpyHostToDevice));

	cudaSetDevice(3);
	checkCudaErrors(cudaMemcpy3DAsync(&evenPsiParams_d3));
	checkCudaErrors(cudaMemcpy3DAsync(&oddPsiParams_d3));
	checkCudaErrors(cudaMemcpy3DAsync(&potParams_d3));
	checkCudaErrors(cudaMemcpyAsync(d_lapind_d3, &lapind_d3[0], lapind_d3.size() * sizeof(int2), cudaMemcpyHostToDevice));
	
	// Clear host memory after data has been copied to devices
	cudaDeviceSynchronize();
	Psi0.clear();
	pot.clear();
	bpos.clear();
	lapind_d0.clear();
	lapind_d1.clear();
	lapind_d2.clear();
	lapind_d3.clear();
	cudaFreeHost(h_oddPsi);
	cudaFreeHost(h_pot);
#if !(SAVE_PICTURE || SAVE_VOLUME)
	cudaFreeHost(h_evenPsi);
#endif

	// Integrate in time
	uint3 dimensions_d0 = make_uint3(xsize, ysize, zsize_d0);
	uint3 dimensions_d1 = make_uint3(xsize, ysize, zsize_d1);
	uint3 dimensions_d2 = make_uint3(xsize, ysize, zsize_d2);
	uint3 dimensions_d3 = make_uint3(xsize, ysize, zsize_d3);
	double2 lapfacs = make_double2(lapfac, lapfac0);
	uint iter = 0;
	dim3 dimBlock(THREAD_BLOCK_X, THREAD_BLOCK_Y, THREAD_BLOCK_Z * VALUES_IN_BLOCK);
	dim3 dimGrid_d0((xsize + THREAD_BLOCK_X - 1) / THREAD_BLOCK_X,
					(ysize + THREAD_BLOCK_Y - 1) / THREAD_BLOCK_Y,
					(zsize_d0 + THREAD_BLOCK_Z - 1) / THREAD_BLOCK_Z);
	dim3 dimGrid_d1((xsize + THREAD_BLOCK_X - 1) / THREAD_BLOCK_X,
					(ysize + THREAD_BLOCK_Y - 1) / THREAD_BLOCK_Y,
					(zsize_d1 + THREAD_BLOCK_Z - 1) / THREAD_BLOCK_Z);
	dim3 dimGrid_d2((xsize + THREAD_BLOCK_X - 1) / THREAD_BLOCK_X,
					(ysize + THREAD_BLOCK_Y - 1) / THREAD_BLOCK_Y,
					(zsize_d2 + THREAD_BLOCK_Z - 1) / THREAD_BLOCK_Z);
	dim3 dimGrid_d3((xsize + THREAD_BLOCK_X - 1) / THREAD_BLOCK_X,
					(ysize + THREAD_BLOCK_Y - 1) / THREAD_BLOCK_Y,
					(zsize_d3 + THREAD_BLOCK_Z - 1) / THREAD_BLOCK_Z);

	cudaExtent oneSliceExtent = make_cudaExtent(dxsize * sizeof(BlockPsis), dysize, 1);
			
	cudaMemcpy3DParms even_d0_to_d1 = {0};
	cudaMemcpy3DParms even_d1_to_d0 = {0};
	cudaMemcpy3DParms even_d1_to_d2 = {0};
	cudaMemcpy3DParms even_d2_to_d1 = {0};
	cudaMemcpy3DParms even_d2_to_d3 = {0};
	cudaMemcpy3DParms even_d3_to_d2 = {0};
	
	cudaMemcpy3DParms odd_d0_to_d1 = {0};
	cudaMemcpy3DParms odd_d1_to_d0 = {0};
	cudaMemcpy3DParms odd_d1_to_d2 = {0};
	cudaMemcpy3DParms odd_d2_to_d1 = {0};
	cudaMemcpy3DParms odd_d2_to_d3 = {0};
	cudaMemcpy3DParms odd_d3_to_d2 = {0};

	cudaPitchedPtr even_d0_src = d_cudaEvenPsi_d0;
	cudaPitchedPtr even_d0_dst = d_cudaEvenPsi_d0;
	cudaPitchedPtr even_d1_src = d_cudaEvenPsi_d1;
	cudaPitchedPtr even_d1_dst = d_cudaEvenPsi_d1;
	cudaPitchedPtr even_d2_src = d_cudaEvenPsi_d2;
	cudaPitchedPtr even_d2_dst = d_cudaEvenPsi_d2;
	cudaPitchedPtr even_d3_src = d_cudaEvenPsi_d3;
	cudaPitchedPtr even_d3_dst = d_cudaEvenPsi_d3;
	
	cudaPitchedPtr even_d1_src_to_d2 = d_cudaEvenPsi_d1;
	cudaPitchedPtr even_d1_dst_from_d2 = d_cudaEvenPsi_d1;
	cudaPitchedPtr even_d2_src_to_d1 = d_cudaEvenPsi_d2;
	cudaPitchedPtr even_d2_dst_from_d1 = d_cudaEvenPsi_d2;

	cudaPitchedPtr odd_d0_src = d_cudaOddPsi_d0;
	cudaPitchedPtr odd_d0_dst = d_cudaOddPsi_d0;
	cudaPitchedPtr odd_d1_src = d_cudaOddPsi_d1;
	cudaPitchedPtr odd_d1_dst = d_cudaOddPsi_d1;
	cudaPitchedPtr odd_d2_src = d_cudaOddPsi_d2;
	cudaPitchedPtr odd_d2_dst = d_cudaOddPsi_d2;
	cudaPitchedPtr odd_d3_src = d_cudaOddPsi_d3;
	cudaPitchedPtr odd_d3_dst = d_cudaOddPsi_d3;
	
	cudaPitchedPtr odd_d1_src_to_d2 = d_cudaOddPsi_d1;
	cudaPitchedPtr odd_d1_dst_from_d2 = d_cudaOddPsi_d1;
	cudaPitchedPtr odd_d2_src_to_d1 = d_cudaOddPsi_d2;
	cudaPitchedPtr odd_d2_dst_from_d1 = d_cudaOddPsi_d2;

	even_d0_src.ptr = ((char*)even_d0_src.ptr) + d_evenPsi_d0.slicePitch * (dzsize_d0 - 2);
	even_d0_dst.ptr = ((char*)even_d0_dst.ptr) + d_evenPsi_d0.slicePitch * (dzsize_d0 - 1);
	even_d1_src.ptr = ((char*)even_d1_src.ptr) + d_evenPsi_d1.slicePitch * 1;
	even_d1_dst.ptr = ((char*)even_d1_dst.ptr) + d_evenPsi_d1.slicePitch * 0;
	even_d2_src.ptr = ((char*)even_d2_src.ptr) + d_evenPsi_d2.slicePitch * (dzsize_d2 - 2);
	even_d2_dst.ptr = ((char*)even_d2_dst.ptr) + d_evenPsi_d2.slicePitch * (dzsize_d2 - 1);
	even_d3_src.ptr = ((char*)even_d3_src.ptr) + d_evenPsi_d3.slicePitch * 1;
	even_d3_dst.ptr = ((char*)even_d3_dst.ptr) + d_evenPsi_d3.slicePitch * 0;
	
	even_d1_src_to_d2.ptr = ((char*)even_d1_src_to_d2.ptr) + d_evenPsi_d1.slicePitch * (dzsize_d1 - 2);
	even_d1_dst_from_d2.ptr = ((char*)even_d1_dst_from_d2.ptr) + d_evenPsi_d1.slicePitch * (dzsize_d1 - 1);
	even_d2_src_to_d1.ptr = ((char*)even_d2_src_to_d1.ptr) + d_evenPsi_d2.slicePitch * 1;
	even_d2_dst_from_d1.ptr = ((char*)even_d2_dst_from_d1.ptr) + d_evenPsi_d2.slicePitch * 0;

	odd_d0_src.ptr = ((char*)odd_d0_src.ptr) + d_oddPsi_d0.slicePitch * (dzsize_d0 - 2);
	odd_d0_dst.ptr = ((char*)odd_d0_dst.ptr) + d_oddPsi_d0.slicePitch * (dzsize_d0 - 1);
	odd_d1_src.ptr = ((char*)odd_d1_src.ptr) + d_oddPsi_d1.slicePitch * 1;
	odd_d1_dst.ptr = ((char*)odd_d1_dst.ptr) + d_oddPsi_d1.slicePitch * 0;
	odd_d2_src.ptr = ((char*)odd_d2_src.ptr) + d_oddPsi_d2.slicePitch * (dzsize_d2 - 2);
	odd_d2_dst.ptr = ((char*)odd_d2_dst.ptr) + d_oddPsi_d2.slicePitch * (dzsize_d2 - 1);
	odd_d3_src.ptr = ((char*)odd_d3_src.ptr) + d_oddPsi_d3.slicePitch * 1;
	odd_d3_dst.ptr = ((char*)odd_d3_dst.ptr) + d_oddPsi_d3.slicePitch * 0;
	
	odd_d1_src_to_d2.ptr = ((char*)odd_d1_src_to_d2.ptr) + d_oddPsi_d1.slicePitch * (dzsize_d1 - 2);
	odd_d1_dst_from_d2.ptr = ((char*)odd_d1_dst_from_d2.ptr) + d_oddPsi_d1.slicePitch * (dzsize_d1 - 1);
	odd_d2_src_to_d1.ptr = ((char*)odd_d2_src_to_d1.ptr) + d_oddPsi_d2.slicePitch * 1;
	odd_d2_dst_from_d1.ptr = ((char*)odd_d2_dst_from_d1.ptr) + d_oddPsi_d2.slicePitch * 0;

	even_d0_to_d1.srcPtr = even_d0_src;
	even_d0_to_d1.dstPtr = even_d1_dst;
	even_d0_to_d1.extent = oneSliceExtent;
	even_d0_to_d1.kind = cudaMemcpyDefault;
	even_d1_to_d0.srcPtr = even_d1_src;
	even_d1_to_d0.dstPtr = even_d0_dst;
	even_d1_to_d0.extent = oneSliceExtent;
	even_d1_to_d0.kind = cudaMemcpyDefault;
	even_d2_to_d3.srcPtr = even_d2_src;
	even_d2_to_d3.dstPtr = even_d3_dst;
	even_d2_to_d3.extent = oneSliceExtent;
	even_d2_to_d3.kind = cudaMemcpyDefault;
	even_d3_to_d2.srcPtr = even_d3_src;
	even_d3_to_d2.dstPtr = even_d2_dst;
	even_d3_to_d2.extent = oneSliceExtent;
	even_d3_to_d2.kind = cudaMemcpyDefault;
	
	even_d1_to_d2.srcPtr = even_d1_src_to_d2;
	even_d1_to_d2.dstPtr = even_d2_dst_from_d1;
	even_d1_to_d2.extent = oneSliceExtent;
	even_d1_to_d2.kind = cudaMemcpyDefault;
	
	even_d2_to_d1.srcPtr = even_d2_src_to_d1;
	even_d2_to_d1.dstPtr = even_d1_dst_from_d2;
	even_d2_to_d1.extent = oneSliceExtent;
	even_d2_to_d1.kind = cudaMemcpyDefault;

	odd_d0_to_d1.srcPtr = odd_d0_src;
	odd_d0_to_d1.dstPtr = odd_d1_dst;
	odd_d0_to_d1.extent = oneSliceExtent;
	odd_d0_to_d1.kind = cudaMemcpyDefault;
	odd_d1_to_d0.srcPtr = odd_d1_src;
	odd_d1_to_d0.dstPtr = odd_d0_dst;
	odd_d1_to_d0.extent = oneSliceExtent;
	odd_d1_to_d0.kind = cudaMemcpyDefault;
	odd_d2_to_d3.srcPtr = odd_d2_src;
	odd_d2_to_d3.dstPtr = odd_d3_dst;
	odd_d2_to_d3.extent = oneSliceExtent;
	odd_d2_to_d3.kind = cudaMemcpyDefault;
	odd_d3_to_d2.srcPtr = odd_d3_src;
	odd_d3_to_d2.dstPtr = odd_d2_dst;
	odd_d3_to_d2.extent = oneSliceExtent;
	odd_d3_to_d2.kind = cudaMemcpyDefault;

	odd_d1_to_d2.srcPtr = odd_d1_src_to_d2;
	odd_d1_to_d2.dstPtr = odd_d2_dst_from_d1;
	odd_d1_to_d2.extent = oneSliceExtent;
	odd_d1_to_d2.kind = cudaMemcpyDefault;
	
	odd_d2_to_d1.srcPtr = odd_d2_src_to_d1	;
	odd_d2_to_d1.dstPtr = odd_d1_dst_from_d2;
	odd_d2_to_d1.extent = oneSliceExtent;
	odd_d2_to_d1.kind = cudaMemcpyDefault;

	//std::cout << "block dim = " << dimBlock.x << ", " << dimBlock.y << ", "<< dimBlock.z << std::endl;
	//std::cout << "grid dim = " << dimGrid.x << ", " << dimGrid.y << ", "<< dimGrid.z << std::endl;

	cudaStream_t stream0_d0;
	
	cudaStream_t stream0_d1;
	cudaStream_t stream1_d1;
	
	cudaStream_t stream0_d2;
	cudaStream_t stream1_d2;
	
	cudaStream_t stream0_d3;
	
	cudaEvent_t event_d0_to_d1;
	
	cudaEvent_t event_d1_to_d0;
	cudaEvent_t event_d1_to_d2;
	cudaEvent_t event_d1_kernel;
	
	cudaEvent_t event_d2_to_d1;
	cudaEvent_t event_d2_to_d3;
	cudaEvent_t event_d2_kernel;
	
	cudaEvent_t event_d3_to_d2;
	
	cudaSetDevice(0);
	cudaStreamCreate(&stream0_d0);
	cudaEventCreate(&event_d0_to_d1);
	cudaEventRecord(event_d0_to_d1, stream0_d0);
	
	cudaSetDevice(1);
	cudaStreamCreate(&stream0_d1);
	cudaEventCreate(&event_d1_to_d0);
	cudaEventRecord(event_d1_to_d0, stream0_d1);
	cudaStreamCreate(&stream1_d1);
	cudaEventCreate(&event_d1_to_d2);
	cudaEventRecord(event_d1_to_d2, stream1_d1);
	cudaEventCreate(&event_d1_kernel);
	
	cudaSetDevice(2);
	cudaStreamCreate(&stream0_d2);
	cudaEventCreate(&event_d2_to_d1);
	cudaEventRecord(event_d2_to_d1, stream0_d2);
	cudaStreamCreate(&stream1_d2);
	cudaEventCreate(&event_d2_to_d3);
	cudaEventRecord(event_d2_to_d3, stream1_d2);
	cudaEventCreate(&event_d2_kernel);
	
	cudaSetDevice(3);
	cudaStreamCreate(&stream0_d3);
	cudaEventCreate(&event_d3_to_d2);
	cudaEventRecord(event_d3_to_d2, stream0_d3);

	d_cudaEvenPsi_d0.ptr = ((char*)d_cudaEvenPsi_d0.ptr) + d_evenPsi_d0.slicePitch;
	h_cudaEvenPsi_d0.ptr = ((BlockPsis*)h_cudaEvenPsi_d0.ptr) + dxsize * dysize;
	psiExtent_d0.depth -= 2;

#if SAVE_PICTURE || SAVE_VOLUME
	cudaMemcpy3DParms evenPsiBackParams_d0 = {0};
	evenPsiBackParams_d0.srcPtr = d_cudaEvenPsi_d0;
	evenPsiBackParams_d0.dstPtr = h_cudaEvenPsi_d0;
	evenPsiBackParams_d0.extent = psiExtent_d0;
	evenPsiBackParams_d0.kind = cudaMemcpyDeviceToHost;
	
	d_cudaEvenPsi_d1.ptr = ((char*)d_cudaEvenPsi_d1.ptr) + d_evenPsi_d1.slicePitch;
	h_cudaEvenPsi_d1.ptr = ((BlockPsis*)h_cudaEvenPsi_d1.ptr) + dxsize * dysize;
	psiExtent_d1.depth -= 2;
	
	cudaMemcpy3DParms evenPsiBackParams_d1 = {0};
	evenPsiBackParams_d1.srcPtr = d_cudaEvenPsi_d1;
	evenPsiBackParams_d1.dstPtr = h_cudaEvenPsi_d1;
	evenPsiBackParams_d1.extent = psiExtent_d1;
	evenPsiBackParams_d1.kind = cudaMemcpyDeviceToHost;
	
	d_cudaEvenPsi_d2.ptr = ((char*)d_cudaEvenPsi_d2.ptr) + d_evenPsi_d2.slicePitch;
	h_cudaEvenPsi_d2.ptr = ((BlockPsis*)h_cudaEvenPsi_d2.ptr) + dxsize * dysize;
	psiExtent_d2.depth -= 2;
	
	cudaMemcpy3DParms evenPsiBackParams_d2 = {0};
	evenPsiBackParams_d2.srcPtr = d_cudaEvenPsi_d2;
	evenPsiBackParams_d2.dstPtr = h_cudaEvenPsi_d2;
	evenPsiBackParams_d2.extent = psiExtent_d2;
	evenPsiBackParams_d2.kind = cudaMemcpyDeviceToHost;
	
	d_cudaEvenPsi_d3.ptr = ((char*)d_cudaEvenPsi_d3.ptr) + d_evenPsi_d3.slicePitch;
	h_cudaEvenPsi_d3.ptr = ((BlockPsis*)h_cudaEvenPsi_d3.ptr) + dxsize * dysize;
	psiExtent_d3.depth -= 2;
	
	cudaMemcpy3DParms evenPsiBackParams_d3 = {0};
	evenPsiBackParams_d3.srcPtr = d_cudaEvenPsi_d3;
	evenPsiBackParams_d3.dstPtr = h_cudaEvenPsi_d3;
	evenPsiBackParams_d3.extent = psiExtent_d3;
	evenPsiBackParams_d3.kind = cudaMemcpyDeviceToHost;
#endif
	const uint time0 = clock();
	while(true)
	{
#if SAVE_PICTURE || SAVE_VOLUME
		//cudaDeviceSynchronize();
#endif

#if SAVE_PICTURE
		// draw picture
		Picture pic(dxsize, dysize);
		k = zsize / 2 + 1;
		for(j=0; j<dysize; j++)
		{
			for(i=0; i<dxsize; i++)
			{
				const uint idx = k * dxsize * dysize + j * dxsize + i;
				double norm = sqrt(h_evenPsi[idx].values[0].x*h_evenPsi[idx].values[0].x + h_evenPsi[idx].values[0].y*h_evenPsi[idx].values[0].y);

				pic.setColor(i, j, 5.0 * Vector4(h_evenPsi[idx].values[0].x, norm, h_evenPsi[idx].values[0].y, 1.0));
			}
		}
		std::ostringstream picpath;
		picpath << "tulokset/kuva" << iter << ".bmp";
		pic.save(picpath.str(), false);
#endif

#if SAVE_VOLUME
		// save volume map
		const ddouble fmax = state.searchFunctionMax();
		const ddouble unit = 60000.0 / (bsize * fmax * fmax);
		Buffer<ushort> vol(dxsize * dysize * (zsize + 2));
		for(k=0; k<(zsize + 2); k++)
		{
			for(j=0; j<dysize; j++)
			{
				for(i=0; i<dxsize; i++)
				{
					const uint idx = k * dxsize * dysize + j * dxsize + i;
					ddouble sum = 0.0;
					for(l=0; l<bsize; l++) 
					{
						sum += h_evenPsi[idx].values[0].x*h_evenPsi[idx].values[0].x + h_evenPsi[idx].values[0].y*h_evenPsi[idx].values[0].y;
					}
					sum *= unit;
					vol[idx] = (sum > 65535.0 ? 65535 : ushort(sum));
				}
			}
		}
		Text volpath;
		volpath << "volume" << iter << ".mhd";
		saveVolumeMap(volpath.str(), vol, dxsize, dysize, (zsize + 2), block_scale * BLOCK_WIDTH);
#endif

		// finish iteration
		if(++iter > number_of_iterations) break;

		// integrate one iteration
		std::cout << "Iteration " << iter << std::endl;
		for(uint step=0; step<steps_per_iteration; step++)
		{
			// update odd values
			cudaSetDevice(0);
			cudaStreamWaitEvent(stream0_d0, event_d1_to_d0, 0);
			update<<<dimGrid_d0, dimBlock, 0, stream0_d0>>>(d_oddPsi_d0, d_evenPsi_d0, d_pot_d0, d_lapind_d0, lapfacs, g, dimensions_d0);
			cudaSetDevice(1);
			cudaStreamWaitEvent(stream0_d1, event_d0_to_d1, 0);
			cudaStreamWaitEvent(stream0_d1, event_d2_to_d1, 0);
			update<<<dimGrid_d1, dimBlock, 0, stream0_d1>>>(d_oddPsi_d1, d_evenPsi_d1, d_pot_d1, d_lapind_d1, lapfacs, g, dimensions_d1);
			cudaEventRecord(event_d1_kernel, stream0_d1);
			cudaSetDevice(2);
			cudaStreamWaitEvent(stream0_d2, event_d1_to_d2, 0);
			cudaStreamWaitEvent(stream0_d2, event_d3_to_d2, 0);
			update<<<dimGrid_d2, dimBlock, 0, stream0_d2>>>(d_oddPsi_d2, d_evenPsi_d2, d_pot_d2, d_lapind_d2, lapfacs, g, dimensions_d2);
			cudaEventRecord(event_d2_kernel, stream0_d2);
			cudaSetDevice(3);
			cudaStreamWaitEvent(stream0_d3, event_d2_to_d3, 0);
			update<<<dimGrid_d3, dimBlock, 0, stream0_d3>>>(d_oddPsi_d3, d_evenPsi_d3, d_pot_d3, d_lapind_d3, lapfacs, g, dimensions_d3);

			cudaSetDevice(0);
			cudaMemcpy3DAsync(&odd_d0_to_d1, stream0_d0);
			cudaEventRecord(event_d0_to_d1, stream0_d0);
			cudaSetDevice(1);
			cudaMemcpy3DAsync(&odd_d1_to_d0, stream0_d1);
			cudaEventRecord(event_d1_to_d0, stream0_d1);
			cudaStreamWaitEvent(stream1_d1, event_d1_kernel, 0);
			cudaMemcpy3DAsync(&odd_d1_to_d2, stream1_d1);
			cudaEventRecord(event_d1_to_d2, stream1_d1);
			cudaSetDevice(2);
			cudaMemcpy3DAsync(&odd_d2_to_d3, stream0_d2);
			cudaEventRecord(event_d2_to_d3, stream0_d2);
			cudaStreamWaitEvent(stream1_d2, event_d2_kernel, 0);
			cudaMemcpy3DAsync(&odd_d2_to_d1, stream1_d2);
			cudaEventRecord(event_d2_to_d1, stream1_d2);
			cudaSetDevice(3);
			cudaMemcpy3DAsync(&odd_d3_to_d2, stream0_d3);
			cudaEventRecord(event_d3_to_d2, stream0_d3);

			// update even values
			cudaSetDevice(0);
			cudaStreamWaitEvent(stream0_d0, event_d1_to_d0, 0);
			update<<<dimGrid_d0, dimBlock, 0, stream0_d0>>>(d_evenPsi_d0, d_oddPsi_d0, d_pot_d0, d_lapind_d0, lapfacs, g, dimensions_d0);
			cudaSetDevice(1);
			cudaStreamWaitEvent(stream0_d1, event_d0_to_d1, 0);
			cudaStreamWaitEvent(stream0_d1, event_d2_to_d1, 0);
			update<<<dimGrid_d1, dimBlock, 0, stream0_d1>>>(d_evenPsi_d1, d_oddPsi_d1, d_pot_d1, d_lapind_d1, lapfacs, g, dimensions_d1);
			cudaEventRecord(event_d1_kernel, stream0_d1);
			cudaSetDevice(2);
			cudaStreamWaitEvent(stream0_d2, event_d1_to_d2, 0);
			cudaStreamWaitEvent(stream0_d2, event_d3_to_d2, 0);
			update<<<dimGrid_d2, dimBlock, 0, stream0_d2>>>(d_evenPsi_d2, d_oddPsi_d2, d_pot_d2, d_lapind_d2, lapfacs, g, dimensions_d2);
			cudaEventRecord(event_d2_kernel, stream0_d2);
			cudaSetDevice(3);
			cudaStreamWaitEvent(stream0_d3, event_d2_to_d3, 0);
			update<<<dimGrid_d3, dimBlock, 0, stream0_d3>>>(d_evenPsi_d3, d_oddPsi_d3, d_pot_d3, d_lapind_d3, lapfacs, g, dimensions_d3);

			cudaSetDevice(0);
			cudaMemcpy3DAsync(&even_d0_to_d1, stream0_d0);
			cudaEventRecord(event_d0_to_d1, stream0_d0);
			cudaSetDevice(1);
			cudaMemcpy3DAsync(&even_d1_to_d0, stream0_d1);
			cudaEventRecord(event_d1_to_d0, stream0_d1);
			cudaStreamWaitEvent(stream1_d1, event_d1_kernel, 0);
			cudaMemcpy3DAsync(&even_d1_to_d2, stream1_d1);
			cudaEventRecord(event_d1_to_d2, stream1_d1);
			cudaSetDevice(2);
			cudaMemcpy3DAsync(&even_d2_to_d3, stream0_d2);
			cudaEventRecord(event_d2_to_d3, stream0_d2);
			cudaStreamWaitEvent(stream1_d2, event_d2_kernel, 0);
			cudaMemcpy3DAsync(&even_d2_to_d1, stream1_d2);
			cudaEventRecord(event_d2_to_d1, stream1_d2);
			cudaSetDevice(3);
			cudaMemcpy3DAsync(&even_d3_to_d2, stream0_d3);
			cudaEventRecord(event_d3_to_d2, stream0_d3);			
		}
#if SAVE_PICTURE || SAVE_VOLUME
		// Copy back from device memory to host memory
		cudaSetDevice(0);
		checkCudaErrors(cudaMemcpy3DAsync(&evenPsiBackParams_d0));

		cudaSetDevice(1);
		checkCudaErrors(cudaMemcpy3DAsync(&evenPsiBackParams_d1));
		
		cudaSetDevice(2);
		checkCudaErrors(cudaMemcpy3DAsync(&evenPsiBackParams_d2));

		cudaSetDevice(3);
		checkCudaErrors(cudaMemcpy3DAsync(&evenPsiBackParams_d3));
#endif
	}
	std::cout << "iteration time = " << (1e-6 * (clock() - time0)) / number_of_iterations << std::endl;
	std::cout << "total time = " << 1e-6 * (clock() - time0) << std::endl;

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch kernels (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	return 0;
}    

int main ( int argc, char** argv )
{
#if LOAD_STATE_FROM_DISK
	VortexState state;
	state.load("state.dat");
	const ddouble eps = 1e-5 * state.searchFunctionMax();
	const ddouble maxr = state.searchMaxR(eps);
	const ddouble maxz = state.searchMaxZ(eps);
#else
	// preliminary vortex state to find vortex size
	VortexState state0;
	state0.setKappa(KAPPA);
	state0.setG(G);
	if(IS_3D) state0.setRange(0.0, 15.0, 35.0, 0.2, 0.2); // use this for 3d
	else state0.setRange(0.0, 15.0, 1.0, 0.2, 1.0); // use this for 2d
	state0.iterateSolution(potentialRZ, 10000, 1e-29);
	const ddouble eps = 1e-5 * state0.searchFunctionMax();
	const ddouble minr = state0.searchMinR(eps);
	ddouble maxr = state0.searchMaxR(eps);
	ddouble maxz = state0.searchMaxZ(eps);
	//std::cout << "maxf=" << 1e6*eps << " minr=" << minr << " maxr=" << maxr << " maxz=" << maxz << std::endl;

	// more accurate vortex state
	VortexState state;
	state.setKappa(KAPPA);
	state.setG(G);
	if(IS_3D) state.setRange(minr, maxr, maxz, 0.03, 0.03); // use this for 3d
	else state.setRange(minr, maxr, 1.0, 0.03, 1.0); // use this for 2d
	state.initialize(state0);
	state.iterateSolution(potentialRZ, 10000, 1e-29);
	state.save("state.dat");
	maxr = state.searchMaxR(eps);
	maxz = state.searchMaxZ(eps);
	//std::cout << "maxf=" << state.searchFunctionMax() << std::endl;
#endif

	const int number_of_iterations = 50;
	const ddouble iteration_period = 1.0;
	const ddouble block_scale = PIx2 / (20.0 * sqrt(state.integrateCurvature()));
	
	std::cout << "4 GPUs version pasimysiini" << std::endl;
	std::cout << "kappa = " << KAPPA << std::endl;
	std::cout << "g = " << G << std::endl;
	std::cout << "block_scale = " << block_scale << std::endl;
	std::cout << "iteration_period = " << iteration_period << std::endl;
	std::cout << "maxr = " << maxr << std::endl;
	std::cout << "maxz = " << maxz << std::endl;

	// integrate in time using DEC
	if(IS_3D) integrateInTime(state, block_scale, Vector3(-maxr, -maxr, -maxz), Vector3(maxr, maxr, maxz), iteration_period, number_of_iterations); // use this for 3d
	else integrateInTime(state, block_scale, Vector3(-maxr, -maxr, 0.0), Vector3(maxr, maxr, 0.0), iteration_period, number_of_iterations); // use this for 2d

	return 0;
}
