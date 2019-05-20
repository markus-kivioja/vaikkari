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
#define SAVE_VOLUME 0

#define THREAD_BLOCK_X 8
#define THREAD_BLOCK_Y 8
#define THREAD_BLOCK_Z 1
#define FACE_COUNT 4
#define VALUES_IN_BLOCK 12
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
	ind[0] = make_int2(0, 9);
	ind[1] = make_int2(0, 10);
	ind[2] = make_int2(-nx + nz, 2);
	ind[3] = make_int2(-nx, 3);
	ind[4] = make_int2(0, 2);
	ind[5] = make_int2(0, 3);
	ind[6] = make_int2(nx - ny, 5);
	ind[7] = make_int2(-ny, 8);
	ind[8] = make_int2(0, 1);
	ind[9] = make_int2(0, 4);
	ind[10] = make_int2(nx - nz, 0);
	ind[11] = make_int2(-nz, 11);
	ind[12] = make_int2(0, 1);
	ind[13] = make_int2(0, 4);
	ind[14] = make_int2(nx, 0);
	ind[15] = make_int2(0, 11);
	ind[16] = make_int2(0, 2);
	ind[17] = make_int2(0, 3);
	ind[18] = make_int2(nx, 5);
	ind[19] = make_int2(0, 8);
	ind[20] = make_int2(0, 6);
	ind[21] = make_int2(0, 7);
	ind[22] = make_int2(-nx + ny, 1);
	ind[23] = make_int2(-nx, 4);
	ind[24] = make_int2(0, 5);
	ind[25] = make_int2(0, 8);
	ind[26] = make_int2(ny - nz, 9);
	ind[27] = make_int2(-nz, 10);
	ind[28] = make_int2(0, 5);
	ind[29] = make_int2(0, 8);
	ind[30] = make_int2(ny, 9);
	ind[31] = make_int2(0, 10);
	ind[32] = make_int2(0, 6);
	ind[33] = make_int2(0, 7);
	ind[34] = make_int2(ny, 1);
	ind[35] = make_int2(0, 4);
	ind[36] = make_int2(0, 0);
	ind[37] = make_int2(0, 11);
	ind[38] = make_int2(-ny + nz, 6);
	ind[39] = make_int2(-ny, 7);
	ind[40] = make_int2(0, 0);
	ind[41] = make_int2(0, 11);
	ind[42] = make_int2(nz, 6);
	ind[43] = make_int2(0, 7);
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

__global__ void update(PitchedPtr nextStep, PitchedPtr prevStep, PitchedPtr potentials, int2* lapind, double2 lapfacs, double g, uint3 dimensions)
{
	size_t xid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t yid = blockIdx.y * blockDim.y + threadIdx.y;
	size_t zid = blockIdx.z * blockDim.z + threadIdx.z;

	// Load Laplacian indices into LDS
	__shared__ int2 ldsLapind[INDICES_PER_BLOCK];
	size_t threadIdxInBlock = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
	if (threadIdxInBlock < INDICES_PER_BLOCK)
	{
		ldsLapind[threadIdxInBlock] = lapind[threadIdxInBlock];
	}
	__syncthreads();
	
	size_t dataZid = zid / VALUES_IN_BLOCK;
	
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
	size_t nodeId = zid % VALUES_IN_BLOCK;
	uint lapi = nodeId * FACE_COUNT;
	double2 sum = (*(BlockPsis*)(prevPsi + ldsLapind[lapi].x)).values[ldsLapind[lapi++].y];
	sum += (*(BlockPsis*)(prevPsi + ldsLapind[lapi].x)).values[ldsLapind[lapi++].y];
	sum += (*(BlockPsis*)(prevPsi + ldsLapind[lapi].x)).values[ldsLapind[lapi++].y];
	sum += (*(BlockPsis*)(prevPsi + ldsLapind[lapi].x)).values[ldsLapind[lapi++].y];
	double2 prev = (*(BlockPsis*)prevPsi).values[nodeId];
	double normsq = prev.x * prev.x + prev.y * prev.y;
	sum = lapfacs.x * sum + (lapfacs.y + (*pot).values[nodeId] + g * normsq) * prev;

	(*nextPsi).values[nodeId] += make_double2(sum.y, -sum.x);
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
	size_t dysize = ysize + 2; // One element buffer to both ends
	size_t dzsize = zsize + 2; // One element buffer to both ends
    cudaExtent psiExtent = make_cudaExtent(dxsize * sizeof(BlockPsis), dysize, dzsize);
    cudaExtent potExtent = make_cudaExtent(dxsize * sizeof(BlockPots), dysize, dzsize);
    
    cudaPitchedPtr d_cudaEvenPsi;
    cudaPitchedPtr d_cudaOddPsi;
    cudaPitchedPtr d_cudaPot;
    
	checkCudaErrors(cudaMalloc3D(&d_cudaEvenPsi, psiExtent));
    checkCudaErrors(cudaMalloc3D(&d_cudaOddPsi, psiExtent));
    checkCudaErrors(cudaMalloc3D(&d_cudaPot, potExtent));

    size_t offset = d_cudaEvenPsi.pitch * dysize + d_cudaEvenPsi.pitch + sizeof(BlockPsis);
    size_t potOffset = d_cudaPot.pitch * dysize + d_cudaPot.pitch + sizeof(BlockPots);
    PitchedPtr d_evenPsi = {(char*)d_cudaEvenPsi.ptr + offset, d_cudaEvenPsi.pitch, d_cudaEvenPsi.pitch * dysize};
    PitchedPtr d_oddPsi = {(char*)d_cudaOddPsi.ptr + offset, d_cudaOddPsi.pitch, d_cudaOddPsi.pitch * dysize};
    PitchedPtr d_pot = {(char*)d_cudaPot.ptr + potOffset, d_cudaPot.pitch, d_cudaPot.pitch * dysize};
    
	// find terms for laplacian
	Buffer<int2> lapind;
	ddouble lapfac = -0.5 * getLaplacian(lapind, sizeof(BlockPsis), d_evenPsi.pitch, d_evenPsi.slicePitch) / (block_scale * block_scale);
	const uint lapsize = lapind.size() / bsize;
	ddouble lapfac0 = lapsize * (-lapfac);
	
	//std::cout << "lapsize = " << lapsize << ", lapfac = " << lapfac << ", lapfac0 = " << lapfac0 << std::endl;

	// compute time step size
	const uint steps_per_iteration = uint(iteration_period * (maxpot + lapfac0)) + 1; // number of time steps per iteration period
	const ddouble time_step_size = iteration_period / ddouble(steps_per_iteration); // time step in time units
	
	std::cout << "steps_per_iteration = " << steps_per_iteration << std::endl;

	// multiply terms with time_step_size
	g *= time_step_size;
	lapfac *= time_step_size;
	lapfac0 *= time_step_size;
	for(i=0; i<vsize; i++) pot[i] *= time_step_size;
    
    int2* d_lapind;
    checkCudaErrors(cudaMalloc(&d_lapind, lapind.size() * sizeof(int2)));
    
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
	
	cudaPitchedPtr h_cudaEvenPsi = {0};
    cudaPitchedPtr h_cudaOddPsi = {0};
	cudaPitchedPtr h_cudaPot = {0};
	
	h_cudaEvenPsi.ptr = h_evenPsi;
	h_cudaEvenPsi.pitch = dxsize * sizeof(BlockPsis);
	h_cudaEvenPsi.xsize = d_cudaEvenPsi.xsize;
	h_cudaEvenPsi.ysize = d_cudaEvenPsi.ysize;
	
	h_cudaOddPsi.ptr = h_oddPsi;
	h_cudaOddPsi.pitch = dxsize * sizeof(BlockPsis);
	h_cudaOddPsi.xsize = d_cudaOddPsi.xsize;
	h_cudaOddPsi.ysize = d_cudaOddPsi.ysize;
	
	h_cudaPot.ptr = h_pot;
	h_cudaPot.pitch = dxsize * sizeof(BlockPots);
	h_cudaPot.xsize = d_cudaPot.xsize;
	h_cudaPot.ysize = d_cudaPot.ysize;
	
	// Copy from host memory to device memory
	cudaMemcpy3DParms evenPsiParams = {0};
	cudaMemcpy3DParms oddPsiParams = {0};
    cudaMemcpy3DParms potParams = {0};
    
    evenPsiParams.srcPtr = h_cudaEvenPsi;
    evenPsiParams.dstPtr = d_cudaEvenPsi;
    evenPsiParams.extent = psiExtent;
    evenPsiParams.kind = cudaMemcpyHostToDevice;
    
    oddPsiParams.srcPtr = h_cudaOddPsi;
    oddPsiParams.dstPtr = d_cudaOddPsi;
    oddPsiParams.extent = psiExtent;
    oddPsiParams.kind = cudaMemcpyHostToDevice;
    
	potParams.srcPtr = h_cudaPot;
    potParams.dstPtr = d_cudaPot;
    potParams.extent = potExtent;
    potParams.kind = cudaMemcpyHostToDevice;
    
    checkCudaErrors(cudaMemcpy3D(&evenPsiParams));
    checkCudaErrors(cudaMemcpy3D(&oddPsiParams));
    checkCudaErrors(cudaMemcpy3D(&potParams));
    checkCudaErrors(cudaMemcpy(d_lapind, &lapind[0], lapind.size() * sizeof(int2), cudaMemcpyHostToDevice));

	// Clear host memory after data has been copied to devices
	cudaDeviceSynchronize();
	Psi0.clear();
	pot.clear();
	bpos.clear();
	lapind.clear();
	cudaFreeHost(h_oddPsi);
	cudaFreeHost(h_pot);
#if !(SAVE_PICTURE || SAVE_VOLUME)
	cudaFreeHost(h_evenPsi);
#endif

	// Integrate in time
	uint3 dimensions = make_uint3(xsize, ysize, zsize);
	double2 lapfacs = make_double2(lapfac, lapfac0);
	uint iter = 0;
	dim3 dimBlock(THREAD_BLOCK_X, THREAD_BLOCK_Y, THREAD_BLOCK_Z * VALUES_IN_BLOCK);
    dim3 dimGrid((xsize + THREAD_BLOCK_X - 1) / THREAD_BLOCK_X,
				 (ysize + THREAD_BLOCK_Y - 1) / THREAD_BLOCK_Y,
				 (zsize + THREAD_BLOCK_Z - 1) / THREAD_BLOCK_Z);
#if SAVE_PICTURE || SAVE_VOLUME
	cudaMemcpy3DParms evenPsiBackParams = {0};
	evenPsiBackParams.srcPtr = d_cudaEvenPsi;
	evenPsiBackParams.dstPtr = h_cudaEvenPsi;
	evenPsiBackParams.extent = psiExtent;
	evenPsiBackParams.kind = cudaMemcpyDeviceToHost;
#endif
	const uint time0 = clock();
	while(true)
	{
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
		picpath << "kuva" << iter << ".bmp";
		pic.save(picpath.str(), false);
#endif

#if SAVE_VOLUME
		// save volume map
		const ddouble fmax = state.searchFunctionMax();
		const ddouble unit = 60000.0 / (bsize * fmax * fmax);
		Buffer<ushort> vol(dxsize * dysize * dzsize);
		for(k=0; k<dzsize; k++)
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
		saveVolumeMap(volpath.str(), vol, dxsize, dysize, dzsize, block_scale * BLOCK_WIDTH);
#endif

		// finish iteration
		if(++iter > number_of_iterations) break;

		// integrate one iteration
		std::cout << "Iteration " << iter << std::endl;
		for(uint step=0; step<steps_per_iteration; step++)
		{
			// update odd values
			update<<<dimGrid, dimBlock>>>(d_oddPsi, d_evenPsi, d_pot, d_lapind, lapfacs, g, dimensions);
			// update even values
			update<<<dimGrid, dimBlock>>>(d_evenPsi, d_oddPsi, d_pot, d_lapind, lapfacs, g, dimensions);
		}
#if SAVE_PICTURE || SAVE_VOLUME
		// Copy back from device memory to host memory
		checkCudaErrors(cudaMemcpy3D(&evenPsiBackParams));
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
	
	std::cout << "1 GPU version" << std::endl;
	std::cout << "kappa = " << KAPPA << std::endl;
	std::cout << "g = " << G << std::endl;
	std::cout << "ranks = 576" << std::endl;
	std::cout << "block_scale = " << block_scale << std::endl;
	std::cout << "iteration_period = " << iteration_period << std::endl;
	std::cout << "maxr = " << maxr << std::endl;
	std::cout << "maxz = " << maxz << std::endl;

	// integrate in time using DEC
	if(IS_3D) integrateInTime(state, block_scale, Vector3(-maxr, -maxr, -maxz), Vector3(maxr, maxr, maxz), iteration_period, number_of_iterations); // use this for 3d
	else integrateInTime(state, block_scale, Vector3(-maxr, -maxr, 0.0), Vector3(maxr, maxr, 0.0), iteration_period, number_of_iterations); // use this for 2d

	return 0;
}
