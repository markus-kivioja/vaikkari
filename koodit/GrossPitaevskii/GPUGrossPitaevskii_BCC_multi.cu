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

ddouble potentialV3(const Vector3& p)
{
	return 0.5 * (p.x * p.x + p.y * p.y + RATIO * RATIO * p.z * p.z);
}

bool saveVolumeMap(const std::string& path, const Buffer<ushort>& vol, const uint xsize, const uint ysize, const uint zsize, const Vector3& h)
{
	Text rawpath;
	rawpath << path << ".raw";

	// save raw
	std::ofstream fs(rawpath.str().c_str(), std::ios_base::binary | std::ios::trunc);
	if (fs.fail()) return false;
	fs.write((char*)&vol[0], 2 * xsize * ysize * zsize);
	fs.close();

	// save header
	Text text;

	text << "ObjectType              = Image" << std::endl;
	text << "NDims                   = 3" << std::endl;
	text << "BinaryData              = True" << std::endl;
	text << "CompressedData          = False" << std::endl;
	text << "BinaryDataByteOrderMSB  = False" << std::endl;
	text << "TransformMatrix         = 1 0 0 0 1 0 0 0 1" << std::endl;
	text << "Offset                  = " << -0.5 * xsize * h.x << " " << -0.5 * ysize * h.y << " " << -0.5 * zsize * h.z << std::endl;
	text << "CenterOfRotation        = 0 0 0" << std::endl;
	text << "DimSize                 = " << xsize << " " << ysize << " " << zsize << std::endl;
	text << "ElementSpacing          = " << h.x << " " << h.y << " " << h.z << std::endl;
	text << "ElementNumberOfChannels = 1" << std::endl;
	text << "ElementType             = MET_USHORT" << std::endl;
	text << "ElementDataFile         = " << rawpath.str() << std::endl;
	text.save(path);
	return true;
}

// bcc grid
const Vector3 BLOCK_WIDTH = sqrt(8.0) * Vector3(1, 1, 1); // dimensions of unit block
const ddouble VOLUME = sqrt(32.0 / 9.0); // volume of body elements
const bool IS_3D = true; // 3-dimensional
void getPositions(Buffer<Vector3>& pos)
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
ddouble getLaplacian(Buffer<int2>& ind, const int nx, const int ny, const int nz) // nx, ny, nz in bytes
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
inline __host__ __device__ void operator+=(double2& a, double2 b)
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
	double2 sum = (*(BlockPsis*)(prevPsi + ldsLapInd[face].x)).values[ldsLapInd[face++].y];
	sum += (*(BlockPsis*)(prevPsi + ldsLapInd[face].x)).values[ldsLapInd[face++].y];
	sum += (*(BlockPsis*)(prevPsi + ldsLapInd[face].x)).values[ldsLapInd[face++].y];
	sum += (*(BlockPsis*)(prevPsi + ldsLapInd[face].x)).values[ldsLapInd[face++].y];

	double2 prev = (*(BlockPsis*)prevPsi).values[dualNodeId];
	double normsq = prev.x * prev.x + prev.y * prev.y;
	sum = lapfacs.x * sum + (lapfacs.y + (*pot).values[dualNodeId] + g * normsq) * prev;

	(*nextPsi).values[dualNodeId] += make_double2(sum.y, -sum.x);
};

uint integrateInTime(const VortexState& state, const ddouble block_scale, const Vector3& minp, const Vector3& maxp, const ddouble iteration_period, const uint number_of_iterations, uint gpuCount)
{
	uint i, j, k, l;

	// find dimensions
	const Vector3 domain = maxp - minp;
	const uint xsize = uint(domain.x / (block_scale * BLOCK_WIDTH.x)) + 1;
	const uint ysize = uint(domain.y / (block_scale * BLOCK_WIDTH.y)) + 1;
	const uint zsize = uint(domain.z / (block_scale * BLOCK_WIDTH.z)) + 1;
	const Vector3 p0 = 0.5 * (minp + maxp - block_scale * Vector3(BLOCK_WIDTH.x * xsize, BLOCK_WIDTH.y * ysize, BLOCK_WIDTH.z * zsize));

	std::vector<uint> zSizes(gpuCount);
	uint zRemainder = zsize % gpuCount;
	for (uint gpuIdx = 0; gpuIdx < gpuCount; ++gpuIdx)
	{
		zSizes[gpuIdx] = zsize / gpuCount;
		if (zRemainder)
		{
			zSizes[gpuIdx]++;
			zRemainder--;
		}
	}

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
	Buffer<Complex> Psi0(vsize, Complex(0, 0)); // initial discrete wave function
	Buffer<ddouble> pot(vsize, 0.0); // discrete potential multiplied by time step size
	ddouble g = state.getG(); // effective interaction strength
	ddouble maxpot = 0.0; // maximal value of potential
	for (k = 0; k < zsize; k++)
	{
		for (j = 0; j < ysize; j++)
		{
			for (i = 0; i < xsize; i++)
			{
				for (l = 0; l < bsize; l++)
				{
					const uint ii = ii0 + k * bxysize + j * bxsize + i * bsize + l;
					const Vector3 p(p0.x + block_scale * (i * BLOCK_WIDTH.x + bpos[l].x), p0.y + block_scale * (j * BLOCK_WIDTH.y + bpos[l].y), p0.z + block_scale * (k * BLOCK_WIDTH.z + bpos[l].z)); // position
					Psi0[ii] = state.getPsi(p);
					pot[ii] = potentialV3(p);
					const ddouble poti = pot[ii] + g * Psi0[ii].normsq();
					if (poti > maxpot) maxpot = poti;
				}
			}
		}
	}

	Buffer<int2> dummyLapind;
	ddouble lapfac = -0.5 * getLaplacian(dummyLapind, 0, 0, 0) / (block_scale * block_scale);
	const uint lapsize = dummyLapind.size() / bsize;
	ddouble lapfac0 = lapsize * (-lapfac);

	// compute time step size
	const uint steps_per_iteration = uint(iteration_period * (maxpot + lapfac0)) + 1; // number of time steps per iteration period
	const ddouble time_step_size = iteration_period / ddouble(steps_per_iteration); // time step in time units

	std::cout << "steps_per_iteration = " << steps_per_iteration << std::endl;

	// multiply terms with time_step_size
	g *= time_step_size;
	lapfac *= time_step_size;
	lapfac0 *= time_step_size;
	for (i = 0; i < vsize; i++) pot[i] *= time_step_size;

	// Initialize host memory
	size_t dxsize = xsize + 2; // One element buffer to both ends
	size_t dysize = ysize + 2;
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
	for (k = 0; k < zsize; k++)
	{
		for (j = 0; j < ysize; j++)
		{
			for (i = 0; i < xsize; i++)
			{
				for (l = 0; l < bsize; l++)
				{
					const uint srcI = ii0 + k * bxysize + j * bxsize + i * bsize + l;
					const uint dstI = (k + 1) * dxsize * dysize + (j + 1) * dxsize + (i + 1);
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

	// Initialize device memory
	std::vector<cudaPitchedPtr> d_cudaEvenPsis(gpuCount);
	std::vector<cudaPitchedPtr> d_cudaOddPsis(gpuCount);
	std::vector<cudaPitchedPtr> d_cudaPots(gpuCount);
	std::vector<PitchedPtr> d_evenPsis(gpuCount);
	std::vector<PitchedPtr> d_oddPsis(gpuCount);
	std::vector<PitchedPtr> d_pots(gpuCount);
	std::vector<int2*> d_lapinds(gpuCount);
	std::vector<cudaPitchedPtr> h_cudaEvenPsis(gpuCount);
	std::vector<cudaPitchedPtr> h_cudaOddPsis(gpuCount);
	std::vector<cudaPitchedPtr> h_cudaPots(gpuCount);
	std::vector<cudaExtent> psiExtents(gpuCount);

	std::vector<size_t> dzSizes(gpuCount);
	for (uint gpuIdx = 0; gpuIdx < gpuCount; ++gpuIdx)
	{
		dzSizes[gpuIdx] = zSizes[gpuIdx] + 2;

		psiExtents[gpuIdx] = make_cudaExtent(dxsize * sizeof(BlockPsis), dysize, dzSizes[gpuIdx]);
		cudaExtent potExtent = make_cudaExtent(dxsize * sizeof(BlockPots), dysize, dzSizes[gpuIdx]);

		cudaSetDevice(gpuIdx);
		for (uint peerGpu = 0; peerGpu < gpuCount; ++peerGpu)
		{
			cudaDeviceEnablePeerAccess(peerGpu, 0);
		}
		checkCudaErrors(cudaMalloc3D(&d_cudaEvenPsis[gpuIdx], psiExtents[gpuIdx]));
		checkCudaErrors(cudaMalloc3D(&d_cudaOddPsis[gpuIdx], psiExtents[gpuIdx]));
		checkCudaErrors(cudaMalloc3D(&d_cudaPots[gpuIdx], potExtent));

		// Offsets are for the zero valued padding on the edges, offset = z + y + x in bytes
		size_t offset = d_cudaEvenPsis[gpuIdx].pitch * dysize + d_cudaEvenPsis[gpuIdx].pitch + sizeof(BlockPsis);
		size_t potOffset = d_cudaPots[gpuIdx].pitch * dysize + d_cudaPots[gpuIdx].pitch + sizeof(BlockPots);
		PitchedPtr d_evenPsi = { (char*)d_cudaEvenPsis[gpuIdx].ptr + offset, d_cudaEvenPsis[gpuIdx].pitch, d_cudaEvenPsis[gpuIdx].pitch * dysize };
		PitchedPtr d_oddPsi = { (char*)d_cudaOddPsis[gpuIdx].ptr + offset, d_cudaOddPsis[gpuIdx].pitch, d_cudaOddPsis[gpuIdx].pitch * dysize };
		PitchedPtr d_pot = { (char*)d_cudaPots[gpuIdx].ptr + potOffset, d_cudaPots[gpuIdx].pitch, d_cudaPots[gpuIdx].pitch * dysize };
		d_evenPsis[gpuIdx] = d_evenPsi;
		d_oddPsis[gpuIdx] = d_oddPsi;
		d_pots[gpuIdx] = d_pot;

		// find terms for laplacian
		Buffer<int2> lapind;
		getLaplacian(lapind, sizeof(BlockPsis), d_evenPsis[gpuIdx].pitch, d_evenPsis[gpuIdx].slicePitch);

		checkCudaErrors(cudaMalloc(&d_lapinds[gpuIdx], lapind.size() * sizeof(int2)));

		bool first = (gpuIdx == 0);

		h_cudaEvenPsis[gpuIdx].ptr = first ? h_evenPsi : ((BlockPsis*)h_cudaEvenPsis[gpuIdx - 1].ptr) + dxsize * dysize * (dzSizes[gpuIdx - 1] - 2);
		h_cudaEvenPsis[gpuIdx].pitch = dxsize * sizeof(BlockPsis);
		h_cudaEvenPsis[gpuIdx].xsize = d_cudaEvenPsis[gpuIdx].xsize;
		h_cudaEvenPsis[gpuIdx].ysize = d_cudaEvenPsis[gpuIdx].ysize;

		h_cudaOddPsis[gpuIdx].ptr = first ? h_oddPsi : ((BlockPsis*)h_cudaOddPsis[gpuIdx - 1].ptr) + dxsize * dysize * (dzSizes[gpuIdx - 1] - 2);
		h_cudaOddPsis[gpuIdx].pitch = dxsize * sizeof(BlockPsis);
		h_cudaOddPsis[gpuIdx].xsize = d_cudaOddPsis[gpuIdx].xsize;
		h_cudaOddPsis[gpuIdx].ysize = d_cudaOddPsis[gpuIdx].ysize;

		h_cudaPots[gpuIdx].ptr = first ? h_pot : ((BlockPots*)h_cudaPots[gpuIdx - 1].ptr) + dxsize * dysize * (dzSizes[gpuIdx - 1] - 2);
		h_cudaPots[gpuIdx].pitch = dxsize * sizeof(BlockPots);
		h_cudaPots[gpuIdx].xsize = d_cudaPots[gpuIdx].xsize;
		h_cudaPots[gpuIdx].ysize = d_cudaPots[gpuIdx].ysize;

		// Copy from host memory to device memory
		cudaMemcpy3DParms evenPsiParams = { 0 };
		cudaMemcpy3DParms oddPsiParams = { 0 };
		cudaMemcpy3DParms potParams = { 0 };

		evenPsiParams.srcPtr = h_cudaEvenPsis[gpuIdx];
		evenPsiParams.dstPtr = d_cudaEvenPsis[gpuIdx];
		evenPsiParams.extent = psiExtents[gpuIdx];
		evenPsiParams.kind = cudaMemcpyHostToDevice;

		oddPsiParams.srcPtr = h_cudaOddPsis[gpuIdx];
		oddPsiParams.dstPtr = d_cudaOddPsis[gpuIdx];
		oddPsiParams.extent = psiExtents[gpuIdx];
		oddPsiParams.kind = cudaMemcpyHostToDevice;

		potParams.srcPtr = h_cudaPots[gpuIdx];
		potParams.dstPtr = d_cudaPots[gpuIdx];
		potParams.extent = potExtent;
		potParams.kind = cudaMemcpyHostToDevice;

		checkCudaErrors(cudaMemcpy3DAsync(&evenPsiParams));
		checkCudaErrors(cudaMemcpy3DAsync(&oddPsiParams));
		checkCudaErrors(cudaMemcpy3DAsync(&potParams));
		checkCudaErrors(cudaMemcpyAsync(d_lapinds[gpuIdx], &lapind[0], lapind.size() * sizeof(int2), cudaMemcpyHostToDevice));

		cudaDeviceSynchronize();
		lapind.clear();
	}

	// Clear host memory after data has been copied to devices
	Psi0.clear();
	pot.clear();
	bpos.clear();
	cudaFreeHost(h_oddPsi);
	cudaFreeHost(h_pot);
#if !(SAVE_PICTURE || SAVE_VOLUME)
	cudaFreeHost(h_evenPsi);
#endif

	// Integrate in time
	dim3 dimBlock(THREAD_BLOCK_X, THREAD_BLOCK_Y, THREAD_BLOCK_Z * VALUES_IN_BLOCK);
	std::vector<dim3> dimGrids(gpuCount);
	std::vector<uint3> dimensions(gpuCount);
	for (uint gpuIdx = 0; gpuIdx < gpuCount; ++gpuIdx)
	{
		dimensions[gpuIdx] = make_uint3(xsize, ysize, zSizes[gpuIdx]);
		dimGrids[gpuIdx] = dim3((xsize + THREAD_BLOCK_X - 1) / THREAD_BLOCK_X,
			(ysize + THREAD_BLOCK_Y - 1) / THREAD_BLOCK_Y,
			(zSizes[gpuIdx] + THREAD_BLOCK_Z - 1) / THREAD_BLOCK_Z);
	}

	std::vector<cudaMemcpy3DParms> evenMemcpiesFrom(gpuCount - 1, { 0 });
	std::vector<cudaMemcpy3DParms> evenMemcpiesTo(gpuCount - 1, { 0 });
	std::vector<cudaMemcpy3DParms> oddMemcpiesFrom(gpuCount - 1, { 0 });
	std::vector<cudaMemcpy3DParms> oddMemcpiesTo(gpuCount - 1, { 0 });
	
	cudaExtent oneSliceExtent = make_cudaExtent(dxsize * sizeof(BlockPsis), dysize, 1);
	for (uint gpuIdx = 0; gpuIdx < gpuCount - 1; ++gpuIdx)
	{
		cudaPitchedPtr evenFromSrc = d_cudaEvenPsis[gpuIdx];
		cudaPitchedPtr evenFromDst = d_cudaEvenPsis[gpuIdx + 1];
		cudaPitchedPtr evenToSrc = d_cudaEvenPsis[gpuIdx + 1];
		cudaPitchedPtr evenToDst = d_cudaEvenPsis[gpuIdx];
	
		cudaPitchedPtr oddFromSrc = d_cudaOddPsis[gpuIdx];
		cudaPitchedPtr oddFromDst = d_cudaOddPsis[gpuIdx + 1];
		cudaPitchedPtr oddToSrc = d_cudaOddPsis[gpuIdx + 1];
		cudaPitchedPtr oddToDst = d_cudaOddPsis[gpuIdx];
	
		evenFromSrc.ptr = ((char*)evenFromSrc.ptr) + d_evenPsis[gpuIdx].slicePitch * (dzSizes[gpuIdx] - 2);
		evenToDst.ptr = ((char*)evenToDst.ptr) + d_evenPsis[gpuIdx].slicePitch * (dzSizes[gpuIdx] - 1);
		evenToSrc.ptr = ((char*)evenToSrc.ptr) + d_evenPsis[gpuIdx + 1].slicePitch * 1;
		evenFromDst.ptr = ((char*)evenFromDst.ptr) + d_evenPsis[gpuIdx + 1].slicePitch * 0;
	
		oddFromSrc.ptr = ((char*)oddFromSrc.ptr) + d_oddPsis[gpuIdx].slicePitch * (dzSizes[gpuIdx] - 2);
		oddToDst.ptr = ((char*)oddToDst.ptr) + d_oddPsis[gpuIdx].slicePitch * (dzSizes[gpuIdx] - 1);
		oddToSrc.ptr = ((char*)oddToSrc.ptr) + d_oddPsis[gpuIdx + 1].slicePitch * 1;
		oddFromDst.ptr = ((char*)oddFromDst.ptr) + d_oddPsis[gpuIdx + 1].slicePitch * 0;

		evenMemcpiesFrom[gpuIdx].srcPtr = evenFromSrc;
		evenMemcpiesFrom[gpuIdx].dstPtr = evenFromDst;
		evenMemcpiesFrom[gpuIdx].extent = oneSliceExtent;
		evenMemcpiesFrom[gpuIdx].kind = cudaMemcpyDefault;

		evenMemcpiesTo[gpuIdx].srcPtr = evenToSrc;
		evenMemcpiesTo[gpuIdx].dstPtr = evenToDst;
		evenMemcpiesTo[gpuIdx].extent = oneSliceExtent;
		evenMemcpiesTo[gpuIdx].kind = cudaMemcpyDefault;

		oddMemcpiesFrom[gpuIdx].srcPtr = oddFromSrc;
		oddMemcpiesFrom[gpuIdx].dstPtr = oddFromDst;
		oddMemcpiesFrom[gpuIdx].extent = oneSliceExtent;
		oddMemcpiesFrom[gpuIdx].kind = cudaMemcpyDefault;

		oddMemcpiesTo[gpuIdx].srcPtr = oddToSrc;
		oddMemcpiesTo[gpuIdx].dstPtr = oddToDst;
		oddMemcpiesTo[gpuIdx].extent = oneSliceExtent;
		oddMemcpiesTo[gpuIdx].kind = cudaMemcpyDefault;
	}

	struct StreamsAndEvents
	{
		cudaStream_t backwardsStream;
		cudaStream_t forwardsStream;
		cudaEvent_t backwardsEvent;
		cudaEvent_t forwardsEvent;
	};
	std::vector<StreamsAndEvents> streamAndEvents(gpuCount);
	for (uint gpuIdx = 0; gpuIdx < gpuCount; ++gpuIdx)
	{
		cudaSetDevice(gpuIdx);

		cudaStreamCreate(&streamAndEvents[gpuIdx].backwardsStream);
		cudaEventCreate(&streamAndEvents[gpuIdx].backwardsEvent);
		cudaEventRecord(streamAndEvents[gpuIdx].backwardsEvent, streamAndEvents[gpuIdx].backwardsStream);

		cudaStreamCreate(&streamAndEvents[gpuIdx].forwardsStream);
		cudaEventCreate(&streamAndEvents[gpuIdx].forwardsEvent);
		cudaEventRecord(streamAndEvents[gpuIdx].forwardsEvent, streamAndEvents[gpuIdx].forwardsStream);
	}

#if SAVE_PICTURE || SAVE_VOLUME
	std::vector<cudaMemcpy3DParms> evenPsiBackParams(gpuCount, { 0 });
	for (uint gpuIdx = 0; gpuIdx < gpuCount; ++gpuIdx)
	{
		d_cudaEvenPsis[gpuIdx].ptr = ((char*)d_cudaEvenPsis[gpuIdx].ptr) + d_evenPsis[gpuIdx].slicePitch;
		h_cudaEvenPsis[gpuIdx].ptr = ((BlockPsis*)h_cudaEvenPsis[gpuIdx].ptr) + dxsize * dysize;
		psiExtents[gpuIdx].depth -= 2;

		evenPsiBackParams[gpuIdx].srcPtr = d_cudaEvenPsis[gpuIdx];
		evenPsiBackParams[gpuIdx].dstPtr = h_cudaEvenPsis[gpuIdx];
		evenPsiBackParams[gpuIdx].extent = psiExtents[gpuIdx];
		evenPsiBackParams[gpuIdx].kind = cudaMemcpyDeviceToHost;
	}
#endif
	double2 lapfacs = make_double2(lapfac, lapfac0);

	const uint time0 = clock();
	uint iter = 0;
	while (true)
	{
#if SAVE_PICTURE || SAVE_VOLUME
		//cudaDeviceSynchronize();
#endif

#if SAVE_PICTURE
		// draw picture
		Picture pic(dxsize, dysize);
		k = zsize / 2 + 1;
		for (j = 0; j < dysize; j++)
		{
			for (i = 0; i < dxsize; i++)
			{
				const uint idx = k * dxsize * dysize + j * dxsize + i;
				double norm = sqrt(h_evenPsi[idx].values[0].x * h_evenPsi[idx].values[0].x + h_evenPsi[idx].values[0].y * h_evenPsi[idx].values[0].y);

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
		for (k = 0; k < (zsize + 2); k++)
		{
			for (j = 0; j < dysize; j++)
			{
				for (i = 0; i < dxsize; i++)
				{
					const uint idx = k * dxsize * dysize + j * dxsize + i;
					ddouble sum = 0.0;
					for (l = 0; l < bsize; l++)
					{
						sum += h_evenPsi[idx].values[0].x * h_evenPsi[idx].values[0].x + h_evenPsi[idx].values[0].y * h_evenPsi[idx].values[0].y;
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
		if (++iter > number_of_iterations) break;

		// integrate one iteration
		std::cout << "Iteration " << iter << std::endl;
		for (uint step = 0; step < steps_per_iteration; step++)
		{
			// update odd values
			for (uint gpuIdx = 0; gpuIdx < gpuCount; ++gpuIdx)
			{
				cudaSetDevice(gpuIdx);
				if (gpuIdx < gpuCount - 1)
					cudaStreamWaitEvent(streamAndEvents[gpuIdx].forwardsStream, streamAndEvents[gpuIdx + 1].backwardsEvent, 0);
				if (gpuIdx > 0)
					cudaStreamWaitEvent(streamAndEvents[gpuIdx].forwardsStream, streamAndEvents[gpuIdx - 1].forwardsEvent, 0);
				update << <dimGrids[gpuIdx], dimBlock, 0, streamAndEvents[gpuIdx].forwardsStream >> > (d_oddPsis[gpuIdx], d_evenPsis[gpuIdx], d_pots[gpuIdx], d_lapinds[gpuIdx], lapfacs, g, dimensions[gpuIdx]);
			}

			for (uint gpuIdx = 0; gpuIdx < gpuCount; ++gpuIdx)
			{
				cudaSetDevice(gpuIdx);
				if (gpuIdx < gpuCount - 1)
				{
					cudaMemcpy3DAsync(&oddMemcpiesFrom[gpuIdx], streamAndEvents[gpuIdx].forwardsStream);
					cudaEventRecord(streamAndEvents[gpuIdx].forwardsEvent, streamAndEvents[gpuIdx].forwardsStream);
				}
				if (gpuIdx > 0)
				{
					cudaMemcpy3DAsync(&oddMemcpiesTo[gpuIdx - 1], streamAndEvents[3].backwardsStream);
					cudaEventRecord(streamAndEvents[3].backwardsEvent, streamAndEvents[3].backwardsStream);
				}
			}

			// update even values
			for (uint gpuIdx = 0; gpuIdx < gpuCount; ++gpuIdx)
			{
				cudaSetDevice(gpuIdx);
				if (gpuIdx < gpuCount - 1)
					cudaStreamWaitEvent(streamAndEvents[gpuIdx].forwardsStream, streamAndEvents[gpuIdx + 1].backwardsEvent, 0);
				if (gpuIdx > 0)
					cudaStreamWaitEvent(streamAndEvents[gpuIdx].forwardsStream, streamAndEvents[gpuIdx - 1].forwardsEvent, 0);
				update << <dimGrids[gpuIdx], dimBlock, 0, streamAndEvents[gpuIdx].forwardsStream >> > (d_evenPsis[gpuIdx], d_oddPsis[gpuIdx], d_pots[gpuIdx], d_lapinds[gpuIdx], lapfacs, g, dimensions[gpuIdx]);
			}

			for (uint gpuIdx = 0; gpuIdx < gpuCount; ++gpuIdx)
			{
				cudaSetDevice(gpuIdx);
				if (gpuIdx < gpuCount - 1)
				{
					cudaMemcpy3DAsync(&evenMemcpiesFrom[gpuIdx], streamAndEvents[gpuIdx].forwardsStream);
					cudaEventRecord(streamAndEvents[gpuIdx].forwardsEvent, streamAndEvents[gpuIdx].forwardsStream);
				}
				if (gpuIdx > 0)
				{
					cudaMemcpy3DAsync(&evenMemcpiesTo[gpuIdx - 1], streamAndEvents[3].backwardsStream);
					cudaEventRecord(streamAndEvents[3].backwardsEvent, streamAndEvents[3].backwardsStream);
				}
			}
		}
#if SAVE_PICTURE || SAVE_VOLUME
		// Copy back from device memory to host memory
		for (uint gpuIdx = 0; gpuIdx < gpuCount; ++gpuIdx)
		{
			cudaSetDevice(gpuIdx);
			checkCudaErrors(cudaMemcpy3DAsync(&evenPsiBackParams[gpuIdx]));
		}
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

int main(int argc, char** argv)
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
	if (IS_3D) state0.setRange(0.0, 15.0, 35.0, 0.2, 0.2); // use this for 3d
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
	if (IS_3D) state.setRange(minr, maxr, maxz, 0.03, 0.03); // use this for 3d
	else state.setRange(minr, maxr, 1.0, 0.03, 1.0); // use this for 2d
	state.initialize(state0);
	state.iterateSolution(potentialRZ, 10000, 1e-29);
	state.save("state.dat");
	maxr = state.searchMaxR(eps);
	maxz = state.searchMaxZ(eps);
	//std::cout << "maxf=" << state.searchFunctionMax() << std::endl;
#endif
	uint gpuCount = (argc > 1) ? std::stoi(argv[1]) : 4;

	const int number_of_iterations = 10;
	const ddouble iteration_period = 0.1;
	const ddouble block_scale = PIx2 / (20.0 * sqrt(state.integrateCurvature()));

	std::cout << "4 GPUs version pasimysiini" << std::endl;
	std::cout << "kappa = " << KAPPA << std::endl;
	std::cout << "g = " << G << std::endl;
	std::cout << "block_scale = " << block_scale << std::endl;
	std::cout << "iteration_period = " << iteration_period << std::endl;
	std::cout << "maxr = " << maxr << std::endl;
	std::cout << "maxz = " << maxz << std::endl;

	// integrate in time using DEC
	if (IS_3D) integrateInTime(state, block_scale, Vector3(-maxr, -maxr, -maxz), Vector3(maxr, maxr, maxz), iteration_period, number_of_iterations, gpuCount); // use this for 3d
	else integrateInTime(state, block_scale, Vector3(-maxr, -maxr, 0.0), Vector3(maxr, maxr, 0.0), iteration_period, number_of_iterations, gpuCount); // use this for 2d

	return 0;
}
