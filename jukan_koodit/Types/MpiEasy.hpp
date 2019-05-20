#ifndef _MPIEASY_H_INCLUDED_
#define _MPIEASY_H_INCLUDED_

#include "Types.hpp"
#include "Buffer.hpp"

void initMPI();
void finalizeMPI();
uint getMPIrank();
uint getMPIranks();
void sendMPI(void *data, const uint size, const uint rank, const uint tag);
void recvMPI(void *data, const uint size, const uint rank, const uint tag);
void sumMPI(uint &sum);
void sumMPI(ullong &sum);
void sumMPI(ddouble &sum);
void sumMPI(Complex &sum);
void sumMPI(Vector3 &sum);
void sumMPI(Vector4 &sum);
void sumMPI(Buffer<ddouble> &sum);
void minMPI(uint &sum);
void maxMPI(uint &sum);
void minMPI(ddouble &sum);
void maxMPI(ddouble &sum);
void barrierMPI();

#endif //_MPIEASY_H_INCLUDED_
