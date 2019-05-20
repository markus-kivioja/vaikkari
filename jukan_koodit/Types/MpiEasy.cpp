#include "MpiEasy.hpp"
#include <mpi.h>
#include <iostream>

const uint MAXSIZE = 4000;

void initMPI()
{
	MPI_Init(NULL, NULL);
}

void finalizeMPI()
{
	MPI_Finalize();
}

uint getMPIrank()
{
	uint rank;
	MPI_Comm_rank(MPI_COMM_WORLD, (int*)&rank);
	return rank;
}

uint getMPIranks()
{
	uint ranks;
	MPI_Comm_size(MPI_COMM_WORLD, (int*)&ranks);
	return ranks;
}

void sendMPI(void *data, const uint size, const uint rank, const uint tag)
{
	uint j;
//	for(j=0; j+MAXSIZE<size; j+=MAXSIZE) MPI_Send(&((char *)data)[j], MAXSIZE, MPI_CHAR, rank, tag + j/TAGDIVIDOR, MPI_COMM_WORLD);
//	MPI_Send(&((char *)data)[j], size - j, MPI_CHAR, rank, tag + j/TAGDIVIDOR, MPI_COMM_WORLD);
	for(j=0; j+MAXSIZE<size; j+=MAXSIZE) MPI_Send(&((char *)data)[j], MAXSIZE, MPI_CHAR, rank, tag, MPI_COMM_WORLD);
	MPI_Send(&((char *)data)[j], size - j, MPI_CHAR, rank, tag, MPI_COMM_WORLD);
}

void recvMPI(void *data, const uint size, const uint rank, const uint tag)
{
	uint j;
//	for(j=0; j+MAXSIZE<size; j+=MAXSIZE) MPI_Recv(&((char *)data)[j], MAXSIZE, MPI_CHAR, rank, tag + j/TAGDIVIDOR, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//	MPI_Recv(&((char *)data)[j], size - j, MPI_CHAR, rank, tag + j/TAGDIVIDOR, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	for(j=0; j+MAXSIZE<size; j+=MAXSIZE) MPI_Recv(&((char *)data)[j], MAXSIZE, MPI_CHAR, rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&((char *)data)[j], size - j, MPI_CHAR, rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void sumMPI(uint &sum)
{
	uint send = sum;
	MPI_Reduce(&send, &sum, 1, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Bcast(&sum, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

/*	uint rank, ranks;
	MPI_Comm_rank(MPI_COMM_WORLD, (int*)&rank);
	MPI_Comm_size(MPI_COMM_WORLD, (int*)&ranks);
	if(rank != 0)
	{
		MPI_Send(&sum, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
		MPI_Recv(&sum, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	else
	{
		uint i;
		for(i=1; i<ranks; i++)
		{
			uint s = 0;
			MPI_Recv(&s, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			sum += s;
		}
		for(i=1; i<ranks; i++)
		{
			MPI_Send(&sum, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
		}
	}
*/}

void sumMPI(ullong &sum)
{
	ullong send = sum;
	MPI_Reduce(&send, &sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Bcast(&sum, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
}

void sumMPI(ddouble &sum)
{
	ddouble send = sum;
	MPI_Reduce(&send, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Bcast(&sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

/*	uint rank, ranks;
	MPI_Comm_rank(MPI_COMM_WORLD, (int*)&rank);
	MPI_Comm_size(MPI_COMM_WORLD, (int*)&ranks);
	if(rank != 0)
	{
		MPI_Send(&sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		MPI_Recv(&sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	else
	{
		uint i;
		for(i=1; i<ranks; i++)
		{
			ddouble s = 0.0;
			MPI_Recv(&s, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			sum += s;
		}
		for(i=1; i<ranks; i++)
		{
			MPI_Send(&sum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		}
	}
*/}

void sumMPI(Complex &sum)
{
	Buffer<ddouble> send(2);
	send[0] = sum.r;
	send[1] = sum.i;
	sumMPI(send);
	sum.r = send[0];
	sum.i = send[1];

/*	uint rank, ranks;
	MPI_Comm_rank(MPI_COMM_WORLD, (int*)&rank);
	MPI_Comm_size(MPI_COMM_WORLD, (int*)&ranks);
	if(rank != 0)
	{
		MPI_Send(&sum, 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		MPI_Recv(&sum, 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	else
	{
		uint i;
		for(i=1; i<ranks; i++)
		{
			Complex s(0.0, 0.0);
			MPI_Recv(&s, 2, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			sum += s;
		}
		for(i=1; i<ranks; i++)
		{
			MPI_Send(&sum, 2, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		}
	}
*/}

void sumMPI(Vector3 &sum)
{
	Buffer<ddouble> send(3);
	send[0] = sum.x;
	send[1] = sum.y;
	send[2] = sum.z;
	sumMPI(send);
	sum.x = send[0];
	sum.y = send[1];
	sum.z = send[2];
/*
	uint rank, ranks;
	MPI_Comm_rank(MPI_COMM_WORLD, (int*)&rank);
	MPI_Comm_size(MPI_COMM_WORLD, (int*)&ranks);
	if(rank != 0)
	{
		MPI_Send(&sum, 3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		MPI_Recv(&sum, 3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	else
	{
		uint i;
		for(i=1; i<ranks; i++)
		{
			Vector3 s(0.0, 0.0, 0.0);
			MPI_Recv(&s, 3, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			sum += s;
		}
		for(i=1; i<ranks; i++)
		{
			MPI_Send(&sum, 3, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		}
	}
*/}

void sumMPI(Vector4 &sum)
{
	Buffer<ddouble> send(4);
	send[0] = sum.x;
	send[1] = sum.y;
	send[2] = sum.z;
	send[3] = sum.t;
	sumMPI(send);
	sum.x = send[0];
	sum.y = send[1];
	sum.z = send[2];
	sum.t = send[3];
}

void sumMPI(Buffer<ddouble> &sum)
{
	Buffer<ddouble> send = sum;
	MPI_Reduce(&send[0], &sum[0], sum.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Bcast(&sum[0], sum.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

/*	uint rank, ranks;
	MPI_Comm_rank(MPI_COMM_WORLD, (int*)&rank);
	MPI_Comm_size(MPI_COMM_WORLD, (int*)&ranks);
	if(rank != 0)
	{
		sendMPI(&sum[0], sum.size() * sizeof(ddouble), 0, 0);
		recvMPI(&sum[0], sum.size() * sizeof(ddouble), 0, 0);
	}
	else
	{
		uint i, j;
		Buffer<ddouble> s(sum.size());
		for(i=1; i<ranks; i++)
		{
			recvMPI(&s[0], sum.size() * sizeof(ddouble), i, 0);
			for(j=0; j<sum.size(); j++) sum[j] += s[j];
		}
		for(i=1; i<ranks; i++)
		{
			sendMPI(&sum[0], sum.size() * sizeof(ddouble), i, 0);
		}
	}
*/}

void minMPI(uint &sum)
{
	uint send = sum;
	MPI_Reduce(&send, &sum, 1, MPI_UNSIGNED, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Bcast(&sum, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
}

void maxMPI(uint &sum)
{
	uint send = sum;
	MPI_Reduce(&send, &sum, 1, MPI_UNSIGNED, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Bcast(&sum, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
}

void minMPI(ddouble &sum)
{
	ddouble send = sum;
	MPI_Reduce(&send, &sum, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Bcast(&sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

/*	uint rank, ranks;
	MPI_Comm_rank(MPI_COMM_WORLD, (int*)&rank);
	MPI_Comm_size(MPI_COMM_WORLD, (int*)&ranks);
	if(rank != 0)
	{
		MPI_Send(&sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		MPI_Recv(&sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	else
	{
		uint i;
		for(i=1; i<ranks; i++)
		{
			ddouble s = 0.0;
			MPI_Recv(&s, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(s < sum) sum = s;
		}
		for(i=1; i<ranks; i++)
		{
			MPI_Send(&sum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		}
	}
*/}

void maxMPI(ddouble &sum)
{
	ddouble send = sum;
	MPI_Reduce(&send, &sum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Bcast(&sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

/*	uint rank, ranks;
	MPI_Comm_rank(MPI_COMM_WORLD, (int*)&rank);
	MPI_Comm_size(MPI_COMM_WORLD, (int*)&ranks);
	if(rank != 0)
	{
		MPI_Send(&sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		MPI_Recv(&sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	else
	{
		uint i;
		for(i=1; i<ranks; i++)
		{
			ddouble s = 0.0;
			MPI_Recv(&s, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(s > sum) sum = s;
		}
		for(i=1; i<ranks; i++)
		{
			MPI_Send(&sum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		}
	}
*/}

void barrierMPI()
{
	MPI_Barrier(MPI_COMM_WORLD);
}

