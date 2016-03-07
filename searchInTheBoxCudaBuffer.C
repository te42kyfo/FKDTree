#define MAX_SIZE 128
#define NUM_DIMENSIONS 3
#define MAX_RESULT_SIZE 256
#define RANGE 0.2f;
#define BLOCKSIZE 1024
#include "cuda.h"
#include <stdlib.h>

__global__ typedef struct
{
    
    unsigned int data[MAX_SIZE];
    unsigned int front;
    unsigned int tail;
    unsigned int size;
} Queue;
__global__ bool push_back(Queue* queue, unsigned int index)
{
    if (queue->size < MAX_SIZE)
    {
        queue->data[queue->tail] = index;
        queue->tail = (queue->tail + 1) % MAX_SIZE;
        queue->size++;
        return true;
    }
    return false;
    
}

__global__ unsigned int pop_front(Queue* queue)
{
    if (queue->size > 0)
    {
        unsigned int element = queue->data[queue->front];
        queue->front = (queue->front + 1) % MAX_SIZE;
        queue->size--;
        return element;
    }
}

__global__ void erase_first_n_elements(Queue* queue, unsigned int n)
{
    unsigned int elementsToErase = queue->size - n > 0 ? n : queue->size;
    queue->size -=elementsToErase;
    queue->front = (queue->front + elementsToErase) % MAX_SIZE;
    
}


__global__ unsigned int leftSonIndex(unsigned int index)
{
    return 2 * index + 1;
}


__global__ unsigned int rightSonIndex(unsigned int index)
{
    return 2 * index + 2;
}


__global__ bool intersects(unsigned int index,  float* theDimensions, unsigned int nPoints,
                           float* minPoint, float* maxPoint, int dimension)
{
    return (theDimensions[nPoints * dimension + index] <= maxPoint[dimension]
            && theDimensions[nPoints * dimension + index] >= minPoint[dimension]);
}


__global__ bool isInTheBox(unsigned int index,  float* theDimensions, unsigned int nPoints,
                           float* minPoint, float* maxPoint)
{
    bool inTheBox = true;
    for (int i = 0; i < NUM_DIMENSIONS; ++i)
    {
        inTheBox &= (theDimensions[nPoints * i + index] <= maxPoint[i]
                     && theDimensions[nPoints * i + index] >= minPoint[i]);
    }
    
    return inTheBox;
}


__global__ void CUDASearchInTheKDBox(unsigned int nPoints,  float* dimensions,  unsigned int* ids,  unsigned int* results)
{
    
    // Global Thread ID
    unsigned int point_index = blockIdx.x*blockDim.x+threadIdx.x;
    
    //	float range = 0.1f;
    if(point_index < nPoints)
    {
        
        int theDepth = floor(log2((float)nPoints));
        float minPoint[NUM_DIMENSIONS];
        float maxPoint[NUM_DIMENSIONS];
        for(int i = 0; i<NUM_DIMENSIONS; ++i)
        {
            minPoint[i] = dimensions[nPoints*i+point_index] - RANGE;
            maxPoint[i] = dimensions[nPoints*i+point_index] + RANGE;
        }
        
        Queue indecesToVisit;
        indecesToVisit.front = indecesToVisit.tail =indecesToVisit.size =0;
        unsigned int pointsFound=0;
        unsigned int resultIndex = nPoints + MAX_RESULT_SIZE*point_index;
        push_back(&indecesToVisit, 0);
        
        for (int depth = 0; depth < theDepth + 1; ++depth)
        {
            int dimension = depth % NUM_DIMENSIONS;
            unsigned int numberOfIndecesToVisitThisDepth =
            indecesToVisit.size;
            
            for (unsigned int visitedIndecesThisDepth = 0;
                 visitedIndecesThisDepth < numberOfIndecesToVisitThisDepth;
                 visitedIndecesThisDepth++)
            {
                
                //				unsigned int index = indecesToVisit.data[(indecesToVisit.front+visitedIndecesThisDepth)% MAX_SIZE];
                unsigned int index = pop_front(&indecesToVisit);
                //				if(point_index == 0)
                //				{
                //					printf("index: %d, dimensions: %f %f %f\n", index, dimensions[index], dimensions[nPoints+index], dimensions[2*nPoints+index]);
                //				}
                
                bool intersection = intersects(index,dimensions, nPoints, minPoint, maxPoint,
                                               dimension);
                
                if(intersection && isInTheBox(index, dimensions, nPoints, minPoint, maxPoint))
                {
                    if(pointsFound < MAX_RESULT_SIZE)
                    {
                        //						if(point_index == 0)
                        //						{
                        //							printf("index: %d added to results", index);
                        //						}
                        
                        results[resultIndex] = index;
                        resultIndex++;
                        pointsFound++;
                    }
                    
                }
                
                bool isLowerThanBoxMin = dimensions[nPoints*dimension + index]
                < minPoint[dimension];
                int startSon = isLowerThanBoxMin; //left son = 0, right son =1
                
                int endSon = isLowerThanBoxMin || intersection;
                
                for (int whichSon = startSon; whichSon < endSon + 1; ++whichSon)
                {
                    unsigned int indexToAdd = leftSonIndex(index) + whichSon;
                    
                    if (indexToAdd < nPoints)
                    {
                        push_back(&indecesToVisit,indexToAdd);
                        
                        if(indecesToVisit.size == MAX_SIZE)
                            printf("queue limit hit");
                    }
                }
            }
            
            //			erase_first_n_elements(&indecesToVisit,numberOfIndecesToVisitThisDepth );
        }
        
        results[point_index] = pointsFound;
        
    }
    
}

void CUDAKernelWrapper(unsigned int nPoints,float *h_dim,unsigned int *h_ids,unsigned int *h_results)
{
    
    // Device vectors
    float *d_dim;
    unsigned int *d_ids;
    unsigned int *d_results

    // Allocate device memory
    cudaMalloc(&d_dim, 3*nPoints * sizeof(float));
    cudaMalloc(&d_ids, nPoints * sizeof(unsigned int));
    cudaMalloc(&d_results, (nPoints + nPoints * maxResultSize)
               * sizeof(unsigned int));
    
    // Copy host vectors to device
    cudaMemcpy( d_dim, h_dim, 3*nPoints * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy( d_ids, h_ids, nPoints * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_results, h_results, (nPoints + nPoints * maxResultSize)
               * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Number of thread blocks in grid
    unsigned int gridSize = (int)ceil((float)n/BLOCKSIZE);
    
    // Execute the kernel
    CUDASearchInTheKDBox<<<gridSize, BLOCKSIZE>>>(nPoints, d_dim, d_ids,d_results);
    
    // Copy array back to host
    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );
    
    // Sum up vector c and print result divided by n, this should equal 1 within error
    double sum = 0;
    for(i=0; i<n; i++)
        sum += h_c[i];
    printf("final result: %f\n", sum/n);
    
    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
    
}

