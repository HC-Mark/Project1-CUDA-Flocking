#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

#define CELL_WIDTH_FACTOR 2.0f

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  //initialize numObjects here
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = CELL_WIDTH_FACTOR * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");
  
  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  //these two arrays only use to store cell information, so no need to be as many as numObjects
  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);

  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
    glm::vec3 perceived_center(0.f,0.f,0.f);
    glm::vec3 avoidance_velocity(0.f, 0.f, 0.f);
    glm::vec3 perceived_velocity(0.f, 0.f, 0.f);
    glm::vec3 return_vel(0.f, 0.f, 0.f);
    float neighbor_count_rule1 = 0;
    float neighbor_count_rule3 = 0;

    //pre load all needed data
    glm::vec3 curr_boid_pos = pos[iSelf];
    for (int idx = 0; idx < N; ++idx)
    {
        //if b = boid skip the rest actions
        if (idx == iSelf) continue;
        
        //load current boid pos
        glm::vec3 idx_boid_pos = pos[idx];
        float dist = glm::distance(idx_boid_pos, curr_boid_pos);
        // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
        if (dist < rule1Distance)
        {
            perceived_center += idx_boid_pos;
            neighbor_count_rule1++;
        }
        
        // Rule 2: boids try to stay a distance d away from each other
        if (dist < rule2Distance)
        {
            avoidance_velocity -= (idx_boid_pos - curr_boid_pos);
        }

        // Rule 3: boids try to match the speed of surrounding boids
        if (dist < rule3Distance)
        {
            perceived_velocity += vel[idx];
            neighbor_count_rule3++;
        }
    }

    //if we use N-1, the particles will shrink to the center of cube  -- helped by Hannar
    glm::vec3 rule1_component = glm::vec3(0.f, 0.f, 0.f);
    glm::vec3 rule2_component = glm::vec3(0.f, 0.f, 0.f);
    glm::vec3 rule3_component = glm::vec3(0.f, 0.f, 0.f);
    if (neighbor_count_rule1 > 0)
    {
        rule1_component = (perceived_center / neighbor_count_rule1 - curr_boid_pos) * rule1Scale;
    }
    rule2_component = avoidance_velocity * rule2Scale;
    if (neighbor_count_rule3 > 0)
    {
        rule3_component = (perceived_velocity / neighbor_count_rule3) * rule3Scale;
    }

    //helped by Hanna ReadMe Rule part sum all rules' and current velocity
    return_vel += vel[iSelf] + rule1_component + rule2_component + rule3_component;
  
    return return_vel;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  // Compute a new velocity based on pos and vel1
    //Compute the index of current thread
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index > N)
    {
        return;
    }
    glm::vec3 new_velocity = computeVelocityChange(N, index, pos, vel1);
  // Clamp the speed
    float curr_speed = glm::length(new_velocity);
    //if the total speed of vel is larger than maxSpeed, we normalize the vel and apply the maxSpeed we allow  -- do we need to care negative speed?
    if (curr_speed > maxSpeed)
    {
        new_velocity = glm::normalize(new_velocity) * maxSpeed;
    }
    
  // Record the new velocity into vel2. Question: why NOT vel1? --- because other boids might need that
    vel2[index] = new_velocity;
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    //compute the correspond index in x, y, z axis and use gridIndex3Dto1D to store the actual 1D index  -- why we need inverseCellWidth  -- mult is faster than divide
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx >= N) {
        return;
    }
        //pre store info
        glm::vec3 curr_pos = pos[idx];
        int idx_x = (curr_pos.x - gridMin.x) * inverseCellWidth;
        int idx_y = (curr_pos.y - gridMin.y) * inverseCellWidth;
        int idx_z = (curr_pos.z - gridMin.z) * inverseCellWidth;

        //combine to get the 1D index
        int gridIndex = gridIndex3Dto1D(idx_x, idx_y, idx_z, gridResolution);

        //store to indices and gridIndices correspondingly
        // - Set up a parallel array of integer indices as pointers to the actual
        //   boid data in pos and vel1/vel2
        indices[idx] = idx;
        gridIndices[idx] = gridIndex;
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"
    //what happen for those we don't include any particle? is there an identifier to show its identity?

    //so why do we need th Particles array?  we don't even pass in it -- the arranged start and end indices are for us to access dev_particleArrayIndices, which has been sorted
    //may have better way to do
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx >= N) {
        return;
    }
    int target_grid_index = particleGridIndices[idx];
    //look at the one before and the one after, if diff, store
    //head must be a start
    if (idx == 0)
    {
        gridCellStartIndices[target_grid_index] = idx;
    }

    if (idx == N - 1)
    {
        gridCellEndIndices[target_grid_index] = idx;
        return;
    }


    //check one before and one after
    if (target_grid_index != 0 && target_grid_index != particleGridIndices[idx - 1])
    {
        //start of a cell
        gridCellStartIndices[target_grid_index] = idx;
    
    }

    if (target_grid_index != N - 1 && target_grid_index != particleGridIndices[idx + 1])
    {
        gridCellEndIndices[target_grid_index] = idx;
        
    }

    //int next_grid_index = particleGridIndices[idx + 1];
    //if (target_grid_index != next_grid_index)
    //{
    //    gridCellStartIndices[next_grid_index] = idx + 1;
    //    gridCellEndIndices[target_grid_index] = idx;
    //}

}

//very similar to photon mapping
__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx >= N) {
        return;
    }
    glm::vec3 curr_boid_pos = pos[idx];
    int idx_x = (curr_boid_pos.x - gridMin.x) * inverseCellWidth;
    int idx_y = (curr_boid_pos.y - gridMin.y) * inverseCellWidth;
    int idx_z = (curr_boid_pos.z - gridMin.z) * inverseCellWidth;

    //combine to get the 1D index
    int gridIndex = gridIndex3Dto1D(idx_x, idx_y, idx_z, gridResolution);
  // - Identify which cells may contain neighbors. This isn't always 8.
    //by calculating those cells that interact with the sphere with the neighbor_radius(std::max(std::max(rule1Distance, rule2Distance), rule3Distance))  -not allowed
    //float neighbor_radius = cellWidth / CELL_WIDTH_FACTOR;
    const float neighbor_radius = glm::max(glm::max(rule1Distance, rule2Distance), rule3Distance);
    int max_x = (curr_boid_pos.x + neighbor_radius - gridMin.x) * inverseCellWidth;
    max_x = max_x > gridResolution - 1 ? gridResolution - 1 : max_x;
    int min_x = (curr_boid_pos.x - neighbor_radius - gridMin.x) * inverseCellWidth;
    min_x = min_x < 0 ? 0 : min_x;
    int max_y = (curr_boid_pos.y + neighbor_radius - gridMin.y) * inverseCellWidth;
    max_y = max_y > gridResolution - 1 ? gridResolution - 1 : max_y;
    int min_y = (curr_boid_pos.y + neighbor_radius - gridMin.y) * inverseCellWidth;
    min_y = min_y < 0 ? 0 : min_y;
    int max_z = (curr_boid_pos.z + neighbor_radius - gridMin.z) * inverseCellWidth;
    max_z = max_z > gridResolution - 1 ? gridResolution - 1 : max_z;
    int min_z = (curr_boid_pos.z + neighbor_radius - gridMin.z) * inverseCellWidth;
    min_z = min_z < 0 ? 0 : min_z;

  // - For each cell, read the start/end indices in the boid pointer array.
    glm::vec3 perceived_center(0.f, 0.f, 0.f);
    glm::vec3 avoidance_velocity(0.f, 0.f, 0.f);
    glm::vec3 perceived_velocity(0.f, 0.f, 0.f);
    glm::vec3 new_velocity(0.f, 0.f, 0.f);
    float neighbor_count_rule1 = 0;
    float neighbor_count_rule3 = 0;
    for (int x_cord_idx = min_x; x_cord_idx <= max_x; ++x_cord_idx)
    {
        for (int y_cord_idx = min_y; y_cord_idx <= max_y; ++y_cord_idx)
        {
            for (int z_cord_idx = min_z; z_cord_idx <= max_z; ++z_cord_idx)
            {
                int curr_gridIndex = gridIndex3Dto1D(x_cord_idx, y_cord_idx, z_cord_idx, gridResolution);
                //read the start/end indices
                int start = gridCellStartIndices[curr_gridIndex];
                int end = gridCellEndIndices[curr_gridIndex];
                if (start == -1 || end == -1) {
                    continue; //no boid in this cell
                }
                else
                {
                    // - Access each boid in the cell and compute velocity change from
                    //   the boids rules, if this boid is within the neighborhood distance.
                    for (int boid_array_idx = start; boid_array_idx <= end; ++boid_array_idx)
                    {
                        //the boid_array_idx is only the index in particleArray, need to load it out -- buggy for only few particles moving
                        int boid_idx = particleArrayIndices[boid_array_idx];

                        if (boid_idx == idx) continue;

                        //pre load temp boid pos
                        glm::vec3 idx_boid_pos = pos[boid_idx];
                        float dist = glm::distance(idx_boid_pos, curr_boid_pos);
                        // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
                        if (dist < rule1Distance)
                        {
                            perceived_center += idx_boid_pos;
                            neighbor_count_rule1++;
                        }

                        // Rule 2: boids try to stay a distance d away from each other
                        if (dist < rule2Distance)
                        {
                            avoidance_velocity -= (idx_boid_pos - curr_boid_pos);
                        }

                        // Rule 3: boids try to match the speed of surrounding boids
                        if (dist < rule3Distance)
                        {
                            perceived_velocity += vel1[idx];
                            neighbor_count_rule3++;
                        }
                        
                    }
                
                }
            }
        
        }
    
    }

    //compute the new velocity
    glm::vec3 rule1_component = glm::vec3(0.f, 0.f, 0.f);
    glm::vec3 rule2_component = glm::vec3(0.f, 0.f, 0.f);
    glm::vec3 rule3_component = glm::vec3(0.f, 0.f, 0.f);
    if (neighbor_count_rule1 > 0)
    {
        rule1_component = (perceived_center / neighbor_count_rule1 - curr_boid_pos) * rule1Scale;
    }
    rule2_component = avoidance_velocity * rule2Scale;
    if (neighbor_count_rule3 > 0)
    {
        rule3_component = (perceived_velocity / neighbor_count_rule3) * rule3Scale;
    }
    new_velocity += vel1[idx] + rule1_component + rule2_component + rule3_component;
  // - Clamp the speed change before putting the new speed in vel2
    float curr_speed = glm::length(new_velocity);
    //if the total speed of vel is larger than maxSpeed, we normalize the vel and apply the maxSpeed we allow
    if (curr_speed > maxSpeed)
    {
        new_velocity = glm::normalize(new_velocity) * maxSpeed;
    }

    vel2[idx] = new_velocity;
}



__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
    int gridSize = (numObjects + blockSize - 1) / blockSize; //helped by Gangzheng Tong
    dim3 blocksPerGrid(gridSize);
    //first compute the new velocity
    kernUpdateVelocityBruteForce <<< blocksPerGrid, threadsPerBlock >>> (numObjects, dev_pos, dev_vel1, dev_vel2);

    //Then update the pos
    kernUpdatePos << < blocksPerGrid, threadsPerBlock >> > (numObjects, dt, dev_pos, dev_vel2);

  // TODO-1.2 ping-pong the velocity buffers -- swap content
    cudaMemcpy(dev_vel1, dev_vel2, sizeof(glm::vec3) * numObjects, cudaMemcpyDeviceToDevice);
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:

  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.

    //set up blocks
    int gridSizeParticle = (numObjects + blockSize - 1) / blockSize; //helped by Gangzheng Tong
    dim3 blocksPerGridParticle(gridSizeParticle);
    int gridSizeGridCell = (gridCellCount + blockSize - 1) / blockSize;
    dim3 blocksPerGridGridCell(gridSizeGridCell);
    //call kernel
    kernComputeIndices << < blocksPerGridParticle, threadsPerBlock >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices); //sort by grid index

  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
    //first initialize the two indices list to be -1
    kernResetIntBuffer <<< blocksPerGridGridCell, threadsPerBlock >>> (gridCellCount, dev_gridCellStartIndices, -1);
    kernResetIntBuffer <<< blocksPerGridGridCell, threadsPerBlock >>> (gridCellCount, dev_gridCellEndIndices, -1);

    //then call the kernel to compute those who contain boids
    kernIdentifyCellStartEnd <<< blocksPerGridParticle, threadsPerBlock >>> (numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
  // - Perform velocity updates using neighbor search
    kernUpdateVelNeighborSearchScattered << < blocksPerGridParticle, threadsPerBlock >> > (
        numObjects, gridSideCount, gridMinimum,
        gridInverseCellWidth, gridCellWidth,
        dev_gridCellStartIndices, dev_gridCellEndIndices,
        dev_particleArrayIndices,
        dev_pos, dev_vel1, dev_vel2);
  // - Update positions
    kernUpdatePos << < blocksPerGridParticle, threadsPerBlock >> > (numObjects, dt, dev_pos, dev_vel2);
  // - Ping-pong buffers as needed
    cudaMemcpy(dev_vel1, dev_vel2, sizeof(glm::vec3) * numObjects, cudaMemcpyDeviceToDevice);
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  //2.1 free
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
}

void Boids::unitTest() {
    // LOOK-1.2 Feel free to write additional tests here.

    // test unstable sort
    int *dev_intKeys;
    int *dev_intValues;
    int N = 10;

    std::unique_ptr<int[]>intKeys{ new int[N] };
    std::unique_ptr<int[]>intValues{ new int[N] };

    intKeys[0] = 0; intValues[0] = 0;
    intKeys[1] = 1; intValues[1] = 1;
    intKeys[2] = 0; intValues[2] = 2;
    intKeys[3] = 3; intValues[3] = 3;
    intKeys[4] = 0; intValues[4] = 4;
    intKeys[5] = 2; intValues[5] = 5;
    intKeys[6] = 2; intValues[6] = 6;
    intKeys[7] = 0; intValues[7] = 7;
    intKeys[8] = 5; intValues[8] = 8;
    intKeys[9] = 6; intValues[9] = 9;

    cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
    checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

    cudaMalloc((void**)&dev_intValues, N * sizeof(int));
    checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

    dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

    std::cout << "before unstable sort: " << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << "  key: " << intKeys[i];
        std::cout << " value: " << intValues[i] << std::endl;
    }

    // How to copy data to the GPU
    cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

    // Wrap device vectors in thrust iterators for use with thrust.
    thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
    thrust::device_ptr<int> dev_thrust_values(dev_intValues);
    // LOOK-2.1 Example for using thrust::sort_by_key
    thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

    // How to copy data back to the CPU side from the GPU
    cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
    checkCUDAErrorWithLine("memcpy back failed!");

    std::cout << "after unstable sort: " << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << "  key: " << intKeys[i];
        std::cout << " value: " << intValues[i] << std::endl;
    }

    // cleanup
    cudaFree(dev_intKeys);
    cudaFree(dev_intValues);
    checkCUDAErrorWithLine("cudaFree failed!");
    return;
}