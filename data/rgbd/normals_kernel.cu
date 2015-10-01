


// addressing 3D matrix: [(idx+w*idy)*3, (idx+w*idy)*3+1, (idx+w*idy)*3+2]
__global__ void depth2normals(float* x, float* n, int w, int h)
{
  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  const int idy = threadIdx.y + blockIdx.y*blockDim.y;

  __shared__ float data[KERNEL_RADIUS + KERNEL_RADIUS]

  if ((idx<w)&&(idy<h)){
    n[(idx+w*idy)*3+1] = float(idx)/float(w);
    //n[(idx+w*idy)*3] = float(idy)/float(h);
  }
  
}
