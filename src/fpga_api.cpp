#include"fpga_api.h"
#include<stdio.h>
#include<fcntl.h>
#include<unistd.h>
#include<sys/mman.h>
#include<cstring>
#include<time.h>

#define min(x,y) (((x)<(y))?(x):(y))
double time_accum = 0.0;

FPGA::FPGA(off_t data_addr, off_t output_addr, int m_size, int v_size)
{
  m_size_ = m_size;
  v_size_ = v_size;

  m1_size_ = v_size * v_size;

  data_size_ = (m_size_+1)*v_size_; // fpga bram data size
  data_size_M = (2*v_size_)*v_size_*sizeof(float);

  fd_ = open("/dev/mem", O_RDWR);
  data_M = static_cast<float*>(mmap(NULL, data_size_M, PROT_READ|PROT_WRITE, MAP_SHARED, fd_, data_addr));
  data_ = new float[data_size_];	

  output_ = static_cast<unsigned int*>(mmap(NULL, sizeof(unsigned int), PROT_READ|PROT_WRITE, MAP_SHARED,fd_, output_addr));
  output_MV = new unsigned int[m_size_];
  // output_M = static_cast<unsigned int*>(NULL);

  num_block_call_ = 0;
}

FPGA::~FPGA()
{
  munmap(data_M, data_size_M);
  munmap(output_, sizeof(unsigned int));
  close(fd_);

  delete[] data_;
  printf("total hardware time cost: %f\n", time_accum/CLOCKS_PER_SEC);
}

float* FPGA::matrix(void)
{
  return data_ + v_size_;
}

float* FPGA::vector(void)
{
  return data_;
}

float* FPGA::matrix_M1(void)
{
  return data_M;
}

float* FPGA::matrix_M2(void)
{
  return data_M + m1_size_;
}

void FPGA::reset(void)
{
  num_block_call_ = 0;
}

int FPGA::num_block_call(void)
{
  return num_block_call_;
}

const float* FPGA::blockMV()
{
  num_block_call_ += 1;

  // cpu version
  float* vec = this->vector();
  float* mat = this->matrix();
  float* out  = reinterpret_cast<float*>(output_MV);  

  for(int i = 0; i < m_size_; ++i)
  {
    out[i] = 0;
    for(int j = 0; j < v_size_; ++j)
      out[i] += vec[j] * mat[v_size_*i + j];
  }

  for(int i = 0; i < m_size_; ++i)
    data_[i] = out[i];

  return data_;    
}

const float* __attribute__((optimize("O0"))) FPGA::blockMM()
{
  num_block_call_ += 1;

  clock_t start = clock();
  // fpga version
  *output_ = 0x5555;
  while(*output_ == 0x5555);

  clock_t end = clock();

  time_accum += (double)(end-start);
  return data_M;    
}

void FPGA::largeMV(const float* large_mat, const float* input, float* output, int num_input, int num_output)
{
  float* vec = this->vector();
  float* mat = this->matrix();

  // 0) Initialize output vector		
  for(int i = 0; i < num_output; ++i)
    output[i] = 0;

  for(int i = 0; i < num_output; i += m_size_)
  {
    for(int j = 0; j < num_input; j += v_size_)
    {			
      // 0) Initialize input vector
      int block_row = min(m_size_, num_output-i);
      int block_col = min(v_size_, num_input-j);

      // 1) Assign a vector
      for(int k=0; k<v_size_; k++){
        if(k<block_col) vec[k] = input[j+k];
        else vec[k] = 0;
      }

      // 2) Assign a matrix
      for(int r1=0; r1<m_size_; r1++){
        for(int c1=0; c1<v_size_; c1++){
          if(r1<block_row && c1<block_col) mat[r1*v_size_+c1] = large_mat[(r1+i) * num_input + (c1+j)];
	  else mat[r1*v_size_+c1] = 0;
	}
      }     
      // 2) Assign a matrix

      // 3) Call a function `blockMV() to execute MV multiplication
      const float* ret = this->blockMV();

      // 4) Accumulate intermediate results
      for(int row = 0; row < block_row; ++row)
        output[i + row] += ret[row];
    } 
  }
}

void FPGA::largeMM(const float* weight_mat, const float* input_mat, float* output, int num_input, int num_output, int num_matrix2)
{
  float* m1 = this->matrix_M1();
  float* m2 = this->matrix_M2();

  // 0) Initialize output vector		
  for(int i = 0; i < num_output*num_matrix2; ++i)
    output[i] = 0;

  for(int i = 0; i < num_output; i += v_size_)
  {
    for(int j = 0; j < num_input; j += v_size_)
    {			
      for(int k = 0; k < num_matrix2; k += v_size_)
      {
        // 0) Initialize input vector
        int block_row = min(v_size_, num_output-i);
        int block_col_1 = min(v_size_, num_input-j);
        int block_col_2 = min(v_size_, num_matrix2-k);

        // 1) Assign a m1
        // IMPLEMENT THIS
	
	for(int r1=0; r1<v_size_; r1++){
	  for(int c1=0; c1<v_size_; c1++){
	    if(r1<block_row && c1<block_col_1) m1[r1*v_size_ + c1] = weight_mat[(r1+i) * num_input+(c1 +j)];
	    else m1[r1*v_size_+c1] = 0;
	  }
	}

        // 2) Assign a m2
        // IMPLEMENT THIS
	for(int r1=0; r1<v_size_; r1++){
	  for(int c1=0; c1<v_size_; c1++){
	    if(r1 < block_col_1 && c1<block_col_2) m2[r1*v_size_+c1] = input_mat[(r1+j) * num_matrix2 + (c1+k)];
	    else m2[r1*v_size_+c1] = 0;
	  }
	}

        // 3) Call a function `blockMM() to execute Matrix matrix multiplication
        const float* ret = this->blockMM();

        // 4) Accumulate intermediate results
        // IMPLEMENT THIS
        // 4) Accumulate intermediate results
        for(int n = 0; n<block_row; ++n)
        {
          for(int m = 0; m<block_col_2; ++m)
          {
            output[(i + n) + (k + m)*num_output] += ret[n*v_size_ + m];
          }
        }
      }
    } 
  }
}

void FPGA::convLowering(const std::vector<std::vector<std::vector<std::vector<float>>>>& cnn_weights,
    std::vector<std::vector<float>>& new_weights,
    const std::vector<std::vector<std::vector<float>>>& inputs,
    std::vector<std::vector<float>>& new_inputs) {
  /*
   * Arguments:
   *
   * conv_weights: [conv_channel, input_channel, conv_height, conv_width]
   * new_weights: [?, ?]
   * inputs: [input_channel, input_height, input_width]
   * new_inputs: [?, ?]
   *
   */

  int conv_channel = cnn_weights.size();
  int input_channel = cnn_weights[0].size();
  int conv_height = cnn_weights[0][0].size();
  int conv_width = cnn_weights[0][0][0].size();
  //int input_channel = cnn_weights.size();
  int input_height = inputs[0].size();
  int input_width = inputs[0][0].size();

  // IMPLEMENT THIS
  // For example,
  // new_weights[0][0] = cnn_weights[0][0][0][0];
  // new_inputs[0][0] = inputs[0][0][0];
  for(int i = 0; i < input_channel; i++){
      for(int j = 0; j < conv_height; j++){
            for(int k = 0; k < conv_width; k++){
		for(int l = 0; l < conv_channel; l++){
		  new_weights[l][i*conv_height*conv_width + j*conv_width + k] = cnn_weights[l][i][j][k];}

		for(int n = 0; n < input_height-conv_height+1; n++){
		  for(int m = 0; m < input_width-conv_width+1; m++){
			 new_inputs[i*conv_height*conv_width + j*conv_width + k][n * (input_width-conv_width+1) + m] = inputs[i][n+j][m+k];}}
								  	}
						      }
			        }
}
