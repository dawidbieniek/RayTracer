#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include "cuda_runtime.h"
#include <iostream>

//void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
//    if (result) {
//        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
//            file << ":" << line << " '" << func << "' \n";
//        // Make sure we call CUDA Device Reset before exiting
//        cudaDeviceReset();
//        exit(99);
//    }
//}
//
//#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
#endif // CUDA_HELPERS_H