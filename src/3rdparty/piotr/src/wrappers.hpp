/*
Copyright (c) 2012, Piotr Dollar
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.
*/

/*******************************************************************************
* Piotr's Computer Vision Matlab Toolbox      Version 3.00
* Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
* Licensed under the Simplified BSD License [see above]
* Project page: http://vision.ucsd.edu/~pdollar/toolbox/doc/
* Original file: https://github.com/pdollar/toolbox/blob/612f9a0451a6abbe2a64768c9e6654692929102e/channels/private/wrappers.hpp

+ author: Klaus Haag: Move external license into this source file
*******************************************************************************/
#ifndef WRAPPERS_HPP_
#define WRAPPERS_HPP_

#ifdef MATLAB_MEX_FILE

// wrapper functions if compiling from Matlab
#include "mex.h"
inline void wrError(const char *errormsg) { mexErrMsgTxt(errormsg); }
inline void* wrCalloc(size_t num, size_t size) { return mxCalloc(num, size); }
inline void* wrMalloc(size_t size) { return mxMalloc(size); }
inline void wrFree(void * ptr) { mxFree(ptr); }

#else

#include <cstdlib>
// wrapper functions if compiling from C/C++
inline void wrError(const char *errormsg) { throw errormsg; }
inline void* wrCalloc(size_t num, size_t size) { return calloc(num, size); }
inline void* wrMalloc(size_t size) { return malloc(size); }
inline void wrFree(void * ptr) { free(ptr); }

#endif

// platform independent aligned memory allocation (see also alFree)
inline void* alMalloc(size_t size, int alignment) {
    const size_t pSize = sizeof(void*);
    const size_t a = alignment - 1;
    void *raw = wrMalloc(size + a + pSize);
    void *aligned = (void*)(((size_t)raw + pSize + a) & ~a);
    *(void**)((size_t)aligned - pSize) = raw;
    return aligned;
}

// platform independent alignned memory de-allocation (see also alMalloc)
inline void alFree(void* aligned) {
    const size_t pSize = sizeof(void*);
    void* raw = *(void**)((char*)aligned - pSize);
    wrFree(raw);
}
#endif
