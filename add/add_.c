#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include "omp.h"
#include <immintrin.h>


#define Width   224
#define Height  224
#define Channel 30



int add(float * x, float * y, float * o, int len)
{

    for(int i = 0; i < len; i++)
    {
        x[i] = x[i] + y[i];
    }
    return 0;
}



int add_8(float * x, float * y, float * o, int len)
{

//     omp_set_num_threads(4);
// // #pragma omp parallel
// #pragma omp parallel for

    for(int i = 0; i < len/8; i++)
    {
        x[i*8] = x[i*8] + y[i*8];
        x[i*8+1] = x[i*8+1] + y[i*8+1];
        x[i*8+2] = x[i*8+2] + y[i*8+2];
        x[i*8+3] = x[i*8+3] + y[i*8+3];
        x[i*8+4] = x[i*8+4] + y[i*8+4];
        x[i*8+5] = x[i*8+5] + y[i*8+5];
        x[i*8+6] = x[i*8+6] + y[i*8+6];
        x[i*8+7] = x[i*8+7] + y[i*8+7];
    }
    return 0;
}


int vec_add(float * x, float * y, float * o, int len)
{
    
    for(int i = 0; i < len/8; i++)

    
    {
        __m256 _a = _mm256_load_ps(x + i*8);
        __m256 _b = _mm256_load_ps(y + i*8);

        __m256 _c = _mm256_add_ps(_a, _b);

        _mm256_store_ps(o+i*8, _c);

    }
    return 0;
}


int vec_add_2(float * x, float * y, float * o, int len)
{
    
    for(int i = 0; i < len/16; i++)
    {
        __m256 _a = _mm256_load_ps(x + i*16);
        __m256 _b = _mm256_load_ps(y + i*16);
        __m256 _c = _mm256_load_ps(x + i*16+8);
        __m256 _d = _mm256_load_ps(y + i*16+8);

        _a = _mm256_add_ps(_a, _b);
        _c = _mm256_add_ps(_c, _d);

        _mm256_store_ps(x+i*16, _a);
        _mm256_store_ps(x+i*16+8, _c);

    }
    return 0;
}

int vec_add_4(float * x, float * y, float * o, int len)
{
    
    for(int i = 0; i < len/32; i++)
    {
        __m256 _a = _mm256_load_ps(x + i*32);
        __m256 _b = _mm256_load_ps(y + i*32);
        __m256 _c = _mm256_load_ps(x + i*32+8);
        __m256 _d = _mm256_load_ps(y + i*32+8);
        __m256 _e = _mm256_load_ps(x + i*32+16);
        __m256 _f = _mm256_load_ps(y + i*32+16);
        __m256 _g = _mm256_load_ps(x + i*32+24);
        __m256 _h = _mm256_load_ps(y + i*32+24);

        _a = _mm256_add_ps(_a, _b);
        _c = _mm256_add_ps(_c, _d);
        _e = _mm256_add_ps(_e, _f);
        _g = _mm256_add_ps(_g, _h);

        _mm256_store_ps(x+i*32, _a);
        _mm256_store_ps(x+i*32+8, _c);
        _mm256_store_ps(x+i*32+16, _e);
        _mm256_store_ps(x+i*32+24, _g);

    }
    return 0;
}


int vec_add_4_omp(float * x, float * y, float * o, int len)
{

    omp_set_num_threads(4);
// #pragma omp parallel
#pragma omp parallel for
    for(int i = 0; i < len/32; i++)
    {
        __m256 _a = _mm256_load_ps(x + i*32);
        __m256 _b = _mm256_load_ps(y + i*32);
        __m256 _c = _mm256_load_ps(x + i*32+8);
        __m256 _d = _mm256_load_ps(y + i*32+8);
        __m256 _e = _mm256_load_ps(x + i*32+16);
        __m256 _f = _mm256_load_ps(y + i*32+16);
        __m256 _g = _mm256_load_ps(x + i*32+24);
        __m256 _h = _mm256_load_ps(y + i*32+24);

        _a = _mm256_add_ps(_a, _b);
        _c = _mm256_add_ps(_c, _d);
        _e = _mm256_add_ps(_e, _f);
        _g = _mm256_add_ps(_g, _h);

        _mm256_store_ps(x+i*32, _a);
        _mm256_store_ps(x+i*32+8, _c);
        _mm256_store_ps(x+i*32+16, _e);
        _mm256_store_ps(x+i*32+24, _g);
    }
    return 0;
}


int main()
{
    struct timeval tv_begin, tv_end;
    long long time;
    int len;

    len = Channel * Width * Height * sizeof(float);

    float * x = aligned_alloc(32, len * sizeof(float));
    float * y = aligned_alloc(32, len * sizeof(float));
    float * o = aligned_alloc(32, len * sizeof(float));

    for(int i = 0; i < len; i++)
    {
        x[i] = i;
        y[i] = i;
    }


    gettimeofday(&tv_begin, NULL);
    add(x, y, o, len);
    gettimeofday(&tv_end, NULL);
    time = 1000000*(tv_end.tv_sec-tv_begin.tv_sec)+tv_end.tv_usec-tv_begin.tv_usec;

    // printf("num = %f \n", o[1]);
    printf("add time = %lld us \n", time);


    gettimeofday(&tv_begin, NULL);
    add_8(x, y, o, len);
    gettimeofday(&tv_end, NULL);
    time = 1000000*(tv_end.tv_sec-tv_begin.tv_sec)+tv_end.tv_usec-tv_begin.tv_usec;
    printf("add_8 time = %lld us \n", time);


    gettimeofday(&tv_begin, NULL);
    vec_add(x, y, o, len);
    gettimeofday(&tv_end, NULL);
    time = 1000000*(tv_end.tv_sec-tv_begin.tv_sec)+tv_end.tv_usec-tv_begin.tv_usec;
    printf("vec_add time = %lld us \n", time);


    gettimeofday(&tv_begin, NULL);
    vec_add_2(x, y, o, len);
    gettimeofday(&tv_end, NULL);
    time = 1000000*(tv_end.tv_sec-tv_begin.tv_sec)+tv_end.tv_usec-tv_begin.tv_usec;
    printf("vec_add_2 time = %lld us \n", time);


    gettimeofday(&tv_begin, NULL);
    vec_add_4(x, y, o, len);
    gettimeofday(&tv_end, NULL);
    time = 1000000*(tv_end.tv_sec-tv_begin.tv_sec)+tv_end.tv_usec-tv_begin.tv_usec;
    printf("vec_add_4 time = %lld us \n", time);

    gettimeofday(&tv_begin, NULL);
    vec_add_4_omp(x, y, o, len);
    gettimeofday(&tv_end, NULL);
    time = 1000000*(tv_end.tv_sec-tv_begin.tv_sec)+tv_end.tv_usec-tv_begin.tv_usec;
    printf("vec_add_4_omp time = %lld us \n", time);

    return 0;
}