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

#define PNUM 16

int add(float * x, float * y, float * o, int len)
{
    omp_set_num_threads(PNUM);
// #pragma omp parallel
#pragma omp parallel for

    for(int i = 0; i < len; i++)
    {
        o[i] = x[i] + y[i];
    }
    return 0;
}



int add_8(float * x, float * y, float * o, int len)
{

//     omp_set_num_threads(4);
// // #pragma omp parallel
// #pragma omp parallel for

    omp_set_num_threads(PNUM);
// #pragma omp parallel
#pragma omp parallel for

    for(int i = 0; i < len/8; i++)
    {
        o[i*8] = x[i*8] + y[i*8];
        o[i*8+1] = x[i*8+1] + y[i*8+1];
        o[i*8+2] = x[i*8+2] + y[i*8+2];
        o[i*8+3] = x[i*8+3] + y[i*8+3];
        o[i*8+4] = x[i*8+4] + y[i*8+4];
        o[i*8+5] = x[i*8+5] + y[i*8+5];
        o[i*8+6] = x[i*8+6] + y[i*8+6];
        o[i*8+7] = x[i*8+7] + y[i*8+7];
    }
    return 0;
}


int vec_add(float * x, float * y, float * o, int len)
{
    
    omp_set_num_threads(PNUM);
// #pragma omp parallel
#pragma omp parallel for
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

    omp_set_num_threads(PNUM);
// #pragma omp parallel
#pragma omp parallel for
    for(int i = 0; i < len/16; i++)
    {
        __m256 _a = _mm256_load_ps(x + i*16);
        __m256 _b = _mm256_load_ps(y + i*16);
        __m256 _c = _mm256_load_ps(x + i*16+8);
        __m256 _d = _mm256_load_ps(y + i*16+8);

        __m256 _o_1 = _mm256_add_ps(_a, _b);
        __m256 _o_2 = _mm256_add_ps(_c, _d);

        _mm256_store_ps(o+i*16, _o_1);
        _mm256_store_ps(o+i*16+8, _o_2);

    }
    return 0;
}

int vec_add_4(float * x, float * y, float * o, int len)
{

    omp_set_num_threads(PNUM);
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

        __m256 _o_1 = _mm256_add_ps(_a, _b);
        __m256 _o_2 = _mm256_add_ps(_c, _d);
        __m256 _o_3 = _mm256_add_ps(_e, _f);
        __m256 _o_4 = _mm256_add_ps(_g, _h);

        _mm256_store_ps(o+i*32, _o_1);
        _mm256_store_ps(o+i*32+8, _o_2);
        _mm256_store_ps(o+i*32+16, _o_3);
        _mm256_store_ps(o+i*32+24, _o_4);

    }
    return 0;
}


int vec_add_4_omp(float * x, float * y, float * o, int len)
{

    omp_set_num_threads(PNUM);
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

        __m256 _o_1 = _mm256_add_ps(_a, _b);
        __m256 _o_2 = _mm256_add_ps(_c, _d);
        __m256 _o_3 = _mm256_add_ps(_e, _f);
        __m256 _o_4 = _mm256_add_ps(_g, _h);

        _mm256_store_ps(o+i*32, _o_1);
        _mm256_store_ps(o+i*32+8, _o_2);
        _mm256_store_ps(o+i*32+16, _o_3);
        _mm256_store_ps(o+i*32+24, _o_4);

    }
    return 0;
}


void tes_alloc(float **x, float **y, float **o, int len)
{
    float * tmp_x = aligned_alloc(32, len * sizeof(float));
    float * tmp_y = aligned_alloc(32, len * sizeof(float));
    float * tmp_o = aligned_alloc(32, len * sizeof(float));

    for(int i = 0; i < len; i++)
    {
        // printf("i = %d \n",i);
        tmp_x[i] = 2;
        tmp_y[i] = 1;
    }

    *x = tmp_x;
    *y = tmp_y;
    *o = tmp_o;
}

int main()
{
    struct timeval tv_begin, tv_end;
    long long time;
    int len;

    len = Channel * Width * Height * sizeof(float);

    float* x1, * y1, * o1;
    tes_alloc(&x1, &y1, &o1, len);

    gettimeofday(&tv_begin, NULL);
    add(x1, y1, o1, len);
    gettimeofday(&tv_end, NULL);
    time = 1000000*(tv_end.tv_sec-tv_begin.tv_sec)+tv_end.tv_usec-tv_begin.tv_usec;

    // printf("num = %f \n", o[1]);
    printf("add time = %lld us \n", time);


    float* x2, * y2, * o2;
    tes_alloc(&x2, &y2, &o2, len);
    gettimeofday(&tv_begin, NULL);
    add_8(x2, y2, o2, len);
    gettimeofday(&tv_end, NULL);
    time = 1000000*(tv_end.tv_sec-tv_begin.tv_sec)+tv_end.tv_usec-tv_begin.tv_usec;
    printf("add_8 time = %lld us \n", time);


    float* x3, * y3, * o3;
    tes_alloc(&x3, &y3, &o3, len);
    gettimeofday(&tv_begin, NULL);
    vec_add(x3, y3, o3, len);
    gettimeofday(&tv_end, NULL);
    time = 1000000*(tv_end.tv_sec-tv_begin.tv_sec)+tv_end.tv_usec-tv_begin.tv_usec;
    printf("vec_add time = %lld us \n", time);

    float* x4, * y4, * o4;
    tes_alloc(&x4, &y4, &o4, len);
    gettimeofday(&tv_begin, NULL);
    vec_add_2(x4, y4, o4, len);
    gettimeofday(&tv_end, NULL);
    time = 1000000*(tv_end.tv_sec-tv_begin.tv_sec)+tv_end.tv_usec-tv_begin.tv_usec;
    printf("vec_add_2 time = %lld us \n", time);


    float* x5, * y5, * o5;
    tes_alloc(&x5, &y5, &o5, len);

    gettimeofday(&tv_begin, NULL);
    vec_add_4(x5, y5, o5, len);
    gettimeofday(&tv_end, NULL);
    time = 1000000*(tv_end.tv_sec-tv_begin.tv_sec)+tv_end.tv_usec-tv_begin.tv_usec;
    printf("vec_add_4 time = %lld us \n", time);



    float* x6, * y6, * o6;
    tes_alloc(&x6, &y6, &o6, len);
    gettimeofday(&tv_begin, NULL);
    vec_add_4_omp(x6, y6, o6, len);
    gettimeofday(&tv_end, NULL);
    time = 1000000*(tv_end.tv_sec-tv_begin.tv_sec)+tv_end.tv_usec-tv_begin.tv_usec;
    printf("vec_add_4_omp time = %lld us \n", time);

    // for(int i = 0; i< 8; i++)
    // {
    //     float* x6, * y6, * o6;
    //     tes_alloc(&x6, &y6, &o6, len);
    //     gettimeofday(&tv_begin, NULL);
    //     vec_add_4_omp(x6, y6, o6, len);
    //     gettimeofday(&tv_end, NULL);
    //     time = 1000000*(tv_end.tv_sec-tv_begin.tv_sec)+tv_end.tv_usec-tv_begin.tv_usec;
    //     printf("vec_add_4_omp time = %lld us \n", time);
    // }
    // check();

    return 0;
}