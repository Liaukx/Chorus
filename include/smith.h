#pragma once

#include "util.h"
#include <cuda_runtime.h>
#include "hash_table.h"

using namespace std;

#define max2(m, n) ((m) > (n) ? (m) : (n))
#define max3(m, n, p) ((m) > (n) ? ((m) > (p) ? (m) : (p)) : ((n) > (p) ? (n) : (p)))
#define MASK5(v) (v & 0b11111)

#define calIndex(i,j,w) ((i+1)*(w)+(j+1))
#define calIndex3D(i,j,k,X,Y,Z) ((i+1)*(Y)*(Z)+(j+1)*(Z) + (k))
#define calTop(i,j,w) ((i)*(w)+(j+2))
#define calLeft(i,j,w) ((i+1)*(w)+(j))
#define calDiag(i,j,w) ((i)*(w)+(j+1))

#define END 0
#define TOP 1
#define LEFT 2
#define DIAG 3


#define MaxAlignLen 3000
#define MaxQueryLen 20000
#define TILE_SIZE 2
#define MaxBW 25
#define BatchSize 2048
#define MaxNumBatch 20
// mutex mu1;

typedef struct
{
    int x; // top
    int y; // left
    int m; // left top
} record;

typedef struct
{
    int score;
    int qlen;
    int clen;
    size_t q_res_d[MaxAlignLen];
    size_t s_res_d[MaxAlignLen];
} SWResult_d;

__device__ inline char get_char_d(const char *s, size_t offset) {
    offset *= 5;
    const unsigned char *p = (const unsigned char *)(s + (offset >> 3));
    return MASK5((unsigned)(((uint16_t)p[0] | ((uint16_t)p[1] << 8)) >> (offset & 7)));
}


class record_matrix{
public:
    record_matrix(int bw, int TileSize):width(2*bw+1),height(TileSize + 1){
        data = (record*) malloc(sizeof(record) * height * width);
        if (data == nullptr)
        {
            printf("CPU out of memory!\n");
            exit(-1);
            return;
        }
        memset(data,0,sizeof(record) * width*height);
    
    }
    const record& get_top(int _q,int _c){
        return data[_q * width + _c + 2];
    }
    const record& get_left(int _q,int _c){
        return data[(_q+1) * width + _c];
    }
    const record& get_diag(int _q,int _c){
        return data[_q * width + _c+1];
    }
    void reset(){
        memcpy(data,data + (height - 1) * width ,width * sizeof(record));
        memset(data + width, 0, (height - 1) *width * sizeof(record));
    }
    record& operator() (int _q, int _c){
        return data[(_q+1) *width + (_c+1)];
    }
    ~record_matrix(){
        free(data);
    }
    record* data;
private:
    int width,height;
};

class direct_matrix{
public:
    direct_matrix(int bw, size_t q_len):width(bw*2 + 1),height(q_len+1){
        data = (int*) malloc(sizeof(int) * width * height);
        memset(data,0,width * height * sizeof(int));
        if (data == nullptr)
        {
            printf("CPU out of memory!\n");
            exit(-1);
            return;
        }
    }
    
    direct_matrix(int num, int bw, size_t q_len):width(bw*2 + 1),height(q_len+1), thread_num(num){
        data = (int*) malloc(sizeof(int) * width * height * num);
        memset(data,0,width * height * num * sizeof(int));
    }

    void assign(size_t _q, size_t _c,int val){
        data[_c+1 * height + _q+1] = val;
    }
    int get(size_t _q, size_t _c){
        return data[_c+1 * height + _q+1];
    }
    ~direct_matrix(){
        free(data);
    }

private:
    int width, height, thread_num;
    int* data;
};
void cigar_to_index(size_t idx, int begin, int* cigar_len, char* cigar_op, int* cigar_cnt,
               size_t* q_start,
               size_t* c_start,
               std::vector<SWResult>& res_s);

void cigar_to_index_and_report(size_t idx, int begin, int* cigar_len, char* cigar_op, int* cigar_cnt,
               size_t* q_start,
               size_t* c_start,
               std::vector<SWResult>& res_s,
            //    uint32_t* num_task,
               int* score, Task* task, const char* q, const char* c);
void generate_report(size_t idx, int begin, std::vector<SWResult>& res_s, int* score, Task* task, const char* q, const char* c);
void generate_report(SWResult *res, const char* q, const char* c);
// void smith_waterman(const char *q, const char *c, const size_t *q_idxs, const size_t *q_lens, const size_t *c_idxs, const size_t *c_lens, size_t num_task, vector<SWResult> &res);

// void banded_smith_waterman(const char *q, const char *c, vector<uint32_t>& q_idxs, vector<uint32_t>& q_lens, vector<size_t>& diags, size_t c_len, size_t num_task, vector<SWResult> &res, ThreadPool* pool, vector<future<int>>& rs);

void smith_waterman_kernel(const int idx, SWResult *res, SWTasks* sw_task);
void cpu_kernel (SWResult *res, 
                const char *q, const char* c, 
                size_t c_len, 
                uint32_t q_idx, uint32_t n,
                uint32_t diag, const int band_width);


void cpu_kernel(SWResult *res, 
                const char *q, const char* c, 
                size_t c_len, 
                int64_t c_begin, int64_t c_end,
                uint32_t q_begin, uint32_t q_len,
                const int band_width);

void gasal_run(SWTasks tasks, vector<SWResult> res[][NUM_STREAM],const char* q_dev, const char* s_dev, int num_g, int span);

void banded_sw_cpu_kernel(
                int num_task,
                uint32_t* q_lens, uint32_t* q_idxs, Task* task,
                const char* query, const char* target, size_t target_len,
                int * max_score,
                size_t* q_end_idx, size_t* s_end_idx,
                char* cigar_op, int* cigar_cnt,int* cigar_len,
                int *direct_matrix, record* tile_matrix,int band_width,
                const int* BLOSUM62);