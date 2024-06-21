#pragma once

#include "util.h"

using namespace std;

#define max2(m, n) ((m) > (n) ? (m) : (n))
#define max3(m, n, p) ((m) > (n) ? ((m) > (p) ? (m) : (p)) : ((n) > (p) ? (n) : (p)))
#define MASK5(v) (v & 0b11111)
#define END 0
#define TOP 1
#define LEFT 2
#define DIAG 3

// mutex mu1;

typedef struct
{
    int x; // top
    int y; // left
    int m; // left top
} record;

inline char get_char(const char *s, size_t offset)
{
    size_t n_bit = offset * 5;
    return MASK5((unsigned)((*((uint16_t *)&(s[n_bit >> 3]))) >> (n_bit & 7)));
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
    }
    
    direct_matrix(int num, int bw, size_t q_len):width(bw*2 + 1),height(q_len+1), thread_num(num){
        data = (int*) malloc(sizeof(int) * width * height * num);
        memset(data,0,width * height * num * sizeof(int));
    }

private:
    int width, height, thread_num;
    int* data;
};
void generate_report(SWResult *res, const char* q, const char* c);
// void smith_waterman(const char *q, const char *c, const size_t *q_idxs, const size_t *q_lens, const size_t *c_idxs, const size_t *c_lens, size_t num_task, vector<SWResult> &res);

// void banded_smith_waterman(const char *q, const char *c, vector<uint32_t>& q_idxs, vector<uint32_t>& q_lens, vector<size_t>& diags, size_t c_len, size_t num_task, vector<SWResult> &res, ThreadPool* pool, vector<future<int>>& rs);

void smith_waterman_kernel(const int idx, SWResult *res, SWTasks* sw_task);
void cpu_kernel(const int idx, SWResult *res, 
                const char *q, const char* c, 
                size_t c_len, uint32_t q_idx, uint32_t n,
                uint32_t diag, const int band_width);

void gasal_run(SWTasks tasks, vector<SWResult> res[][NUM_STREAM],const char* q_dev, const char* s_dev, int num_g, int span);