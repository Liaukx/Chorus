#include "qit.h"
#include "hash_table.h"
#include "smith.h"
#include "util.h"
#include "query_group.h"
#include "output.h"
#include <nvml.h>

#include <mutex>
#include <condition_variable>


struct TimeProfile 
{
    double mem_time = 0;
    double gpu_time = 0;
    double cpu_time = 0;
    double name_time = 0;
};


typedef struct{
    int num_task;
    int band_width;
    size_t begin;
    std::vector<SWResult>& res_s;
    int threadsPerBlock;  // 根据 shared memory 限制调整
    int blocks;
    // device
    uint32_t* q_lens_d;
    uint32_t* q_idxs_d;
    Task* task_d;

    const char* query_d;
    const char* target_d;
    size_t target_len_d;

    int * max_score_d;

    size_t* q_end_idx_d;
    size_t* s_end_idx_d;

    char* cigar_op_d;
    int* cigar_cnt_d;
    int* cigar_len_d;

    int *rd_d;
    record* rt_d;

    int* BLOSUM62_d;

    // host
    uint32_t* q_lens_h;
    uint32_t* q_idxs_h;
    Task* task_h;

    const char* query_h;
    const char* target_h;
    size_t target_len_h;

    int * max_score_h;

    size_t* q_end_idx_h;
    size_t* s_end_idx_h;

    char* cigar_op_h;
    int* cigar_cnt_h;
    int* cigar_len_h;

    int *rd_h;
    record* rt_h;

    const int* BLOSUM62_h;


    // bool is_completed;
    cudaStream_t& stream;
    // cudaEvent_t& kernels_done;
    // cudaStream_t& copy_stream;
    // cudaEvent_t& copies_done;
    // cudaEvent_t &event;
} banded_sw_task;


void blastp(string argv_query, vector<string> argv_dbs, string argv_out);