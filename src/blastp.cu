#include "blastp.h"
#include <assert.h>
#include <chrono>
#include <algorithm> 
#include <nvtx3/nvToolsExt.h>

#define PACK_KEY(k) ((k & ~0x7) | 0x3)

ThreadPool *pool;
mutex mu2;

// vector<SWResult> res_s[MAX_GROUPS_PER_ROUND][NUM_STREAM];

__constant__ uint32_t kHashTableCapacity_dev[MAX_GROUPS_PER_ROUND][MAX_QUERY_PER_GROUP];
__constant__ uint32_t kHashTableOffset_dev[MAX_GROUPS_PER_ROUND][MAX_QUERY_PER_GROUP];

__constant__ int SEED_LENGTH;
__constant__ int QIT_WIDTH;
__constant__ uint32_t MASK;

// 32 bit Murmur3 hash
inline __device__ uint32_t my_hash(uint32_t k, uint32_t kHashTableCapacity)
{
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & (kHashTableCapacity - 1);
}

__device__ int insert_ot(KeyValue *hashtable, uint32_t kHashTableCapacity, uint32_t key, uint32_t value)
{
    key = PACK_KEY(key);
    uint32_t slot = my_hash(key, kHashTableCapacity);
    uint32_t b_slot = slot;
    while (true)
    {
        uint32_t prev = atomicCAS(&hashtable[slot].key, kEmpty, key);
        if (prev == kEmpty || prev == key)
        {
            // hashtable[slot].value = value;
            atomicAdd(&hashtable[slot].value, value);
            return 0;
        }
        slot = (slot + 1) & (kHashTableCapacity - 1);
        if (slot == b_slot)
        {
            return -1;
        }
    }
}

__global__ void seeding_kernel(KeyValue *ht, uint32_t *subj, size_t s_length_block, size_t s_length_total, const uint32_t *q_lengths, const int *q_num, const int *q_idx, int n_query, uint8_t *index_size_dev, uint32_t group_id)
{
    size_t s_begin = ((blockIdx.x * blockDim.x + threadIdx.x) * s_length_block) * 32;

    size_t s_len = s_length_block * 32;
    if (s_begin + s_len >= s_length_total - SEED_LENGTH)
        s_len = s_length_total - SEED_LENGTH - s_begin;
    if (s_len <= 0)
        return;

    size_t s_end = s_begin + s_len;

    for (size_t i = s_begin; i < s_end; i++)
    {
        size_t n_bit = i * 5;
        size_t pos = (n_bit >> 5);
        uint32_t mod = n_bit & 31;
        // assert(pos % 4 == 0);
        uint32_t qit_idx = (subj[pos] >> mod) & MASK;
        if (mod > (31 - (5 * SEED_LENGTH)))
        {
            qit_idx |= (subj[pos + 1] << (32 - mod)) & MASK;
        }

        int hit_size = index_size_dev[qit_idx];

        if (hit_size <= 0)
            continue;

        int qit_p = 0;
        for (int j = 0; j < hit_size; j++)
        {
            int pos = qit_idx * QIT_WIDTH + qit_p;
            int q_num_now = q_num[pos];
            int q_idx_now = q_idx[pos];
            if (q_num_now == -1)
            {
                qit_idx += q_idx_now;
                qit_p = 0;
                pos = qit_idx * QIT_WIDTH;
                q_num_now = q_num[pos];
                q_idx_now = q_idx[pos];
            }

            // printf("%d %d\n",q_num[qit_idx*qit_width+qit_p],q_idx[qit_idx*qit_width+qit_p]);
            unsigned int diag = q_lengths[q_num_now] + i - q_idx_now;
            // KeyValue *pHashTable_addr = ot + q_num_now * kHashTableCapacity_dev[q_num_now];
            KeyValue *pHashTable_addr = ht + kHashTableOffset_dev[group_id][q_num_now];
            int err = insert_ot(pHashTable_addr, kHashTableCapacity_dev[group_id][q_num_now], diag, 1);
            // assert(err != -1);
            if (err == -1)
            {
                printf("Voting Hash Table Full! G%uQ%uK%u\n", group_id, q_num_now, kHashTableCapacity_dev[group_id][q_num_now]);
            }
            qit_p++;
        }
    }
}

__global__ void filter_kernel(KeyValue *ht, Task *tasks, uint32_t *num_task, uint32_t *threshold, uint32_t group_id)
{
    uint32_t q_id = blockIdx.x;
    KeyValue *h_begin = ht + kHashTableOffset_dev[group_id][q_id];

    size_t each_length = (kHashTableCapacity_dev[group_id][q_id] - 1) / blockDim.x + 1;
    h_begin += each_length * threadIdx.x;
    KeyValue *h_end = h_begin + each_length;

    KeyValue *total_end = ht + kHashTableOffset_dev[group_id][q_id] + kHashTableCapacity_dev[group_id][q_id];
    h_end = h_end > total_end ? total_end : h_end;

    Task *task_begin = tasks;

    for (KeyValue *kv = h_begin; kv < h_end; kv++)
    {
        if (kv->key != kEmpty && kv->value != kEmpty && kv->value >= threshold[q_id])
        {
            uint32_t idx = atomicAdd(num_task, 1);
            if (idx >= MAX_FILTER_TASK)
            {
                printf("Filter Task Vector Full! G%uQ%uT%u\n", group_id, q_id, idx);
                return;
            }
            task_begin[idx].key = kv->key;
            task_begin[idx].value = kv->value;
            task_begin[idx].q_id = q_id;
        }
    }

    // size_t total_length = kHashTableOffset_dev[group_id][n_query-1] + kHashTableCapacity_dev[group_id][n_query-1];
    // size_t each_length = (total_length-1)/b + 1;
}

__global__ void banded_sw_kernel(
                int NumTasks,
                uint32_t* q_lens, uint32_t* q_idxs, Task* task,
                const char* query, const char* target, size_t target_len,
                int * max_score,
                size_t* q_end_idx, size_t* s_end_idx,
                char* cigar_op, int* cigar_cnt,int* cigar_len,
                int *rd, record* rt,int band_width,
                int* BLOSUM62){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx >= NumTasks) return;

    size_t query_len = q_lens[task[idx].q_id];
    assert( query_len < MaxQueryLen);
   
    size_t q_idx  = q_idxs[task[idx].q_id];
    size_t diag  = task[idx].key;
    
    int64_t c_begin = (int64_t)diag - band_width - query_len + 2;
    // size_t c_end = diag + band_width;
    
    record* tile = rt + idx * MaxBW * (TILE_SIZE + 1);
    
    // __shared__ int shared_BLOSUM62[26 * 26];
    
    // for(size_t i = 0; i < MaxBW; ++ i){
    //     for(size_t j = 0; j < MaxQueryLen+1; ++ j){
    //         rd[BatchSize * (i * (MaxQueryLen+1) + j) + idx] = 0;
    //     }
    // }
    // __syncthreads(); // Wait for the copy to compl .ete


    size_t width = 2 * band_width + 1;
    size_t height = MaxQueryLen + 1;
    assert(width < MaxBW);

    //init:
    max_score[idx] = 0;
    
    size_t t_height = TILE_SIZE + 1;
    
    // record *rt = (record *)malloc(width * t_height * sizeof(record));
    // memset(rt, 0, width * t_height * sizeof(record));

    size_t max_q = 0;
    size_t max_c = 0;
    int score = 0, Score = 0;
    // cal maxScore and it's position
    for (size_t it = 0; it * TILE_SIZE < query_len; it++) {
        
        size_t q_offset = it * TILE_SIZE;

        for(size_t _q = 0; _q < t_height-1 && q_offset + _q < query_len; ++_q){
            for(size_t _c = 0; _c < width-2; ++_c){
                
                if(c_begin + _c+ q_offset + _q < 0) continue;
                if(c_begin + _c+ q_offset + _q >= target_len) break;

                char chq = query[q_idx + q_offset + _q];
                char chc = get_char_d(target, c_begin + q_offset + _c + _q);
                
                if (chq == END_SIGNAL || chc == END_SIGNAL)
                {
                    continue;
                }
                //rt(_q,_c) -> (_q+1) * width + _c + 1
                // logical m(_q,_c).x = max(m(_q-1,_c).x + SCORE_GAP_EXT, m(_q-1,_c).m +SCORE_GAP, 0 );
                // logical m(_q,_c).y = max(m(_q,_c-1).y + SCORE_GAP_EXT, m(_q,_c-1).m +SCORE_GAP, 0 );
                // logical m(_q,_c).m = max(m(_q-1,_c-1).y,m(_q-1,_c-1).x,m(_q-1,_c-1).m, 0 );
                
                tile[calIndex(_q,_c,MaxBW)].x = max3(tile[calTop(_q,_c,MaxBW)].x + SCORE_GAP_EXT,  tile[calTop(_q,_c,MaxBW)].m + SCORE_GAP, 0);
                tile[calIndex(_q,_c,MaxBW)].y = max3(tile[calLeft(_q,_c,MaxBW)].y + SCORE_GAP_EXT, tile[calLeft(_q,_c,MaxBW)].m + SCORE_GAP, 0);

                if (chq == ILLEGAL_WORD || chc == ILLEGAL_WORD)
                {
                    // illegal word
                    tile[calIndex(_q,_c,MaxBW)].m = 0;
                }
                else
                {
                    tile[calIndex(_q,_c,MaxBW)].m = max2(max3(tile[calDiag(_q,_c,MaxBW)].x, tile[calDiag(_q,_c,MaxBW)].y, tile[calDiag(_q,_c,MaxBW)].m) + BLOSUM62[chq * 26 + chc], 0);
                }

                score = max3(tile[calIndex(_q,_c,MaxBW)].x, tile[calIndex(_q,_c,MaxBW)].y, tile[calIndex(_q,_c,MaxBW)].m);
                
                // printf("(query = %target,c = %c) BLOSUM62 = %d tile[_q * width + _c].s = %d\n", chq+65,chc+65,BLOSUM62[chq * 26 + chc], tile[_q * width + _c].s);
                // (rd + idx*direct_matrixSize)[_c * height + _q + q_offset] = (score == tile[_q * width + _c].x)*TOP + (score == tile[_q * width + _c].y)*LEFT + (tile[_c * height + _q + q_offset].m)*DIAG; 
            
                rd[calIndex(_c, _q+q_offset,height) * BatchSize + idx] = (score?( \
                    (score == tile[calIndex(_q,_c,MaxBW)].m) ? DIAG : \
                    ((score == tile[calIndex(_q,_c,MaxBW)].y) ? LEFT :TOP )):0);
                
                if (Score < score)
                {
                    Score = score;
                    max_c = _c;
                    max_q = _q + q_offset;
                }
                // printf("(q = %c,c = %c) score = %d maxScore = %d direction = %d\n", chq+65,chc+65,r[_q*width + _c].s,r[max_c * height + max_q].s,r[_q * width + _c].d);
            }
        }
        memcpy(tile,tile + (t_height - 1) * MaxBW ,MaxBW * sizeof(record));
        // Hit when target is not long enough, there are some cells should be zero
        memset(tile + MaxBW, 0, (t_height - 1) * MaxBW * sizeof(record));

    }

    max_score[idx] = Score;
    // res[idx].score = Score;
    assert(Score != 0);

    size_t cur_q= max_q;
    size_t cur_c = max_c;

    q_end_idx[idx] = cur_q + q_idx;
    s_end_idx[idx] = c_begin + cur_c + cur_q;

    // int cnt_q = 0, cnt_c = 0;
    int cigar_cur_len = 0;
    while (rd[BatchSize * calIndex(cur_c,cur_q,height) + idx])
    {
        int d = rd[BatchSize * calIndex(cur_c,cur_q,height) + idx];
        // size_t res_q = (d&0x01) ? (cur_q + q_idx) : (size_t)-1;
        // size_t res_c = (d&0x02) ? (c_begin + cur_c + cur_q) : (size_t)-1;
        
        // q_res_d[idx* MaxAlignLen + (cnt_q)] = (res_q);
        // s_res_d[idx* MaxAlignLen + (cnt_c)] = (res_c);
        int cur_cigar_cnt = 0;
        while (rd[BatchSize * calIndex(cur_c,cur_q,height) + idx] && rd[BatchSize * calIndex(cur_c,cur_q,height) + idx]==d){
            cur_cigar_cnt ++;
            
            //TOP 01b, left 10b, diag 11b
            //DIAG : cur_q -= 1
            //TOP : cur_q -= 1, cur_c += 1;
            //LEFT : cur_c -= 1
            cur_q -= (d == DIAG || d == TOP);
            cur_c += (d == TOP); // Increment cur_c if TOP (01b)
            cur_c -= (d == LEFT); // Decrement cur_c if LEFT (10b)
        }
        (cigar_cnt + idx * MaxAlignLen)[cigar_cur_len] = cur_cigar_cnt;
        (cigar_op + idx * MaxAlignLen)[cigar_cur_len++] = ((d==DIAG)?'M':((d==TOP)?'D':'I'));
    }

    // free(rt);
    // printf("@@ cigar_len %d %d\n", idx,cigar_len);
    assert(cigar_cur_len > 0);
    cigar_len[idx] = cigar_cur_len;
}

void handle_results(cudaEvent_t &stream, const char *query, const char *subj, Task *task_host, uint32_t *num_task, QueryGroup &q_group, size_t s_length, int stream_id, vector<SWResult> &res, SWTasks &sw_task, ThreadPool *pool, vector<future<int>> &rs)
{
    cudaEventSynchronize(stream);
    cout << "=";
    res.clear();
    res.resize(*num_task);
    sw_task.q = query;
    sw_task.c = subj;
    sw_task.c_len = s_length;
    sw_task.q_idxs.resize(*num_task);
    sw_task.q_lens.resize(*num_task);
    sw_task.diags.resize(*num_task);
    Task *t_begin = task_host;
    sw_task.num_task = *num_task;
#pragma omp parallel for
    for (int i = 0; i < *num_task; i++)
    {
        Task &kv = *(t_begin + i);
        sw_task.q_idxs[i]=q_group.offset[kv.q_id];
        sw_task.q_lens[i]=q_group.length[kv.q_id];
        sw_task.diags[i]=kv.key;
        res[i].num_q = kv.q_id;
    }
    mu2.lock();
    for (int i = 0; i < sw_task.num_task; ++i)
    {
        rs.emplace_back(pool->enqueue([&, i]
                                      {
            smith_waterman_kernel(i,&res[i],&sw_task);
            return i; }));
    }
    mu2.unlock();
}

void call_results(cudaEvent_t& event, \
                cudaStream_t& stream,
                // int* cigar_len_d, char* cigar_op_d, int* cigar_cnt_d,
                // size_t* q_end_d, size_t* s_end_d,
                // int* score_d,
                const char* query, const char* subj,
                Task* task_host, int num_task,
                int* cigar_len_h, char* cigar_op_h, int* cigar_cnt_h,
                size_t* q_end_h, size_t* s_end_h,
                int* score_h,
                std::vector<SWResult>& res_s, int begin,
                ThreadPool* pool, std::vector<std::future<int>>& rs) {
    CUDA_CALL(cudaEventSynchronize(event));
    CUDA_CALL(cudaEventDestroy(event));
    // CUDA_CALL(cudaFreeAsync(s_end_d, stream));
    // CUDA_CALL(cudaFreeAsync(q_end_d, stream));
    // CUDA_CALL(cudaFreeAsync(cigar_op_d, stream));
    // CUDA_CALL(cudaFreeAsync(cigar_cnt_d, stream));
    // CUDA_CALL(cudaFreeAsync(cigar_len_d, stream));
    // CUDA_CALL(cudaFreeAsync(score_d, stream));
    // cout << "=";
    mu2.lock();
    for (size_t i = 0; i < num_task; ++i) {
        rs.emplace_back(pool->enqueue([&, i] {
            
            cigar_to_index_and_report(i, begin, cigar_len_h, cigar_op_h, cigar_cnt_h,
                        q_end_h, s_end_h, res_s, score_h, task_host, query, subj);
            
            return static_cast<int>(i);
        }));
    }
    mu2.unlock();
}
void call_results_cpu(const char* query, const char* subj,
                Task* task_host, int num_task,
                int* cigar_len_h, char* cigar_op_h, int* cigar_cnt_h,
                size_t* q_end_h, size_t* s_end_h,
                int* score_h,
                std::vector<SWResult>& res_s, int begin,
                ThreadPool* pool, std::vector<std::future<int>>& rs) {
   
    mu2.lock();
    for (size_t i = 0; i < num_task; ++i) {
        rs.emplace_back(pool->enqueue([&, i] {
            
            cigar_to_index_and_report(i, begin, cigar_len_h, cigar_op_h, cigar_cnt_h,
                        q_end_h, s_end_h, res_s, score_h, task_host, query, subj);
            
            return static_cast<int>(i);
        }));
    }
    mu2.unlock();
}
__global__ void initializeArray_rd(int* array, int value, size_t numElements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        array[idx] = value;
    }
}
__global__ void initializeArray_rt(record* array, int value, size_t numElements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        array[idx].x = value;
        array[idx].y = value;
        array[idx].m = value;
    }
}
// void CUDART_CB hostCallBack(cudaStream_t stream, cudaError_t status,void* userData) {
//     CallbackData* data = static_cast<CallbackData*>(userData);
//     data->res_s.resize(data->n);
//     *(data->g_begin) =  *(data->g_begin) + data->off_g;
// #pragma omp parallel for
//     for (size_t i = 0; i < data->n; ++i) {
//         assert(data->cigar_len_h[i] < MaxAlignLen && data->cigar_len_h[i]);
//         //TODO From cigar to index
//         cigar_to_index(data->cigar_len_h[i],
//                        data->cigar_op_h + i * MaxAlignLen,
//                        data->cigar_cnt_h + i * MaxAlignLen,
//                        data->q_end_h[i], data->s_end_h[i],
//                        ref(data->res_s[i].q_res),
//                        ref(data->res_s[i].s_res));
       
//         data->res_s[i].score = data->score_h[i];            
//         data->res_s[i].num_q = data->task_host[i].q_id;
//         generate_report(&data->res_s[i], data->query, data->subj);
//     }
//     data->streamState->callbackCompleted.store(true, std::memory_order_release);
// }


void search_db_batch(const char *query, char *subj[], vector<QueryGroup> &q_groups, size_t s_length[], Task *task_host[][NUM_STREAM], uint32_t *task_num_host[][NUM_STREAM], size_t max_hashtable_capacity, uint32_t max_n_query, uint32_t total_len_query, string db_name, uint32_t db_num, vector<SWResult> *res, size_t total_db_size, TimeProfile &time_prof)
{
    struct timeval t_start, t_end, tt_start;

    gettimeofday(&t_start, NULL);

    CUDA_CALL(cudaMemcpyToSymbol(SEED_LENGTH, &seed_length, sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(QIT_WIDTH, &qit_width, sizeof(int)));
    uint32_t mask = (uint32_t)pow(2, 5 * seed_length) - 1;
    CUDA_CALL(cudaMemcpyToSymbol(MASK, &mask, sizeof(uint32_t)));

    size_t sum_s_len = 0;
    for (int i = 0; i < NUM_STREAM; i++)
    {
        sum_s_len += s_length[i];
        assert(s_length[i] % 32 == 0);
    }

    char *subj_dev;
    CUDA_CALL(cudaMalloc((void **)&subj_dev, sum_s_len / 8 * 5));

    int n_groups = q_groups.size();
    if (n_groups > MAX_GROUPS_PER_ROUND)
        n_groups = MAX_GROUPS_PER_ROUND;

    int *q_num_dev[n_groups];
    int *q_idx_dev[n_groups];
    uint32_t *q_lengths_dev[n_groups];
    uint8_t *index_size_dev[n_groups];
    uint32_t *threshold_dev[n_groups];
    uint32_t *q_offset_dev[n_groups];

    uint32_t kHashTableCapacity_host[MAX_GROUPS_PER_ROUND][MAX_QUERY_PER_GROUP];
    uint32_t kHashTableOffset_host[MAX_GROUPS_PER_ROUND][MAX_QUERY_PER_GROUP];

    for (int g = 0; g < n_groups; g++)
    {
        CUDA_CALL(cudaMalloc((void **)&q_num_dev[g], qit_length * qit_width * sizeof(int)));
        CUDA_CALL(cudaMalloc((void **)&q_idx_dev[g], qit_length * qit_width * sizeof(int)));
        CUDA_CALL(cudaMalloc((void **)&q_lengths_dev[g], MAX_QUERY_PER_GROUP * sizeof(uint32_t)));
        CUDA_CALL(cudaMalloc((void **)&q_offset_dev[g], MAX_QUERY_PER_GROUP * sizeof(uint32_t)));
        CUDA_CALL(cudaMalloc((void **)&index_size_dev[g], qit_length * sizeof(uint8_t)));
        CUDA_CALL(cudaMalloc((void **)&threshold_dev[g], MAX_QUERY_PER_GROUP * sizeof(uint32_t)));
    }
    KeyValue *pHashTable[n_groups][NUM_STREAM];
    Task *task_dev[n_groups][NUM_STREAM];
    uint32_t *task_num_dev[n_groups][NUM_STREAM];

    for (int g = 0; g < n_groups; g++)
    {
        for (int s = 0; s < NUM_STREAM; s++)
        {
            // pHashTable[s] = create_hashtable(max_hashtable_capacity);
            CUDA_CALL(cudaMalloc((void **)&task_dev[g][s], MAX_FILTER_TASK * sizeof(Task)));
            CUDA_CALL(cudaMemset(task_dev[g][s], 0, MAX_FILTER_TASK * sizeof(Task)));
            CUDA_CALL(cudaMalloc((void **)&task_num_dev[g][s], sizeof(uint32_t)));
            CUDA_CALL(cudaMemset(task_num_dev[g][s], 0, sizeof(uint32_t)));
        }
    }

    char *s_name[NUM_STREAM] = {0};
    size_t *s_offsets[NUM_STREAM] = {0};
    size_t *sn_offsets[NUM_STREAM] = {0};
    size_t s_num[NUM_STREAM] = {0};

    int mingridsize_seeding, mingridsize_filter;
    int threadblocksize_seeding, threadblocksize_filter;
    CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&mingridsize_seeding, &threadblocksize_seeding, seeding_kernel, 0, 0));
    CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&mingridsize_filter, &threadblocksize_filter, filter_kernel, 0, 0));

    // cout << "Seeding Block size:" << threadblocksize_seeding <<"," << mingridsize_seeding <<endl;
    // cout << "Filter Block size:" << threadblocksize_filter <<"," << mingridsize_filter <<endl;

    SWTasks sw_tasks_total;
    vector<SWResult> res_s[q_groups.size()][NUM_STREAM];

    char* query_dev;
    CUDA_CALL(cudaMalloc((void **)&query_dev, total_len_query));
    CUDA_CALL(cudaMemcpy(query_dev, query, total_len_query, cudaMemcpyHostToDevice));

    size_t free_byte, total_byte;
    CUDA_CALL(cudaMemGetInfo(&free_byte, &total_byte));
    cout << "GPU mem: " << (double)(total_byte - free_byte) / (1073741824) << " GB / " << (double)total_byte / (1073741824) << " GB" << endl;

    size_t s_begin = 0;
    cudaStream_t streams;
    cudaStream_t malloc_streams;
    CUDA_CALL(cudaStreamCreate(&streams));
    CUDA_CALL(cudaStreamCreate(&malloc_streams));
    vector<size_t> s_begin_vec;

#ifdef USE_GPU_DIFFUSE
    int direct_matrixSize = (MaxQueryLen+1) * MaxBW;
    int threadsPerBlock = 64;  // 根据 shared memory 限制调整
    int blocks = (BatchSize + threadsPerBlock - 1) / threadsPerBlock;
    int blocks_rd = (direct_matrixSize * BatchSize + threadsPerBlock - 1) / threadsPerBlock;
    int blocks_rt = (MaxBW * (TILE_SIZE + 1) * BatchSize + threadsPerBlock - 1) / threadsPerBlock;
    
    int* direction_matrix[n_groups][NUM_STREAM][MaxNumBatch];   // direct_matrixSize * BatchSize * sizeof(int)
    record* tiled_direction_matrix[n_groups][NUM_STREAM][MaxNumBatch];
    int* BLOSUM62_d;
    int* score_d[n_groups][NUM_STREAM][MaxNumBatch], *score_h[n_groups][NUM_STREAM][MaxNumBatch], *score;
    
    size_t* q_end_d[n_groups][NUM_STREAM][MaxNumBatch], *q_end_h[n_groups][NUM_STREAM][MaxNumBatch], *q_end;
    size_t* s_end_d[n_groups][NUM_STREAM][MaxNumBatch], *s_end_h[n_groups][NUM_STREAM][MaxNumBatch], *s_end;
    char* cigar_op_d[n_groups][NUM_STREAM][MaxNumBatch], *cigar_op_h[n_groups][NUM_STREAM][MaxNumBatch], *cigar_op;
    int *cigar_cnt_d[n_groups][NUM_STREAM][MaxNumBatch], *cigar_cnt_h[n_groups][NUM_STREAM][MaxNumBatch], *cigar_cnt;
    int* cigar_len_d[n_groups][NUM_STREAM][MaxNumBatch], *cigar_len_h[n_groups][NUM_STREAM][MaxNumBatch], *cigar_len;

    score = (int*)malloc(sizeof(int) * BatchSize);

    q_end = (size_t*)malloc(sizeof(size_t) * BatchSize);
    s_end = (size_t*)malloc(sizeof(size_t) * BatchSize);
    
    cigar_cnt = (int*)malloc(sizeof(int) * MaxAlignLen * BatchSize);
    cigar_len = (int*)malloc(sizeof(int)* BatchSize);
    cigar_op = (char*)malloc(sizeof(char) * MaxAlignLen * BatchSize);
    int* rd = (int*)malloc(direct_matrixSize * sizeof(int) * BatchSize);   // direct_matrixSize * BatchSize * sizeof(int)
    record* rt = (record*)malloc(MaxBW * (TILE_SIZE + 1) * sizeof(record) * BatchSize); 

    for(int g = 0; g < n_groups; ++ g){

        for (int s = 0; s < NUM_STREAM; s++)
        {
            for(int cur = 0; cur < MaxNumBatch; ++ cur){

                CUDA_CALL(cudaMallocHost(&score_h[g][s][cur], sizeof(int) * BatchSize));
            
                CUDA_CALL(cudaMallocHost(&q_end_h[g][s][cur], sizeof(size_t) * BatchSize));
                CUDA_CALL(cudaMallocHost(&s_end_h[g][s][cur], sizeof(size_t) * BatchSize));
                
                CUDA_CALL(cudaMallocHost(&cigar_cnt_h[g][s][cur], sizeof(int) * MaxAlignLen * BatchSize));
                CUDA_CALL(cudaMallocHost(&cigar_op_h[g][s][cur], sizeof(char) * MaxAlignLen * BatchSize));
                CUDA_CALL(cudaMallocHost(&cigar_len_h[g][s][cur], sizeof(int)* BatchSize));

                CUDA_CALL(cudaMallocAsync((void**)&cigar_op_d[g][s][cur], BatchSize * sizeof(char) * MaxAlignLen, streams));
                CUDA_CALL(cudaMallocAsync((void**)&cigar_cnt_d[g][s][cur], BatchSize * sizeof(int) * MaxAlignLen, streams));

                CUDA_CALL(cudaMallocAsync((void**)&cigar_len_d[g][s][cur], BatchSize * sizeof(int), streams));

                CUDA_CALL(cudaMallocAsync((void**)&score_d[g][s][cur], BatchSize * sizeof(int), streams));
                
                CUDA_CALL(cudaMallocAsync((void**)&q_end_d[g][s][cur], BatchSize * sizeof(size_t), streams));
                CUDA_CALL(cudaMallocAsync((void**)&s_end_d[g][s][cur], BatchSize * sizeof(size_t), streams));
                   
        
            }
        }
    }
    CUDA_CALL(cudaMallocAsync((void**)&BLOSUM62_d, 26 * 26 * sizeof(int), malloc_streams));
    CUDA_CALL(cudaMemcpyAsync(BLOSUM62_d, BLOSUM62, 26 * 26 * sizeof(int),cudaMemcpyHostToDevice,malloc_streams));
    cudaStream_t copy_streams[MaxNumBatch];
    for(int cur = 0; cur < MaxNumBatch; ++ cur){
        cudaStreamCreate(&copy_streams[cur]);
    }
#endif

    for(int s = 0; s < NUM_STREAM; ++ s){
        s_begin_vec.push_back(s_begin);
        size_t s_length_stream = s_length[s];
        size_t s_length_stream_byte = s_length_stream / 8 * 5;
        s_begin += s_length_stream_byte;
        // printf("start stream %d\n", s);
        size_t s_length_stream_block = s_length_stream / 32 * 5;
        size_t each_length_block = (s_length_stream_block - 1) / (mingridsize_seeding * threadblocksize_seeding) + 1;
        CUDA_CALL(cudaMemcpyAsync(subj_dev + s_begin_vec[s], subj[s], s_length_stream_byte, cudaMemcpyHostToDevice, malloc_streams));

    }
    gettimeofday(&t_end, NULL);
    time_prof.mem_time += timeuse(t_start, t_end);

    for (int g_begin = 0; g_begin < q_groups.size(); g_begin += MAX_GROUPS_PER_ROUND)
    {
        sw_tasks_total.c_offset = 0;
        double group_time = 0;
        cout << "Group " << g_begin + 1 << "/" << q_groups.size() << "\t[";
        gettimeofday(&t_start, NULL);
        n_groups = q_groups.size() - g_begin;
        if (n_groups > MAX_GROUPS_PER_ROUND)
            n_groups = MAX_GROUPS_PER_ROUND;
        for (int g = g_begin; g < g_begin + n_groups; g++)
        {
            int g_idx = g - g_begin;
            CUDA_CALL(cudaMemcpy(q_num_dev[g_idx], q_groups[g].qit.q_num, qit_length * qit_width * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(q_idx_dev[g_idx], q_groups[g].qit.q_idx, qit_length * qit_width * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(index_size_dev[g_idx], q_groups[g].qit.index_size, qit_length * sizeof(uint8_t), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(q_lengths_dev[g_idx], q_groups[g].length, MAX_QUERY_PER_GROUP * sizeof(uint32_t), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(q_offset_dev[g_idx], q_groups[g].offset, MAX_QUERY_PER_GROUP * sizeof(uint32_t), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(threshold_dev[g_idx], q_groups[g].min_diag_hit, MAX_QUERY_PER_GROUP * sizeof(uint32_t), cudaMemcpyHostToDevice));
            memcpy(kHashTableCapacity_host[g_idx], q_groups[g].hashtable_capacity, sizeof(uint32_t) * MAX_QUERY_PER_GROUP);
            memcpy(kHashTableOffset_host[g_idx], q_groups[g].hashtable_offset, sizeof(uint32_t) * MAX_QUERY_PER_GROUP);
        }

        CUDA_CALL(cudaMemcpyToSymbol(kHashTableCapacity_dev, kHashTableCapacity_host, sizeof(uint32_t) * MAX_QUERY_PER_GROUP * MAX_GROUPS_PER_ROUND));
        CUDA_CALL(cudaMemcpyToSymbol(kHashTableOffset_dev, kHashTableOffset_host, sizeof(uint32_t) * MAX_QUERY_PER_GROUP * MAX_GROUPS_PER_ROUND));

        
        thread result_threads[n_groups][NUM_STREAM];
#ifdef USE_GPU_DIFFUSE
        
        cudaEvent_t copies_done[n_groups][NUM_STREAM][MaxNumBatch];
        cudaEvent_t kernels_done[n_groups][NUM_STREAM][MaxNumBatch];
        
        thread report_threads[n_groups][NUM_STREAM][MaxNumBatch];
        vector<future<int>> rs_report[n_groups][NUM_STREAM][MaxNumBatch];

#endif
        cudaEvent_t malloc_finished;
        cudaEvent_t seeding_finished[n_groups][NUM_STREAM];
        

        gettimeofday(&t_end, NULL);
        group_time += timeuse(t_start, t_end);
        time_prof.mem_time += timeuse(t_start, t_end);
        // cout << "Prepare mem and data Time: " << timeuse(t_start, t_end) << endl;

        gettimeofday(&t_start, NULL);
        nvtxRangePush("MyProfileRegion");
        for (int g = g_begin; g < g_begin + n_groups; g++)
        {
            int g_idx = g - g_begin;

            for (int s = 0; s < NUM_STREAM; s++)
            {
                CUDA_CALL(cudaEventCreate(&malloc_finished));
                CUDA_CALL(cudaEventCreate(&seeding_finished[g_idx][s]));
#ifdef USE_GPU_DIFFUSE
                for(int cur = 0; cur < MaxNumBatch; ++ cur){
                    CUDA_CALL(cudaEventCreate(&copies_done[g_idx][s][cur]));
                    CUDA_CALL(cudaEventCreate(&kernels_done[g_idx][s][cur]));
                }
#endif
                // pHashTable[g_idx][s] = create_hashtable(max_hashtable_capacity);
                // CUDA_CALL(cudaStreamSynchronize(malloc_streams));
                pHashTable[g_idx][s] = create_hashtable_async(max_hashtable_capacity,malloc_streams);
                size_t s_length_stream = s_length[s];
                size_t s_length_stream_byte = s_length_stream / 8 * 5;
                size_t s_length_stream_block = s_length_stream / 32 * 5;
                size_t each_length_block = (s_length_stream_block - 1) / (mingridsize_seeding * threadblocksize_seeding) + 1;

                int n_query = q_groups[g].n_query;
                if (g > 0)
                {
                    CUDA_CALL(cudaMemsetAsync(task_dev[g_idx][s], 0, MAX_FILTER_TASK * sizeof(Task), malloc_streams));
                    CUDA_CALL(cudaMemsetAsync(task_num_dev[g_idx][s], 0, sizeof(uint32_t), malloc_streams));
                    CUDA_CALL(cudaMemsetAsync(pHashTable[g_idx][s], 0xff, max_hashtable_capacity * sizeof(KeyValue), malloc_streams));
                }
                // assert(s_begin_vec[s] == s_begin);
                CUDA_CALL(cudaEventRecord(malloc_finished, malloc_streams));
                CUDA_CALL(cudaEventSynchronize((malloc_finished)));
                CUDA_CALL(cudaEventDestroy(malloc_finished));

                seeding_kernel<<<mingridsize_seeding, threadblocksize_seeding, 0, streams>>>(pHashTable[g_idx][s], (uint32_t *)(subj_dev + s_begin_vec[s]), each_length_block, s_length_stream, q_lengths_dev[g_idx], q_num_dev[g_idx], q_idx_dev[g_idx], n_query, index_size_dev[g_idx], g_idx);
                filter_kernel<<<n_query, threadblocksize_filter, 0, streams>>>(pHashTable[g_idx][s], task_dev[g_idx][s], task_num_dev[g_idx][s], threshold_dev[g_idx], g_idx);
                
                // destroy_hashtable(pHashTable[g_idx][s]);
                destroy_hashtable_async(pHashTable[g_idx][s],malloc_streams);
                
                CUDA_CALL(cudaMemcpyAsync(task_num_host[g_idx][s], task_num_dev[g_idx][s], sizeof(uint32_t), cudaMemcpyDeviceToHost, streams));    
                CUDA_CALL(cudaMemcpyAsync(task_host[g_idx][s], task_dev[g_idx][s], MAX_FILTER_TASK * sizeof(Task), cudaMemcpyDeviceToHost, streams));
                CUDA_CALL(cudaEventRecord(seeding_finished[g_idx][s], streams));
               
#ifdef USE_GPU_DIFFUSE
                CUDA_CALL(cudaEventSynchronize((seeding_finished[g_idx][s])));
                CUDA_CALL(cudaEventDestroy(seeding_finished[g_idx][s]));

                res_s[g][s].resize(*task_num_host[g_idx][s]);
                // vector<int>num_task_vec;
                // printf("NumTask %d\n", *task_num_host[g_idx][s]);
                assert(*task_num_host[g_idx][s] < MaxNumBatch * BatchSize);
                for(size_t cur = 0; cur < MaxNumBatch; cur ++){
                    size_t it = cur * BatchSize;
                    int num_task = BatchSize;
                    if(it > *task_num_host[g_idx][s]) break;
                    if(it+BatchSize >= *task_num_host[g_idx][s]){
                        num_task =  *task_num_host[g_idx][s] - it;
                                                num_task =  *task_num_host[g_idx][s] - it;
                        if(num_task < (BatchSize >> 1)){
                            struct timeval cpu_start, cpu_end;
                            gettimeofday(&cpu_start, NULL);
                            int g = g_begin + g_idx;
// #pragma omp parallel for
                            banded_sw_cpu_kernel(num_task,
                                q_groups[g].length, q_groups[g].offset, task_host[g_idx][s] + it,
                                query,subj[s], s_length[s],
                                score,
                                q_end, s_end,
                                cigar_op, cigar_cnt, cigar_len,
                                rd, rt, band_width,
                                BLOSUM62);
                            
                            // for(int i = 0; i < num_task; i++){
                            //     cigar_to_index_and_report(i, it, cigar_len, cigar_op, cigar_cnt,
                            //                         q_end, s_end, res_s[g][s], score, task_host[g_idx][s]+it, query, subj[s]);
                                
                            // }
                            report_threads[g_idx][s][cur] = thread(call_results_cpu,
                                                query, subj[s],
                                                task_host[g_idx][s]+it, num_task,
                                                cigar_len, cigar_op, cigar_cnt,
                                                q_end, s_end,
                                                score,
                                                ref(res_s[g][s]), it,
                                                pool, ref(rs_report[g_idx][s][cur]));
                            gettimeofday(&cpu_end, NULL);
                            // cout << "cpu remaind: " << timeuse(cpu_start, cpu_end) << endl;
                            break;
                        }
                    }
                    // num_task_vec.push_back(num_task);
                    
                    
                    CUDA_CALL(cudaMallocAsync((void**)&direction_matrix[g_idx][s][cur], direct_matrixSize * BatchSize * sizeof(int), streams));
                    CUDA_CALL(cudaMallocAsync((void**)&tiled_direction_matrix[g_idx][s][cur], MaxBW * (TILE_SIZE + 1) * BatchSize * sizeof(record), streams));
                    // CUDA_CALL(cudaMemsetAsync(direction_matrix[g_idx][s][cur], 0, direct_matrixSize * BatchSize * sizeof(int),  streams));
                    // CUDA_CALL(cudaMemsetAsync(tiled_direction_matrix[g_idx][s][cur], 0, MaxBW * (TILE_SIZE + 1) * BatchSize * sizeof(record),  streams));
                    initializeArray_rd<<<blocks_rd, threadsPerBlock, 0, streams>>>(direction_matrix[g_idx][s][cur], 0, direct_matrixSize * BatchSize);
                    initializeArray_rt<<<blocks_rt, threadsPerBlock, 0, streams>>>(tiled_direction_matrix[g_idx][s][cur], 0, MaxBW * (TILE_SIZE + 1) * BatchSize);
                    
                    // CUDA_CALL(cudaMallocAsync((void**)&cigar_op_d[g_idx][s][cur], BatchSize * sizeof(char) * MaxAlignLen, streams));
                    // CUDA_CALL(cudaMallocAsync((void**)&cigar_cnt_d[g_idx][s][cur], BatchSize * sizeof(int) * MaxAlignLen, streams));

                    // CUDA_CALL(cudaMallocAsync((void**)&cigar_len_d[g_idx][s][cur], BatchSize * sizeof(int), streams));

                    // CUDA_CALL(cudaMallocAsync((void**)&score_d[g_idx][s][cur], BatchSize * sizeof(int), streams));
                    
                    // CUDA_CALL(cudaMallocAsync((void**)&q_end_d[g_idx][s][cur], BatchSize * sizeof(size_t), streams));
                    // CUDA_CALL(cudaMallocAsync((void**)&s_end_d[g_idx][s][cur], BatchSize * sizeof(size_t), streams));
                    
                    // assert(num_task == num_task_vec[cur]);
                    banded_sw_kernel<<<blocks,threadsPerBlock,0,streams>>>(
                                    num_task,
                                    q_lengths_dev[g_idx], q_offset_dev[g_idx], task_dev[g_idx][s]+it,
                                    query_dev,subj_dev + s_begin_vec[s], s_length[s],
                                    score_d[g_idx][s][cur],
                                    q_end_d[g_idx][s][cur],s_end_d[g_idx][s][cur],
                                    cigar_op_d[g_idx][s][cur],cigar_cnt_d[g_idx][s][cur],cigar_len_d[g_idx][s][cur],
                                    direction_matrix[g_idx][s][cur],tiled_direction_matrix[g_idx][s][cur],band_width,
                                    BLOSUM62_d);
                    CUDA_CALL(cudaEventRecord(kernels_done[g_idx][s][cur], streams));                
                    CUDA_CALL(cudaStreamWaitEvent(copy_streams[cur], kernels_done[g_idx][s][cur], 0));
                    CUDA_CALL(cudaEventDestroy(kernels_done[g_idx][s][cur]));

                    CUDA_CALL(cudaMemcpyAsync(score_h[g_idx][s][cur], score_d[g_idx][s][cur], BatchSize * sizeof(int), cudaMemcpyDeviceToHost, copy_streams[cur]));
                    CUDA_CALL(cudaMemcpyAsync(q_end_h[g_idx][s][cur], q_end_d[g_idx][s][cur], BatchSize * sizeof(size_t), cudaMemcpyDeviceToHost,copy_streams[cur]));
                    CUDA_CALL(cudaMemcpyAsync(s_end_h[g_idx][s][cur], s_end_d[g_idx][s][cur], BatchSize * sizeof(size_t), cudaMemcpyDeviceToHost, copy_streams[cur]));
                    
                    CUDA_CALL(cudaMemcpyAsync(cigar_op_h[g_idx][s][cur], cigar_op_d[g_idx][s][cur], BatchSize * sizeof(char) * MaxAlignLen, cudaMemcpyDeviceToHost,copy_streams[cur]));
                    CUDA_CALL(cudaMemcpyAsync(cigar_cnt_h[g_idx][s][cur], cigar_cnt_d[g_idx][s][cur], BatchSize * sizeof(int) * MaxAlignLen, cudaMemcpyDeviceToHost,copy_streams[cur]));
                    CUDA_CALL(cudaMemcpyAsync(cigar_len_h[g_idx][s][cur], cigar_len_d[g_idx][s][cur], BatchSize * sizeof(int), cudaMemcpyDeviceToHost,copy_streams[cur]));
                    
                    CUDA_CALL(cudaEventRecord(copies_done[g_idx][s][cur], copy_streams[cur]));
                    CUDA_CALL(cudaFreeAsync(direction_matrix[g_idx][s][cur], streams));
                    CUDA_CALL(cudaFreeAsync(tiled_direction_matrix[g_idx][s][cur], streams));
                    // CUDA_CALL(cudaFreeAsync(score_d[g_idx][s][cur], streams));
                    // CUDA_CALL(cudaEventSynchronize((copies_done[g_idx][s][cur])));
                    // for(int i = 0; i < num_task_vec[cur]; ++ i){
                    //     assert(cigar_len_h[i] > 0);
                    // }

                    // CUDA_CALL(cudaEventSynchronize(copies_done[g_idx][s][cur]));
                    // CUDA_CALL(cudaEventDestroy(copies_done[g_idx][s][cur]));
                    report_threads[g_idx][s][cur] = thread(call_results,
                                                    ref(copies_done[g_idx][s][cur]),
                                                    ref(streams),
                                                    // cigar_len_d[g_idx][s][cur], cigar_op_d[g_idx][s][cur],  cigar_cnt_d[g_idx][s][cur], 
                                                    // q_end_d[g_idx][s][cur], s_end_d[g_idx][s][cur],
                                                    // score_d[g_idx][s][cur],
                                                    query, subj[s],
                                                    task_host[g_idx][s]+it, num_task,
                                                    cigar_len_h[g_idx][s][cur], cigar_op_h[g_idx][s][cur], cigar_cnt_h[g_idx][s][cur],
                                                    q_end_h[g_idx][s][cur], s_end_h[g_idx][s][cur],
                                                    score_h[g_idx][s][cur],
                                                    ref(res_s[g][s]), it,
                                                    pool, ref(rs_report[g_idx][s][cur]));
                    
                    
                }
                // CUDA_CALL(cudaDeviceSynchronize());
#else
                result_threads[g_idx][s] = thread(handle_results, ref(seeding_finished[g_idx][s]), query, subj[s], task_host[g_idx][s], task_num_host[g_idx][s], ref(q_groups[g]), s_length[s], s, ref(res_s[g][s]), ref(sw_tasks[g][s]), pool, ref(rs[g_idx][s]));
                cout << "=";
#endif
            }
        }
        CUDA_CALL(cudaStreamSynchronize(streams));
        // CUDA_CALL(cudaDeviceSynchronize());
        nvtxRangePop();
        gettimeofday(&t_end, NULL);
        time_prof.gpu_time += timeuse(t_start, t_end);
        group_time += timeuse(t_start, t_end);
        // cout << "GPU computing Time: " << timeuse(t_start, t_end) << endl;

        if (g_begin == 0)
        {
            gettimeofday(&tt_start, NULL);
            for (int s = 0; s < NUM_STREAM; s++)
            {
                string fname = db_name + "_" + to_string(db_num) + "_" + to_string(s) + ".name";
                int fd = open(fname.data(), O_RDONLY);
                if (fd == -1)
                {
                    std::cerr << "Error opening '" << fname << ". Bailing out." << std::endl;
                    exit(1);
                }
                size_t len = lseek(fd, 0, SEEK_END);
                char *map = (char *)mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
                close(fd);
                s_name[s] = (char *)malloc(len);
                memcpy(s_name[s], map, len);
                munmap(map, len);

                s_num[s] = load_offsets(db_name + "_" + to_string(db_num) + "_" + to_string(s), s_offsets[s], sn_offsets[s]);
            }
            gettimeofday(&t_end, NULL);
            time_prof.name_time += timeuse(tt_start, t_end);
            group_time += timeuse(tt_start, t_end);
            // cout << "Load seqs name Time: " << timeuse(tt_start, t_end) << endl;
        }

        gettimeofday(&tt_start, NULL);

        int hsp_count = 0;

        for (int s = 0; s < NUM_STREAM; s++)
        {
            for (int g = g_begin; g < g_begin + n_groups; g++)
            {
                
                int g_idx = g - g_begin;
                // printf("@@ g=%d|s=%d res_s[g][s]%d task_num_host %d\n",g,s,res_s[g][s].size(),*task_num_host[g_idx][s]);
#ifndef USE_GPU_DIFFUSE
                result_threads[g_idx][s].join();
#endif

#ifdef USE_GPU_DIFFUSE
                for(int cur = 0; cur < MaxNumBatch; ++ cur){
                    size_t it = cur * BatchSize;
                    if(it > *task_num_host[g_idx][s]) break;
                    if (report_threads[g_idx][s][cur].joinable()) {
                        report_threads[g_idx][s][cur].join();
                    }

                    for (auto &r : rs_report[g_idx][s][cur]) {
                        if (r.valid()) {
                            r.get();
                        }
                    }
                }
#endif
                hsp_count += res_s[g][s].size();
                cout << "=";
#ifndef USE_GPU_DIFFUSE
                for (auto &r : rs[g_idx][s])
                    r.get();
#endif
                proceed_result(res, ref(res_s[g][s]), query, subj[s], q_groups[g], s_name[s], s_offsets[s], sn_offsets[s], s_num[s], total_db_size);
                cout << "=";
// #endif
            }
        }

        // g_begin += n_groups;

        gettimeofday(&t_end, NULL);
        time_prof.cpu_time += timeuse(tt_start, t_end);
        group_time += timeuse(tt_start, t_end);
        cout << "] " << group_time << "s, " << hsp_count << " HSPs" << endl;
    }

    gettimeofday(&t_start, NULL);

    // n_groups = q_groups.size();

    // if (n_groups > MAX_GROUPS_PER_ROUND){
    //     n_groups = MAX_GROUPS_PER_ROUND;
    // }
#ifdef USE_GPU_DIFFUSE
    free(score); 
    
    free(s_end); 
    free(q_end); 
    
    free(cigar_op); 
    free(cigar_cnt); 
    free(rd); 
    free(rt); 
    for(int g = 0; g < n_groups; ++ g){

        for(int s = 0; s < NUM_STREAM; ++ s){
            for(int cur = 0; cur < MaxNumBatch; ++ cur){
                CUDA_CALL(cudaFreeHost(score_h[g][s][cur])); 
                
                CUDA_CALL(cudaFreeHost(s_end_h[g][s][cur])); 
                CUDA_CALL(cudaFreeHost(q_end_h[g][s][cur])); 
                
                CUDA_CALL(cudaFreeHost(cigar_op_h[g][s][cur])); 
                CUDA_CALL(cudaFreeHost(cigar_cnt_h[g][s][cur])); 
                CUDA_CALL(cudaFreeHost(cigar_len_h[g][s][cur])); 
                CUDA_CALL(cudaFreeAsync(s_end_d[g][s][cur], streams));
                CUDA_CALL(cudaFreeAsync(q_end_d[g][s][cur], streams));
                CUDA_CALL(cudaFreeAsync(cigar_op_d[g][s][cur], streams));
                CUDA_CALL(cudaFreeAsync(cigar_cnt_d[g][s][cur], streams));
                CUDA_CALL(cudaFreeAsync(cigar_len_d[g][s][cur], streams));
                CUDA_CALL(cudaFreeAsync(score_d[g][s][cur], streams));
            }
        }
    }
    CUDA_CALL(cudaFreeAsync(BLOSUM62_d, malloc_streams)); 
    for(int cur = 0; cur < MaxNumBatch; ++ cur){
        CUDA_CALL(cudaStreamDestroy(copy_streams[cur]));
        
    }   
#endif
    CUDA_CALL(cudaStreamDestroy(streams));
    CUDA_CALL(cudaStreamDestroy(malloc_streams));
    
    for (int g = 0; g < n_groups; g++)
    {
        for (int s = 0; s < NUM_STREAM; s++)
        {
            // destroy_hashtable(pHashTable[s]);
            CUDA_CALL(cudaFree(task_dev[g][s]));
            CUDA_CALL(cudaFree(task_num_dev[g][s]));
        }
        CUDA_CALL(cudaFree(q_num_dev[g]));

        CUDA_CALL(cudaFree(q_idx_dev[g]));

        CUDA_CALL(cudaFree(q_lengths_dev[g]));
        CUDA_CALL(cudaFree(q_offset_dev[g]));

        CUDA_CALL(cudaFree(index_size_dev[g]));

        CUDA_CALL(cudaFree(threshold_dev[g]));
    }

    CUDA_CALL(cudaFree(subj_dev));

    for (int s = 0; s < NUM_STREAM; s++)
    {
        free(s_name[s]);
        free(sn_offsets[s]);
        free(s_offsets[s]);
    }

    gettimeofday(&t_end, NULL);
    time_prof.mem_time += timeuse(t_start, t_end);
    cout << "Free in serch db time: " << timeuse(t_start, t_end) << endl;
}

void blastp(string argv_query, vector<string> argv_dbs, string argv_out)
{
    vector<uint32_t> q_offsets;
    vector<string> q_names;
    char *query;

    struct timeval t_start, t_end;
    gettimeofday(&t_start, NULL);
    uint32_t q_length = load_fasta(argv_query.data(), query, q_offsets, q_names);

    q_offsets.push_back(q_length);

    vector<uint32_t> q_lengths;
    for (int i = 0; i < q_offsets.size() - 1; i++)
    {
        q_lengths.push_back(q_offsets[i + 1] - q_offsets[i] - 1);
    }
    int n_query = q_offsets.size() - 1;
    gettimeofday(&t_end, NULL);
    cout << "Load query Time: " << timeuse(t_start, t_end) << endl;
    gettimeofday(&t_start, NULL);

    vector<SWResult> res_d[n_query];

    size_t max_db_size = 0;
    size_t total_db_size = 0;
    vector<int> db_sizes;
    for (int i = 0; i < argv_dbs.size(); i++)
    {
        db_sizes.push_back(check_db(argv_dbs[i].data(), max_db_size, total_db_size));
        if (db_sizes[i] <= 0)
        {
            cout << "DB " << argv_dbs[i] << " not found!" << endl;
            exit(-1);
        }
    }
    cout << "Max db size = " << max_db_size / (1073741824) << " GB" << endl;
    cout << "Total db size = " << (double)total_db_size / (1073741824) << " GB" << endl;
    total_db_size = (total_db_size * 8) / 5;

    size_t max_hashtable_capacity;
    uint32_t max_n_query;
    vector<QueryGroup> q_groups = init_query_group(n_query, max_db_size, q_lengths, q_offsets, query, max_hashtable_capacity, max_n_query);

    // init_hashtable_capacity(n_query, max_db_size, q_lengths);

    // cout << "Each hash table size = ";
    // for (int i = 0; i < n_query; i++)
    // {
    //     cout << (double)kHashTableCapacity_host[i] * sizeof(KeyValue) * NUM_STREAM / (1073741824) << " ";
    // }
    // cout << "GB, total size = " << (double)kHashTableOffset_host[n_query] * sizeof(KeyValue) * NUM_STREAM / (1073741824) << " GB." << endl;

    uint32_t n_groups = q_groups.size();
    if (n_groups > MAX_GROUPS_PER_ROUND)
        n_groups = MAX_GROUPS_PER_ROUND;
    // KeyValue *hashtable_host[n_groups][NUM_STREAM];
    Task *task_host[n_groups][NUM_STREAM];
    uint32_t *task_num_host[n_groups][NUM_STREAM];

    for (int g = 0; g < n_groups; g++)
    {
        for (int s = 0; s < NUM_STREAM; s++)
        {
            // CUDA_CALL(cudaMallocHost(&hashtable_host[g][s], max_hashtable_capacity * sizeof(KeyValue)));
            CUDA_CALL(cudaMallocHost(&task_host[g][s], MAX_FILTER_TASK * sizeof(Task)));
            CUDA_CALL(cudaMallocHost(&task_num_host[g][s], sizeof(uint32_t)));
        }
    }

    pool = new ThreadPool(num_threads);

    gettimeofday(&t_end, NULL);
    cout << "Prepare Time: " << timeuse(t_start, t_end) << endl;

    TimeProfile time_prof;

    struct timeval c_start, c_end;
    gettimeofday(&c_start, NULL);
    for (int d = 0; d < argv_dbs.size(); d++)
    {
        string db_name(argv_dbs[d]);
        for (int i = 0; i < db_sizes[d]; i++)
        {
            struct timeval start, end;
            gettimeofday(&start, NULL);
            cout << "Search DB " << d + 1 << "/" << argv_dbs.size() << ", Part " << i + 1 << "/" << db_sizes[d] << endl;

            char *subj[NUM_STREAM];
            size_t s_size[NUM_STREAM];
            size_t s_len[NUM_STREAM];

            for (int s = 0; s < NUM_STREAM; s++)
            {
                load_seq(db_name + "_" + to_string(i), s, ref(subj[s]), ref(s_size[s]));
                s_len[s] = (s_size[s] * 8) / 5;
            }

            search_db_batch(query, subj, q_groups, s_len, task_host, task_num_host, max_hashtable_capacity, max_n_query, q_length, db_name, i, res_d, total_db_size, time_prof);

            for (int s = 0; s < NUM_STREAM; s++)
            {
                munmap(subj[s], s_size[s]);
            }
            gettimeofday(&end, NULL);
            cout << "Total Batch Time: " << timeuse(start, end) << endl;
        }
    }
    gettimeofday(&c_end, NULL);
    cout << "Finish searching." << endl;
    cout << "GPU Calculation time:\t" << time_prof.gpu_time << endl;
    cout << "CPU Calculation time:\t" << time_prof.cpu_time << endl;
    cout << "Others time:\t" << time_prof.mem_time << endl;
    cout << "Load seqs name Time:\t" << time_prof.name_time << endl;
    cout << "Total Calculation Time:\t" << timeuse(c_start, c_end) << endl;

    gettimeofday(&t_start, NULL);

    for (int g = 0; g < n_groups; g++)
        for (int s = 0; s < NUM_STREAM; s++)
        {
            CUDA_CALL(cudaFreeHost(task_host[g][s]));
            CUDA_CALL(cudaFreeHost(task_num_host[g][s]));
        }

    gettimeofday(&t_end, NULL);
    cout << "Free memory Time:\t" << timeuse(t_start, t_end) << endl;
    gettimeofday(&t_start, NULL);

    for (int i = 0; i < n_query; i++)
    {
        sort_heap(res_d[i].begin(), res_d[i].end(), [&](const SWResult &sw1, const SWResult &sw2)
                  { return (sw1.e_value == sw2.e_value) ? (sw1.score > sw2.score) : (sw1.e_value < sw2.e_value); });
    }

    int outfmt;
    get_arg("outfmt", outfmt, D_OUTFMT);
    switch (outfmt)
    {
    case 0:
        output_result_tabular(argv_out, res_d, query, q_offsets, q_names);
        break;
    case 1:
        output_result_align(argv_out, res_d, query, q_offsets, q_names);
        break;
    case 2:
        output_result_tabular(argv_out, res_d, query, q_offsets, q_names);
        output_result_fa(argv_out + ".fasta", res_d, query, q_offsets, q_names);
        break;
    case 3:
        output_result_cast(argv_out, res_d, query, q_offsets, q_names);
        break;
    case 4:
        output_result_a3m(argv_out, res_d, query, q_offsets, q_names);
        break;
    case 5:
        output_result_reduce(argv_out, res_d, query, q_offsets, q_names);
        break;
    default:
        break;
    }

    free(query);
    delete pool;

    gettimeofday(&t_end, NULL);

    cout << "Output Time:\t" << timeuse(t_start, t_end) << endl;

    cout << "Finished." << endl;
}
