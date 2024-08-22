#include "blastp.h"
#include <assert.h>
#include <chrono>
#include <algorithm> 
#include <nvtx3/nvToolsExt.h>

#define PACK_KEY(k) ((k & ~0x7) | 0x3)

ThreadPool *pool;
// mutex mu2;

queue<banded_sw_task> banded_sw_task_queue;
queue<report_task> report_task_queue;
std::mutex queue_mutex, mu2;
std::condition_variable sw_beg_cv, sw_end_cv, report_beg_cv, report_end_cv;
// std::condition_variable sw_end_cv;
atomic<bool> bw_finished ={0};
atomic<bool> produce_end= {0};

atomic<bool> report_finish ={0};
atomic<bool> report_end= {0};

// vector<SWResult> res_s[MAX_GROUPS_PER_ROUND][NUM_STREAM];

__constant__ uint32_t kHashTableCapacity_dev[MAX_GROUPS_PER_ROUND][MAX_QUERY_PER_GROUP];
__constant__ uint32_t kHashTableOffset_dev[MAX_GROUPS_PER_ROUND][MAX_QUERY_PER_GROUP];

__constant__ int SEED_LENGTH;
__constant__ int QIT_WIDTH;
__constant__ uint32_t MASK;


int checkGPUUtilization() {
    nvmlInit();
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(0, &device);  // Assuming you're using GPU 0
    
    nvmlUtilization_t utilization;
    nvmlDeviceGetUtilizationRates(device, &utilization);
    
   
    nvmlShutdown();
    return utilization.gpu;
}
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
                uint32_t max_len_query,
                int NumTasks,
                uint32_t* q_lens, uint32_t* q_idxs, Task* task,
                const char* query, const char* target, size_t target_len,
                int * max_score,
                size_t* q_end_idx, size_t* s_end_idx,
                char* cigar_op, int* cigar_cnt,int* cigar_len,
                uint8_t *rd, record* rt,int band_width,
                int* BLOSUM62){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx >= NumTasks) return;

    size_t query_len = q_lens[task[idx].q_id];
    assert( query_len < max_len_query);
   
    size_t q_idx  = q_idxs[task[idx].q_id];
    size_t diag  = task[idx].key;
    
    int64_t c_begin = (int64_t)diag - band_width - query_len + 2;
    // size_t c_end = diag + band_width;
    
    record* tile = rt + idx * MaxBW * (TILE_SIZE + 1);

    size_t width = 2 * band_width + 1;
    size_t height = max_len_query + 1;
    assert(width < MaxBW);

    //init:
    max_score[idx] = 0;
    
    size_t t_height = TILE_SIZE + 1;
    
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

                // int byteIndex = (calIndex(_c, _q+q_offset,height) * BatchSize + idx) >> 2; // 4 bits per byte
                // int bitOffset = ((calIndex(_c, _q+q_offset,height) * BatchSize + idx) & 0x03) << 1;
                // rd[byteIndex] &= ~(0x03 << bitOffset);
                // rd[byteIndex] |= (score?( \
                //         (score == tile[calIndex(_q,_c,MaxBW)].m) ? DIAG << bitOffset : \
                //      ((score == tile[calIndex(_q,_c,MaxBW)].y) ? LEFT << bitOffset: TOP << bitOffset )):0);

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
    // int byteIndex = (BatchSize * calIndex(cur_c,cur_q,height) + idx) / 4;
    // int bitOffset = ( (BatchSize * calIndex(cur_c,cur_q,height) + idx) & 0x03) << 1;
    // while ((rd[byteIndex] >> (bitOffset)) & 0x03 && cigar_cur_len < MaxAlignLen)
    while (rd[BatchSize * calIndex(cur_c,cur_q,height) + idx])
    {
        // byteIndex = (BatchSize * calIndex(cur_c,cur_q,height) + idx) / 4;
        // bitOffset = (BatchSize * calIndex(cur_c,cur_q,height) + idx) % 4;
        // int d = (rd[byteIndex] >> (bitOffset)) & 0x03;

        int d = rd[BatchSize * calIndex(cur_c,cur_q,height) + idx];
        // size_t res_q = (d&0x01) ? (cur_q + q_idx) : (size_t)-1;
        // size_t res_c = (d&0x02) ? (c_begin + cur_c + cur_q) : (size_t)-1;
        
        // q_res_d[idx* MaxAlignLen + (cnt_q)] = (res_q);
        // s_res_d[idx* MaxAlignLen + (cnt_c)] = (res_c);
        int cur_cigar_cnt = 0;
        // while (((rd[byteIndex] >> bitOffset) & 0x03)==d){
        while (rd[BatchSize * calIndex(cur_c,cur_q,height) + idx] && rd[BatchSize * calIndex(cur_c,cur_q,height) + idx]==d){
            cur_cigar_cnt ++;
            
            //TOP 01b, left 10b, diag 11b
            //DIAG : cur_q -= 1
            //TOP : cur_q -= 1, cur_c += 1;
            //LEFT : cur_c -= 1
            cur_q -= (d == DIAG || d == TOP);
            cur_c += (d == TOP); // Increment cur_c if TOP (01b)
            cur_c -= (d == LEFT); // Decrement cur_c if LEFT (10b)
            // byteIndex = (BatchSize * calIndex(cur_c,cur_q,height) + idx) / 4;
            // bitOffset = (BatchSize * calIndex(cur_c,cur_q,height) + idx) % 4;
        }
        (cigar_cnt + idx * MaxAlignLen)[cigar_cur_len] = cur_cigar_cnt;
        (cigar_op + idx * MaxAlignLen)[cigar_cur_len++] = ((d==DIAG)?'M':((d==TOP)?'D':'I'));
    }

    // free(rt);
    // printf("@@ cigar_len %d %d\n", idx,cigar_len);
    assert(cigar_cur_len > 0 && cigar_cur_len <= MaxAlignLen);
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
                // cudaStream_t& stream,
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


void Schedule(queue<banded_sw_task>& banded_sw_task_queue, ThreadPool* sw_pool){
    // double CPU_time = 0;
    // int cpu_cnt = 0, gpu_cnt = 0;
    // struct timeval cpu_schedule_beg, cpu_schedule_end, schedule_beg, schedule_end;
    // gettimeofday(&schedule_beg, NULL);
    cudaStream_t copy_stream;
    cudaStreamCreate(&copy_stream);
    while(true){
        
        std::unique_lock<std::mutex> queue_lock(queue_mutex);
        sw_beg_cv.wait(queue_lock,[&]{
            return !banded_sw_task_queue.empty() || produce_end.load();
        });
        if(banded_sw_task_queue.empty() && produce_end.load()){
            bw_finished = 1;
            queue_lock.unlock();
            sw_end_cv.notify_one();
            break;
        }
        banded_sw_task& cur =  banded_sw_task_queue.front();
        banded_sw_task_queue.pop();
        queue_lock.unlock();
        // printf("begin task\n");
        if(checkGPUUtilization() < 50 && cur.num_task > (BatchSize >> 1) && produce_end.load()){
            // printf("GPU Task\n");
            // gpu_cnt += 1;
            uint8_t* direction_matrix;
            record* tiled_direction_matrix;
            // CUDA_CALL(cudaMallocAsync((void**)&direction_matrix, ((MaxBW * (cur.max_len_query + 1) * BatchSize) >> 1) * sizeof(unsigned char), cur.stream));
            CUDA_CALL(cudaMallocAsync((void**)&direction_matrix, MaxBW * (cur.max_len_query + 1) * BatchSize * sizeof(uint8_t), cur.stream));
            CUDA_CALL(cudaMallocAsync((void**)&tiled_direction_matrix, MaxBW * (TILE_SIZE + 1) * BatchSize * sizeof(record), cur.stream));

            // Initialize memory
            CUDA_CALL(cudaMemsetAsync(direction_matrix, 0, MaxBW * (cur.max_len_query + 1) * BatchSize  * sizeof(uint8_t), cur.stream));
            CUDA_CALL(cudaMemsetAsync(tiled_direction_matrix, 0, MaxBW * (TILE_SIZE + 1) * BatchSize * sizeof(record), cur.stream));
            
           banded_sw_kernel<<<cur.blocks, cur.threadsPerBlock,0,cur.stream>>>(
                                cur.max_len_query,
                                cur.num_task,
                                cur.q_lens_d, cur.q_idxs_d, cur.task_d,
                                cur.query_d, cur.target_d, cur.target_len_d,
                                cur.max_score_d,
                                cur.q_end_idx_d, cur.s_end_idx_d,
                                cur.cigar_op_d, cur.cigar_cnt_d,cur.cigar_len_d,
                                direction_matrix, tiled_direction_matrix, cur.band_width,
                                cur.BLOSUM62_d);
            cudaEvent_t kernels_done;
            CUDA_CALL(cudaEventCreate(&kernels_done));
            cudaEventRecord(kernels_done, cur.stream);
            cudaStreamWaitEvent(copy_stream, kernels_done, 0);
            cudaEventDestroy(kernels_done);

            CUDA_CALL(cudaMemcpyAsync(cur.max_score_h, cur.max_score_d, BatchSize * sizeof(int), cudaMemcpyDeviceToHost, copy_stream));
            CUDA_CALL(cudaMemcpyAsync(cur.q_end_idx_h, cur.q_end_idx_d, BatchSize * sizeof(size_t), cudaMemcpyDeviceToHost,copy_stream));
            CUDA_CALL(cudaMemcpyAsync(cur.s_end_idx_h, cur.s_end_idx_d, BatchSize * sizeof(size_t), cudaMemcpyDeviceToHost, copy_stream));
            
            CUDA_CALL(cudaMemcpyAsync(cur.cigar_op_h, cur.cigar_op_d, BatchSize * sizeof(char) * MaxAlignLen, cudaMemcpyDeviceToHost,copy_stream));
            CUDA_CALL(cudaMemcpyAsync(cur.cigar_cnt_h, cur.cigar_cnt_d, BatchSize * sizeof(int) * MaxAlignLen, cudaMemcpyDeviceToHost,copy_stream));
            CUDA_CALL(cudaMemcpyAsync(cur.cigar_len_h, cur.cigar_len_d, BatchSize * sizeof(int), cudaMemcpyDeviceToHost,copy_stream));
            
            // cudaEventRecord(cur.copies_done, cur.copy_stream);
            
            CUDA_CALL(cudaFreeAsync(tiled_direction_matrix, cur.stream));
            CUDA_CALL(cudaFreeAsync(direction_matrix, cur.stream));
            
        }else{
            // cpu_cnt += 1;
            // printf("CPU Task\n");
            // gettimeofday(&cpu_schedule_beg, NULL);
            // cudaEventRecord(cur.copies_done, cur.copy_stream);
           
            banded_sw_cpu_kernel_thread_pool(cur.max_len_query, cur.num_task,
                                cur.q_lens_h, cur.q_idxs_h, cur.task_h,
                                cur.query_h, cur.target_h, cur.target_len_h,
                                cur.max_score_h,
                                cur.q_end_idx_h, cur.s_end_idx_h,
                                cur.cigar_op_h, cur.cigar_cnt_h,cur.cigar_len_h,
                                cur.band_width,
                                cur.BLOSUM62_h,sw_pool);
            
            // gettimeofday(&cpu_schedule_end, NULL);
            // CPU_time += timeuse(cpu_schedule_beg, cpu_schedule_end);
        }
        // banded_sw_task_queue.pop();
        // printf("end task\n");
    }
    sw_pool->wait();
    cudaStreamSynchronize(copy_stream);
    // cudaStreamDestroy(copy_stream);
    // gettimeofday(&schedule_end, NULL);
    // printf("schedule CPU schedule Time: %f,  CPU cnt: %d, GPU cnt %d\n", CPU_time, cpu_cnt , gpu_cnt );
    // printf("schedule Time: %f\n", timeuse(schedule_beg, schedule_end));
    return;
}
void _report(report_task& cur){
    for(size_t idx = 0; idx < cur.num_task; ++ idx){
        pool->enqueue([=] {
            cigar_to_index_and_report(idx, cur.begin, 
                                    cur.cigar_len, cur.cigar_op, cur.cigar_cnt,
                                    cur.q_start, cur.c_start,
                                    cur.res_s,
                                    cur.score,
                                    cur.task,
                                    cur.query, cur.target);
        });
    }
}
void Schedule_report(queue<report_task>& report_task_queue, ThreadPool* pool){
    struct timeval cpu_schedule_beg, cpu_schedule_end, schedule_beg, schedule_end;
    gettimeofday(&schedule_beg, NULL);
       
    while(true){
        
        std::unique_lock<std::mutex> queue_lock(mu2);
        report_beg_cv.wait(queue_lock,[&]{
            return !report_task_queue.empty() || report_end.load();
        });
        if(report_task_queue.empty() && report_end.load()){
            report_finish = 1;
            queue_lock.unlock();
            report_end_cv.notify_one();
            break;
        }
        report_task& cur =  report_task_queue.front();
        report_task_queue.pop();
        queue_lock.unlock();
        _report(cur);
    }
}

void search_db_batch(ThreadPool* sw_pool, uint32_t max_len_query, char *query, char *subj[],\
                    vector<QueryGroup> &q_groups, size_t s_length[],\
                    Task *task_host[][NUM_STREAM], uint32_t *task_num_host[][NUM_STREAM],\
                    size_t max_hashtable_capacity, uint32_t max_n_query, uint32_t total_len_query,\
                    string db_name, uint32_t db_num, vector<SWResult> *res, \
                    size_t total_db_size, TimeProfile &time_prof,\
                    
                    KeyValue *pHashTable[][NUM_STREAM],
                    Task *task_dev[][NUM_STREAM],
                    uint32_t *task_num_dev[][NUM_STREAM],

                    
                    int *q_num_dev[],
                    int *q_idx_dev[],
                    uint32_t *q_lengths_dev[],
                    uint8_t *index_size_dev[],
                    uint32_t *threshold_dev[],
                    uint32_t *q_offset_dev[],
                    uint32_t kHashTableCapacity_host[][MAX_QUERY_PER_GROUP],
                    uint32_t kHashTableOffset_host[][MAX_QUERY_PER_GROUP],
                    
                    char *s_name[NUM_STREAM],
                    size_t *s_offsets[NUM_STREAM],
                    size_t *sn_offsets[NUM_STREAM],
                    size_t* s_num,
                   
                    
                    char* query_dev,
                    cudaStream_t& streams,
                    cudaStream_t& malloc_streams,
                    cudaStream_t& sw_stream,
                    // cudaStream_t& copy_stream,
                    // int* direction_matrix[][NUM_STREAM][MaxNumBatch],
                    // record* tiled_direction_matrix[][NUM_STREAM][MaxNumBatch],
                    int* BLOSUM62_d,
                    const int* BLOSUM62,
                    
                    int* score_d[][NUM_STREAM][MaxNumBatch], 
                    int* score_h[][NUM_STREAM][MaxNumBatch],
                    
                    size_t* q_end_d[][NUM_STREAM][MaxNumBatch],
                    size_t*q_end_h[][NUM_STREAM][MaxNumBatch],
                    
                    size_t* s_end_d[][NUM_STREAM][MaxNumBatch],
                    size_t*s_end_h[][NUM_STREAM][MaxNumBatch],
                    
                    char* cigar_op_d[][NUM_STREAM][MaxNumBatch],
                    char*cigar_op_h[][NUM_STREAM][MaxNumBatch],
                    
                    int*cigar_cnt_d[][NUM_STREAM][MaxNumBatch], 
                    int* cigar_cnt_h[][NUM_STREAM][MaxNumBatch],
                    
                    int* cigar_len_d[][NUM_STREAM][MaxNumBatch], 
                    int*cigar_len_h[][NUM_STREAM][MaxNumBatch]
                    
                    )
{
    struct timeval t_start, t_end, tt_start;

    gettimeofday(&t_start, NULL);
    int threadsPerBlock = 128;  // 根据 shared memory 限制调整
    int blocks = (BatchSize + threadsPerBlock - 1) / threadsPerBlock;
    

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

    for (int g = 0; g < n_groups; g++)
    {
        for (int s = 0; s < NUM_STREAM; s++)
        {
            // pHashTable[s] = create_hashtable(max_hashtable_capacity);
            CUDA_CALL(cudaMemset(task_dev[g][s], 0, MAX_FILTER_TASK * sizeof(Task)));
            CUDA_CALL(cudaMemset(task_num_dev[g][s], 0, sizeof(uint32_t)));
        }
    }

    int mingridsize_seeding, mingridsize_filter, mingridsize_sw;
    int threadblocksize_seeding, threadblocksize_filter, threadblocksize_sw;
    CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&mingridsize_seeding, &threadblocksize_seeding, seeding_kernel, 0, 0));
    CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&mingridsize_filter, &threadblocksize_filter, filter_kernel, 0, 0));
    CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&mingridsize_sw, &threadblocksize_sw, banded_sw_kernel, 0, 0));

    // cout << "Seeding Block size:" << threadblocksize_seeding <<"," << mingridsize_seeding <<endl;
    // cout << "Filter Block size:" << threadblocksize_filter <<"," << mingridsize_filter <<endl;

    vector<SWResult> res_s[q_groups.size()][NUM_STREAM];
    size_t free_byte, total_byte;
    CUDA_CALL(cudaMemGetInfo(&free_byte, &total_byte));
    cout << "GPU mem: " << (double)(total_byte - free_byte) / (1073741824) << " GB / " << (double)total_byte / (1073741824) << " GB" << endl;

    size_t s_begin = 0;
    vector<size_t> s_begin_vec;
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
        double group_time = 0;
        cout << "Group " << g_begin + 1 << "/" << q_groups.size() << "\t[";
        gettimeofday(&t_start, NULL);
        n_groups = q_groups.size() - g_begin;
        if (n_groups > MAX_GROUPS_PER_ROUND)
            n_groups = MAX_GROUPS_PER_ROUND;
    
        bw_finished = 0;
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
        
        // cudaEvent_t copies_done[n_groups][NUM_STREAM][MaxNumBatch];
        // cudaEvent_t kernels_done[n_groups][NUM_STREAM][MaxNumBatch];

        cudaEvent_t malloc_finished;
        cudaEvent_t seeding_finished[n_groups][NUM_STREAM];
        

        gettimeofday(&t_end, NULL);
        group_time += timeuse(t_start, t_end);
        time_prof.mem_time += timeuse(t_start, t_end);
        // cout << "Prepare mem and data Time: " << timeuse(t_start, t_end) << endl;

        gettimeofday(&t_start, NULL);

        assert(banded_sw_task_queue.size()==0);
        thread consumer_thread(Schedule, ref(banded_sw_task_queue), sw_pool);

        for (int g = g_begin; g < g_begin + n_groups; g++)
        {
            int g_idx = g - g_begin;

            for (int s = 0; s < NUM_STREAM; s++)
            {
                CUDA_CALL(cudaEventCreate(&malloc_finished));
                CUDA_CALL(cudaEventCreate(&seeding_finished[g_idx][s]));

                // for(int cur = 0; cur < MaxNumBatch; ++ cur){
                //     CUDA_CALL(cudaEventCreate(&copies_done[g_idx][s][cur]));
                //     CUDA_CALL(cudaEventCreate(&kernels_done[g_idx][s][cur]));
                // }
                
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
                    }
                    // num_task_vec.push_back(num_task);
                    
                    banded_sw_task tmp = { max_len_query, num_task,band_width,it, ref(res_s[g][s]),
                                    threadsPerBlock,
                                    blocks,
                                    // threadblocksize_sw,
                                    // mingridsize_sw,
                                    q_lengths_dev[g_idx], q_offset_dev[g_idx], task_dev[g_idx][s]+it,
                                    query_dev,subj_dev + s_begin_vec[s], s_length[s],
                                    score_d[g_idx][s][cur],
                                    q_end_d[g_idx][s][cur],s_end_d[g_idx][s][cur],
                                    cigar_op_d[g_idx][s][cur],cigar_cnt_d[g_idx][s][cur],cigar_len_d[g_idx][s][cur],
                                    // direction_matrix[g_idx][s][cur],tiled_direction_matrix[g_idx][s][cur],
                                    BLOSUM62_d,
                                    q_groups[g].length, q_groups[g].offset, task_host[g_idx][s] + it,
                                    query,subj[s], s_length[s],
                                    score_h[g_idx][s][cur],
                                    q_end_h[g_idx][s][cur], s_end_h[g_idx][s][cur],
                                    cigar_op_h[g_idx][s][cur], cigar_cnt_h[g_idx][s][cur], cigar_len_h[g_idx][s][cur],
                                    // rd, rt, 
                                    BLOSUM62,
                                    ref(sw_stream)
                                    // ref(kernels_done[g_idx][s][cur]),
                                    // ref(copy_streams[cur]),
                                    // ref(copies_done[g_idx][s][cur]),
                                    };
                    std::unique_lock<std::mutex> queue_lock(queue_mutex);
                    banded_sw_task_queue.push(tmp);
                    queue_lock.unlock();
                    sw_beg_cv.notify_one();
                }

            }
        }

        struct timeval schedule_beg, schedule_end, schedule_reset;
        gettimeofday(&schedule_beg, NULL);
        produce_end = 1;
        std::unique_lock<std::mutex> queue_lock(queue_mutex);
        sw_end_cv.wait(queue_lock, [&]{ return bw_finished.load(); });
        queue_lock.unlock();
        bw_finished = 0;
        produce_end = 0;
        gettimeofday(&schedule_reset, NULL);

        // CUDA_CALL(cudaStreamSynchronize(streams));
        // CUDA_CALL(cudaStreamSynchronize(streams));
        // CUDA_CALL(cudaStreamSynchronize(sw_stream));
        // CUDA_CALL(cudaStreamSynchronize(copy_stream));

        
        consumer_thread.join();
        
        gettimeofday(&schedule_end, NULL);
        // cout << "synchronize:reset Time: " << timeuse(schedule_beg, schedule_reset) << endl;
        // cout << "synchronize Time: " << timeuse(schedule_beg, schedule_end) << endl;

        gettimeofday(&t_end, NULL);
        time_prof.gpu_time += timeuse(t_start, t_end);
        group_time += timeuse(t_start, t_end);
        // cout << "GPU computing Time: " << timeuse(t_start, t_end) << endl;

        struct timeval report_start_time, report_end_time;
        gettimeofday(&report_start_time, NULL);
        thread report_thread(Schedule_report, ref(report_task_queue), pool);

        for (int g = g_begin; g < g_begin + n_groups; g++)
        {
            int g_idx = g - g_begin;

            for (int s = 0; s < NUM_STREAM; s++)
            {
                for(size_t cur = 0; cur < MaxNumBatch; cur ++){
                    size_t it = cur * BatchSize;
                    int num_task = BatchSize;
                    if(it > *task_num_host[g_idx][s]) break;
                    if(it+BatchSize >= *task_num_host[g_idx][s]){
                        num_task =  *task_num_host[g_idx][s] - it;
                    }
                    report_task tmp = {num_task, it, 
                        cigar_len_h[g_idx][s][cur], cigar_op_h[g_idx][s][cur], cigar_cnt_h[g_idx][s][cur],
                        q_end_h[g_idx][s][cur], s_end_h[g_idx][s][cur],
                        ref(res_s[g][s]),
                        score_h[g_idx][s][cur],
                        task_host[g_idx][s]+it,
                        query, subj[s]
                    };
                    std::unique_lock<std::mutex> queue_lock(mu2);
                    report_task_queue.push(tmp);
                    queue_lock.unlock();
                    report_beg_cv.notify_one();
                    
                }
            }
        }
        report_end = 1;
        std::unique_lock<std::mutex> report_lock(mu2);
        report_end_cv.wait(report_lock, [&]{ return report_finish.load(); });
        report_lock.unlock();
        report_finish = 0;
        report_end = 0;
        

        gettimeofday(&report_end_time, NULL);
        time_prof.cpu_time +=  timeuse(report_start_time, report_end_time);
        // cout << "Report Time: " << timeuse(report_start_time, report_end_time) << endl;

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

        pool->wait();
        report_thread.join();

        for (int s = 0; s < NUM_STREAM; s++)
        {
            for (int g = g_begin; g < g_begin + n_groups; g++)
            {
                
                int g_idx = g - g_begin;
                // printf("@@ g=%d|s=%d res_s[g][s]%d task_num_host %d\n",g,s,res_s[g][s].size(),*task_num_host[g_idx][s]);

                hsp_count += res_s[g][s].size();
                cout << "=";
                proceed_result(res, ref(res_s[g][s]), query, subj[s], q_groups[g], s_name[s], s_offsets[s], sn_offsets[s], s_num[s], total_db_size);
                cout << "=";
            }
        }

        // g_begin += n_groups;

        gettimeofday(&t_end, NULL);
        time_prof.cpu_time += timeuse(tt_start, t_end);
        group_time += timeuse(tt_start, t_end);
        cout << "] " << group_time << "s, " << hsp_count << " HSPs" << endl;
    }

    gettimeofday(&t_start, NULL);
    CUDA_CALL(cudaFree(subj_dev));

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
    uint32_t max_len_query = 0;
    vector<QueryGroup> q_groups = init_query_group(n_query, max_db_size, q_lengths, q_offsets, query, max_hashtable_capacity, max_n_query, max_len_query);

    // init_hashtable_capacity(n_query, max_db_size, q_lengths);

    // cout << "Each hash table size = ";
    // for (int i = 0; i < n_query; i++)
    // {
    //     cout << (double)kHashTableCapacity_host[i] * sizeof(KeyValue) * NUM_STREAM / (1073741824) << " ";
    // }
    // cout << "GB, total size = " << (double)kHashTableOffset_host[n_query] * sizeof(KeyValue) * NUM_STREAM / (1073741824) << " GB." << endl;
    pool = new ThreadPool(num_threads);
    ThreadPool* sw_pool = new ThreadPool(num_threads);

    gettimeofday(&t_end, NULL);
    cout << "Prepare Time: " << timeuse(t_start, t_end) << endl;

    TimeProfile time_prof;

    struct timeval c_start, c_end, overhead_beg, overhead_end;
    gettimeofday(&overhead_beg, NULL);

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


    CUDA_CALL(cudaMemcpyToSymbol(SEED_LENGTH, &seed_length, sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(QIT_WIDTH, &qit_width, sizeof(int)));
    uint32_t mask = (uint32_t)pow(2, 5 * seed_length) - 1;
    CUDA_CALL(cudaMemcpyToSymbol(MASK, &mask, sizeof(uint32_t)));


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
            CUDA_CALL(cudaMalloc((void **)&task_num_dev[g][s], sizeof(uint32_t)));
        }
    }
    char *s_name[NUM_STREAM] = {0};
    size_t *s_offsets[NUM_STREAM] = {0};
    size_t *sn_offsets[NUM_STREAM] = {0};
    size_t s_num[NUM_STREAM] = {0};

    char* query_dev;
    CUDA_CALL(cudaMalloc((void **)&query_dev, q_length));
    CUDA_CALL(cudaMemcpy(query_dev, query, q_length, cudaMemcpyHostToDevice));
    cudaStream_t streams;
    cudaStream_t malloc_streams;
    cudaStream_t sw_stream, copy_stream;
    CUDA_CALL(cudaStreamCreate(&streams));
    CUDA_CALL(cudaStreamCreate(&malloc_streams));
    CUDA_CALL(cudaStreamCreate(&sw_stream));
    CUDA_CALL(cudaStreamCreate(&copy_stream));
    
    // int direct_matrixSize = (max_len_query+1) * MaxBW;
    // int threadsPerBlock = 128;  // 根据 shared memory 限制调整
    // int blocks = (BatchSize + threadsPerBlock - 1) / threadsPerBlock;
    
    int* BLOSUM62_d;
    int* score_d[n_groups][NUM_STREAM][MaxNumBatch], *score_h[n_groups][NUM_STREAM][MaxNumBatch];
    
    size_t* q_end_d[n_groups][NUM_STREAM][MaxNumBatch], *q_end_h[n_groups][NUM_STREAM][MaxNumBatch];
    size_t* s_end_d[n_groups][NUM_STREAM][MaxNumBatch], *s_end_h[n_groups][NUM_STREAM][MaxNumBatch];
    char* cigar_op_d[n_groups][NUM_STREAM][MaxNumBatch], *cigar_op_h[n_groups][NUM_STREAM][MaxNumBatch];
    int *cigar_cnt_d[n_groups][NUM_STREAM][MaxNumBatch], *cigar_cnt_h[n_groups][NUM_STREAM][MaxNumBatch];
    int* cigar_len_d[n_groups][NUM_STREAM][MaxNumBatch], *cigar_len_h[n_groups][NUM_STREAM][MaxNumBatch];

    
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

                CUDA_CALL(cudaMallocAsync((void**)&cigar_op_d[g][s][cur], BatchSize * sizeof(char) * MaxAlignLen, sw_stream));
                CUDA_CALL(cudaMallocAsync((void**)&cigar_cnt_d[g][s][cur], BatchSize * sizeof(int) * MaxAlignLen, sw_stream));

                CUDA_CALL(cudaMallocAsync((void**)&cigar_len_d[g][s][cur], BatchSize * sizeof(int), sw_stream));

                CUDA_CALL(cudaMallocAsync((void**)&score_d[g][s][cur], BatchSize * sizeof(int), sw_stream));
                
                CUDA_CALL(cudaMallocAsync((void**)&q_end_d[g][s][cur], BatchSize * sizeof(size_t), sw_stream));
                CUDA_CALL(cudaMallocAsync((void**)&s_end_d[g][s][cur], BatchSize * sizeof(size_t), sw_stream));
                   
        
            }
        }
    }
    CUDA_CALL(cudaMallocAsync((void**)&BLOSUM62_d, 26 * 26 * sizeof(int), sw_stream));
    CUDA_CALL(cudaMemcpyAsync(BLOSUM62_d, BLOSUM62, 26 * 26 * sizeof(int),cudaMemcpyHostToDevice,sw_stream));
    gettimeofday(&overhead_end,NULL);
    
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
            // 与d,i 有关的： s_len，db_name,i
            search_db_batch(sw_pool, max_len_query, query, subj, q_groups, s_len, task_host, task_num_host, max_hashtable_capacity, max_n_query, q_length, db_name, i, res_d, total_db_size, time_prof,\
                            pHashTable, task_dev, task_num_dev,
                            q_num_dev, q_idx_dev, q_lengths_dev, index_size_dev, threshold_dev, q_offset_dev,
                            kHashTableCapacity_host, kHashTableOffset_host,
                            s_name, s_offsets, sn_offsets, s_num,
                            query_dev, ref(streams), ref(malloc_streams), ref(sw_stream), 
                            BLOSUM62_d,
                            BLOSUM62,
                            score_d, score_h,\
                            q_end_d, q_end_h,
                            s_end_d, s_end_h,
                            cigar_op_d, cigar_op_h,
                            cigar_cnt_d, cigar_cnt_h,
                            cigar_len_d, cigar_len_h
                            );

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

    cout << "CUDA Mem Management Time:\t" << timeuse(overhead_beg, overhead_end) << endl;

    gettimeofday(&t_start, NULL);

    for (int g = 0; g < n_groups; g++)
        for (int s = 0; s < NUM_STREAM; s++)
        {
            CUDA_CALL(cudaFreeHost(task_host[g][s]));
            CUDA_CALL(cudaFreeHost(task_num_host[g][s]));
        }
    for(int g = 0; g < n_groups; ++ g){

        for(int s = 0; s < NUM_STREAM; ++ s){
            for(int cur = 0; cur < MaxNumBatch; ++ cur){
                CUDA_CALL(cudaFreeHost(score_h[g][s][cur])); 
                
                CUDA_CALL(cudaFreeHost(s_end_h[g][s][cur])); 
                CUDA_CALL(cudaFreeHost(q_end_h[g][s][cur])); 
                
                CUDA_CALL(cudaFreeHost(cigar_op_h[g][s][cur])); 
                CUDA_CALL(cudaFreeHost(cigar_cnt_h[g][s][cur])); 
                CUDA_CALL(cudaFreeHost(cigar_len_h[g][s][cur])); 
                
                CUDA_CALL(cudaFreeAsync(s_end_d[g][s][cur], sw_stream));
                CUDA_CALL(cudaFreeAsync(q_end_d[g][s][cur], sw_stream));
                CUDA_CALL(cudaFreeAsync(cigar_op_d[g][s][cur], sw_stream));
                CUDA_CALL(cudaFreeAsync(cigar_cnt_d[g][s][cur], sw_stream));
                CUDA_CALL(cudaFreeAsync(cigar_len_d[g][s][cur], sw_stream));
                CUDA_CALL(cudaFreeAsync(score_d[g][s][cur], sw_stream));
            }
        }
    }
    CUDA_CALL(cudaFreeAsync(BLOSUM62_d, sw_stream)); 

    CUDA_CALL(cudaStreamDestroy(streams));
    CUDA_CALL(cudaStreamDestroy(malloc_streams));
    CUDA_CALL(cudaStreamDestroy(sw_stream));
    CUDA_CALL(cudaStreamDestroy(copy_stream));
    
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


    for (int s = 0; s < NUM_STREAM; s++)
    {
        free(s_name[s]);
        free(sn_offsets[s]);
        free(s_offsets[s]);
    }

    free(query);
    pool->wait();
    delete pool;
    delete sw_pool;

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


    

    gettimeofday(&t_end, NULL);

    cout << "Output Time:\t" << timeuse(t_start, t_end) << endl;

    cout << "Finished." << endl;
}
