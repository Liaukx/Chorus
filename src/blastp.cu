#include "blastp.h"
#include <assert.h>
#include <chrono>
#include <algorithm> 

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
// TODO error in Score 
__global__ void banded_sw_kernel(uint32_t* q_lens, uint32_t* q_idxs,
                //  size_t* diags,
                Task* task,
                const char* q, const char* c, size_t c_len,
                int *rd, record* rt_d,int band_width,
                int * score_d,
                // int* q_len_d,int* s_len_d,
                size_t* q_end_d, size_t* s_end_d,
                char* cigar_op_d, int* cigar_cnt_d,int* cigar_len_d,
                int* BLOSUM62_d){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    size_t n = q_lens[task[idx].q_id];
    if(n > MaxQueryLen) printf("## Query Len %ld\n", n);
    assert( n < MaxQueryLen);
   
    size_t q_idx  = q_idxs[task[idx].q_id];
    size_t diag  = task[idx].key;
    
    int64_t c_begin = (int64_t)diag - band_width - n + 2;
    size_t c_end = diag + band_width;
    int* BLOSUM62 = BLOSUM62_d + idx * 26 * 26;
    record* rt = rt_d + idx * MaxBW * (TILE_SIZE + 1);
    
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
    
    size_t t_height = TILE_SIZE + 1;
    
    // record *rt = (record *)malloc(width * t_height * sizeof(record));
    // memset(rt, 0, width * t_height * sizeof(record));

    size_t max_q = 0;
    size_t max_c = 0;
    int score = 0, Score = 0;
    // cal maxScore and it's position
    for (size_t it = 0; it * TILE_SIZE < n; it++) {
        
        size_t q_offset = it * TILE_SIZE;

        for(size_t _q = 0; _q < t_height-1 && q_offset + _q < n; ++_q){
            for(size_t _c = 0; _c < width-2; ++_c){
                
                if(c_begin + _c+ q_offset + _q < 0) continue;
                if(c_begin + _c+ q_offset + _q >= c_len) break;

                char chq = q[q_idx + q_offset + _q];
                char chc = get_char_d(c, c_begin + q_offset + _c + _q);
                
                if (chq == END_SIGNAL || chc == END_SIGNAL)
                {
                    continue;
                }
                //rt(_q,_c) -> (_q+1) * width + _c + 1
                // logical m(_q,_c).x = max(m(_q-1,_c).x + SCORE_GAP_EXT, m(_q-1,_c).m +SCORE_GAP, 0 );
                // logical m(_q,_c).y = max(m(_q,_c-1).y + SCORE_GAP_EXT, m(_q,_c-1).m +SCORE_GAP, 0 );
                // logical m(_q,_c).m = max(m(_q-1,_c-1).y,m(_q-1,_c-1).x,m(_q-1,_c-1).m, 0 );
                
                rt[calIndex(_q,_c,MaxBW)].x = max3(rt[calTop(_q,_c,MaxBW)].x + SCORE_GAP_EXT,  rt[calTop(_q,_c,MaxBW)].m + SCORE_GAP, 0);
                rt[calIndex(_q,_c,MaxBW)].y = max3(rt[calLeft(_q,_c,MaxBW)].y + SCORE_GAP_EXT, rt[calLeft(_q,_c,MaxBW)].m + SCORE_GAP, 0);

                if (chq == ILLEGAL_WORD || chc == ILLEGAL_WORD)
                {
                    // illegal word
                    rt[calIndex(_q,_c,MaxBW)].m = 0;
                }
                else
                {
                    rt[calIndex(_q,_c,MaxBW)].m = max2(max3(rt[calDiag(_q,_c,MaxBW)].x, rt[calDiag(_q,_c,MaxBW)].y, rt[calDiag(_q,_c,MaxBW)].m) + BLOSUM62[chq * 26 + chc], 0);
                }

                score = max3(rt[calIndex(_q,_c,MaxBW)].x, rt[calIndex(_q,_c,MaxBW)].y, rt[calIndex(_q,_c,MaxBW)].m);
                
                // printf("(q = %c,c = %c) BLOSUM62 = %d rt[_q * width + _c].s = %d\n", chq+65,chc+65,BLOSUM62[chq * 26 + chc], rt[_q * width + _c].s);
                // (rd + idx*direct_matrixSize)[_c * height + _q + q_offset] = (score == rt[_q * width + _c].x)*TOP + (score == rt[_q * width + _c].y)*LEFT + (rt[_c * height + _q + q_offset].m)*DIAG; 
                if(score)
                    rd[calIndex(_c, _q+q_offset,height) * BatchSize + idx] = \
                        (score == rt[calIndex(_q,_c,MaxBW)].m) ? DIAG : \
                        ((score == rt[calIndex(_q,_c,MaxBW)].y) ? LEFT :TOP );
                
                if (Score < score)
                {
                    Score = score;
                    max_c = _c;
                    max_q = _q + q_offset;
                }
                // printf("(q = %c,c = %c) score = %d maxScore = %d direction = %d\n", chq+65,chc+65,r[_q*width + _c].s,r[max_c * height + max_q].s,r[_q * width + _c].d);
            }
        }
        memcpy(rt,rt + (t_height - 1) * MaxBW ,MaxBW * sizeof(record));
        // Hit when target is not long enough, there are some cells should be zero
        memset(rt + MaxBW, 0, (t_height - 1) * MaxBW * sizeof(record));

    }

    score_d[idx] = Score;
    // res[idx].score = Score;
    assert(Score != 0);

    size_t cur_q= max_q;
    size_t cur_c = max_c;

    q_end_d[idx] = cur_q + q_idx;
    s_end_d[idx] = c_begin + cur_c + cur_q;

    int cnt_q = 0, cnt_c = 0;
    int cigar_len = 0;
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
        (cigar_cnt_d + idx * MaxAlignLen)[cigar_len] = cur_cigar_cnt;
        (cigar_op_d + idx * MaxAlignLen)[cigar_len++] = ((d==DIAG)?'M':((d==TOP)?'D':'I'));
    }

    // free(rt);
    assert(cigar_len > 0);
    cigar_len_d[idx] = cigar_len;
}

void cigar_to_index(int cigar_len, char* cigar_op, int* cigar_cnt,
               size_t q_start,
               size_t c_start,
               vector<size_t>& q_res, vector<size_t>& s_res)
{
    size_t cur_q = q_start;
    size_t cur_c = c_start;
    for(int i = 0; i < cigar_len; i ++){
        int cur = cigar_cnt[i];
        char op = cigar_op[i];
        int d = ((op=='M')?DIAG:((op=='D')?TOP:LEFT));
        

        for(int j = 0; j < cur; ++ j){

            int tmp_q = (d&0x01) ? (cur_q) : -1;
            int tmp_c = (d&0x02) ? (cur_c) : -1;
            q_res.push_back(tmp_q);
            s_res.push_back(tmp_c);

            //TOP 01b, left 10b, diag 11b
            //DIAG : cur_q -= 1, cur_c -= 1
            //TOP : cur_q -= 1, 
            //LEFT : cur_c -= 1
            cur_q -= (d == DIAG || d == TOP);
            cur_c -= (d == LEFT || d == DIAG); // Decrement cur_c if LEFT (10b)
        }
    
    }
    reverse(q_res.begin(),q_res.end());
    reverse(s_res.begin(),s_res.end());
    assert(q_res.size() && s_res.size());
}

#ifdef USE_GPU_SW
void handle_results(cudaEvent_t &stream, Task *task_host, uint32_t *num_task, QueryGroup &q_group, size_t s_length, int stream_id, vector<SWResult> &res, SWTasks &sw_task)
{
    cudaEventSynchronize(stream);
    mu2.lock();
    cout << "=";
    res.clear();
    size_t n_task_pre = sw_task.num_task;
    size_t n_task = sw_task.num_task + *num_task;
    sw_task.c_len += s_length;
    sw_task.q_idxs.resize(n_task);
    sw_task.q_lens.resize(n_task);
    sw_task.q_len4_offs.resize(n_task+1);
    sw_task.s_len4_offs.resize(n_task+1);
    sw_task.diags.resize(n_task);
    sw_task.info.resize(n_task);
    Task *t_begin = task_host;
    sw_task.num_task = n_task;
    res.resize(*num_task);
#pragma omp parallel for
    for (int i = 0; i < *num_task; i++)
    {
        Task &kv = *(t_begin + i);
        sw_task.q_idxs[i + n_task_pre]=q_group.offset[kv.q_id];
        sw_task.q_lens[i + n_task_pre]=q_group.length[kv.q_id];
        sw_task.diags[i + n_task_pre] =  sw_task.c_offset + kv.key;
        sw_task.info[i+ n_task_pre].group_id = q_group.group_id;
        sw_task.info[i+ n_task_pre].stream_id = stream_id;
        sw_task.info[i+ n_task_pre].idx = i;
        res[i].num_q = kv.q_id;
    }

    for (int i = 0; i < *num_task; i++)
    {
        int q_len4 = sw_task.q_lens[i+ n_task_pre];
        q_len4 = q_len4 % 4? q_len4 + (4 - (q_len4 % 4)) : q_len4;
        int s_len4 = sw_task.q_lens[i+ n_task_pre] + (band_width << 1);
        s_len4 = s_len4 %4? s_len4 + (4-(s_len4%4)):s_len4;
        sw_task.q_len4_offs[i+ n_task_pre+1] = sw_task.q_len4_offs[i+ n_task_pre] + q_len4;
        sw_task.s_len4_offs[i+ n_task_pre+1] = sw_task.s_len4_offs[i+ n_task_pre] + s_len4;
    }

    sw_task.c_offset += s_length;
    mu2.unlock();

}
#else
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
#endif

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
    KeyValue *pHashTable[NUM_STREAM];
    Task *task_dev[NUM_STREAM];
    uint32_t *task_num_dev[NUM_STREAM];

    for (int s = 0; s < NUM_STREAM; s++)
    {
        // pHashTable[s] = create_hashtable(max_hashtable_capacity);
        CUDA_CALL(cudaMalloc((void **)&task_dev[s], MAX_FILTER_TASK * sizeof(Task)));
        CUDA_CALL(cudaMemset(task_dev[s], 0, MAX_FILTER_TASK * sizeof(Task)));
        CUDA_CALL(cudaMalloc((void **)&task_num_dev[s], sizeof(uint32_t)));
        CUDA_CALL(cudaMemset(task_num_dev[s], 0, sizeof(uint32_t)));
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

    size_t free_byte, total_byte;
    CUDA_CALL(cudaMemGetInfo(&free_byte, &total_byte));
    cout << "GPU mem: " << (double)(total_byte - free_byte) / (1073741824) << " GB / " << (double)total_byte / (1073741824) << " GB" << endl;

#ifndef USE_GPU_SW
    SWTasks sw_tasks[q_groups.size()][NUM_STREAM];
#endif
    SWTasks sw_tasks_total;
    vector<SWResult> res_s[q_groups.size()][NUM_STREAM];

    char* query_dev;
    CUDA_CALL(cudaMalloc((void **)&query_dev, total_len_query));
    CUDA_CALL(cudaMemcpy(query_dev, query, total_len_query, cudaMemcpyHostToDevice));
#ifdef USE_GPU_SW
    sw_tasks_total.q = query;
#endif

    gettimeofday(&t_end, NULL);
    time_prof.mem_time += timeuse(t_start, t_end);

    int g_begin = 0;
    while (g_begin < q_groups.size())
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

        cudaStream_t streams[NUM_STREAM];
        thread result_threads[n_groups][NUM_STREAM];

        cudaEvent_t seeding_finished[n_groups][NUM_STREAM];


#ifdef USE_GUP_DIFFUSE
        cudaEvent_t sw_kernel_finished[NUM_STREAM][200];

        int direct_matrixSize = (MaxQueryLen+1) * MaxBW;
        int threadsPerBlock = 256;  // 根据 shared memory 限制调整
        int blocks = (BatchSize + threadsPerBlock - 1) / threadsPerBlock;
        
        int* rd[NUM_STREAM];   // direct_matrixSize * BatchSize * sizeof(int)
        record* rt[NUM_STREAM];
        static int* BLOSUM62_d[NUM_STREAM];
        int* score_d[NUM_STREAM][200], *score_h[NUM_STREAM][200];
        
        size_t* q_end_d[NUM_STREAM][200], *q_end_h[NUM_STREAM][200];
        size_t* s_end_d[NUM_STREAM][200], *s_end_h[NUM_STREAM][200];
        char* cigar_op_d[NUM_STREAM][200], *cigar_op_h[NUM_STREAM][200];
        int *cigar_cnt_d[NUM_STREAM][200], *cigar_cnt_h[NUM_STREAM][200];
        int* cigar_len_d[NUM_STREAM][200], *cigar_len_h[NUM_STREAM][200];


        for (int s = 0; s < NUM_STREAM; s++)
        {
            for(int i = 0; i < 200; ++ i){
                score_h[s][i] = (int*) malloc(sizeof(int) * BatchSize);

                q_end_h[s][i] = (size_t*) malloc(sizeof(size_t) * BatchSize);
                s_end_h[s][i] = (size_t*) malloc(sizeof(size_t) * BatchSize);
                cigar_cnt_h[s][i] = (int*) malloc(sizeof(int) * MaxAlignLen * BatchSize);
                cigar_op_h[s][i] = (char*) malloc(sizeof(char) * MaxAlignLen * BatchSize);
                cigar_len_h[s][i] = (int*) malloc(sizeof(int) * BatchSize);
            }
        }
        
        // SWResult_d * res_d[NUM_STREAM], *res_h[NUM_STREAM];  // MaxAlignLen * BatchSize * sizeof(SWResult_d)
        // static int* BLOSUM62_d[NUM_STREAM];
        // for (int s = 0; s < NUM_STREAM; s++)
        // {
        //     res_h[s] = (SWResult_d*) malloc(BatchSize * sizeof(SWResult_d));
        // }
    
#endif

#ifndef USE_GPU_SW
        vector<future<int>> rs[n_groups][NUM_STREAM];
#endif
        size_t s_begin = 0;

        gettimeofday(&t_end, NULL);
        group_time += timeuse(t_start, t_end);
        time_prof.mem_time += timeuse(t_start, t_end);
        // cout << "Prepare mem and data Time: " << timeuse(t_start, t_end) << endl;

        gettimeofday(&t_start, NULL);

        for (int s = 0; s < NUM_STREAM; s++)
        {
            CUDA_CALL(cudaStreamCreate(&streams[s]));
#ifdef USE_GUP_DIFFUSE
            CUDA_CALL(cudaMallocAsync((void**)&rd[s], direct_matrixSize * BatchSize * sizeof(int),streams[s]));
            CUDA_CALL(cudaMallocAsync((void**)&rt[s], MaxBW * (TILE_SIZE + 1) * BatchSize * sizeof(record),streams[s]));
            
            // CUDA_CALL(cudaMallocAsync((void**)&res_d[s],BatchSize * sizeof(SWResult_d),streams[s]));
            for(int i = 0; i < 200; ++ i){

                CUDA_CALL(cudaMallocAsync((void**)&score_d[s][i],BatchSize * sizeof(int),streams[s]));
                
                CUDA_CALL(cudaMallocAsync((void**)&q_end_d[s][i],BatchSize * sizeof(size_t),streams[s]));
                CUDA_CALL(cudaMallocAsync((void**)&s_end_d[s][i],BatchSize * sizeof(size_t),streams[s]));
                CUDA_CALL(cudaMallocAsync((void**)&cigar_op_d[s][i],BatchSize * sizeof(char) * MaxAlignLen,streams[s]));
                CUDA_CALL(cudaMallocAsync((void**)&cigar_cnt_d[s][i],BatchSize * sizeof(int) * MaxAlignLen,streams[s]));
                CUDA_CALL(cudaMallocAsync((void**)&cigar_len_d[s][i],BatchSize * sizeof(int),streams[s]));
            }
            
            CUDA_CALL(cudaMallocAsync((void**)&BLOSUM62_d[s], 26 * 26 * sizeof(int) * BatchSize,streams[s]));
            for(int i = 0; i < BatchSize; ++ i)
                CUDA_CALL(cudaMemcpyAsync(BLOSUM62_d[s] + i * 26 * 26, BLOSUM62, 26 * 26 * sizeof(int),cudaMemcpyHostToDevice,streams[s]));
#endif
            pHashTable[s] = create_hashtable_async(max_hashtable_capacity,streams[s]);
            // printf("start stream %d\n", s);
            size_t s_length_stream = s_length[s];
            size_t s_length_stream_byte = s_length_stream / 8 * 5;
            size_t s_length_stream_block = s_length_stream / 32 * 5;
            size_t each_length_block = (s_length_stream_block - 1) / (mingridsize_seeding * threadblocksize_seeding) + 1;

            if (g_begin == 0)
            {
                CUDA_CALL(cudaMemcpyAsync(subj_dev + s_begin, subj[s], s_length_stream_byte, cudaMemcpyHostToDevice, streams[s]));
            }
            if (STREAM_SYNC && s > 0)
            {
                CUDA_CALL(cudaStreamSynchronize(streams[s - 1]));
            }
            for (int g = g_begin; g < g_begin + n_groups; g++)
            {
                int g_idx = g - g_begin;
                CUDA_CALL(cudaEventCreate(&seeding_finished[g_idx][s]));
                int n_query = q_groups[g].n_query;
                if (g > 0)
                {
                    CUDA_CALL(cudaMemsetAsync(task_dev[s], 0, MAX_FILTER_TASK * sizeof(Task), streams[s]));
                    CUDA_CALL(cudaMemsetAsync(task_num_dev[s], 0, sizeof(uint32_t), streams[s]));
                    CUDA_CALL(cudaMemsetAsync(pHashTable[s], 0xff, max_hashtable_capacity * sizeof(KeyValue), streams[s]));
                }
                seeding_kernel<<<mingridsize_seeding, threadblocksize_seeding, 0, streams[s]>>>(pHashTable[s], (uint32_t *)(subj_dev + s_begin), each_length_block, s_length_stream, q_lengths_dev[g_idx], q_num_dev[g_idx], q_idx_dev[g_idx], n_query, index_size_dev[g_idx], g_idx);
                filter_kernel<<<n_query, threadblocksize_filter, 0, streams[s]>>>(pHashTable[s], task_dev[s], task_num_dev[s], threshold_dev[g_idx], g_idx);
                // CUDA_CALL(cudaMemcpyAsync(hashtable_host[g_idx][s], pHashTable[s], max_hashtable_capacity * sizeof(KeyValue), cudaMemcpyDeviceToHost, streams[s]));
                CUDA_CALL(cudaMemcpyAsync(task_host[g_idx][s], task_dev[s], MAX_FILTER_TASK * sizeof(Task), cudaMemcpyDeviceToHost, streams[s]));
                CUDA_CALL(cudaMemcpyAsync(task_num_host[g_idx][s], task_num_dev[s], sizeof(uint32_t), cudaMemcpyDeviceToHost, streams[s]));
                CUDA_CALL(cudaEventRecord(seeding_finished[g_idx][s]));
// #ifdef USE_GPU_SW
//                 result_threads[g_idx][s] = thread(handle_results, ref(seeding_finished[g_idx][s]), task_host[g_idx][s], task_num_host[g_idx][s], ref(q_groups[g]), s_length[s], s, ref(res_s[g][s]), ref(sw_tasks_total));
#ifdef USE_GUP_DIFFUSE

                cudaEventSynchronize(seeding_finished[g_idx][s]);
                size_t cpu_start = 0, n = *task_num_host[g_idx][s];
                // printf("@@ g=%d, s=%d %d\n",g,s,n);
                Task *t_begin = task_host[g_idx][s];
                
                res_s[g][s].resize(n);
                
                for(int it = 0; it < n; ++it) res_s[g][s][it].num_q = task_host[g_idx][s][it].q_id;
                
                // printf("cuda kernel begin\n");
                for(size_t it = 0; it < n; it += BatchSize){
                    if(it+BatchSize >= n){
                        cpu_start = it;
                        break;
                    }
                    CUDA_CALL(cudaMemsetAsync(rd[s], 0, direct_matrixSize * BatchSize * sizeof(int), streams[s]));
                    CUDA_CALL(cudaMemsetAsync(rt[s], 0, MaxBW * (TILE_SIZE + 1) * BatchSize * sizeof(record), streams[s]));

                    CUDA_CALL(cudaEventCreate(&sw_kernel_finished[s][it/BatchSize]));    
                    banded_sw_kernel<<<blocks,threadsPerBlock,0,streams[s]>>>(
                                    q_lengths_dev[g_idx], q_offset_dev[g_idx],task_dev[s]+it,
                                    query_dev,subj_dev + s_begin,s_length[s],
                                    rd[s],rt[s],band_width,
                                    score_d[s][it/BatchSize],
                                    q_end_d[s][it/BatchSize],s_end_d[s][it/BatchSize],
                                    cigar_op_d[s][it/BatchSize],cigar_cnt_d[s][it/BatchSize],cigar_len_d[s][it/BatchSize],
                                    BLOSUM62_d[s]);
                    // CUDA_CALL(cudaMemcpyAsync(res_h[s], res_d[s], BatchSize * sizeof(SWResult_d), cudaMemcpyDeviceToHost,streams[s]));
                    CUDA_CALL(cudaMemcpyAsync(score_h[s][it/BatchSize], score_d[s][it/BatchSize], BatchSize * sizeof(int), cudaMemcpyDeviceToHost,streams[s]));
                    // CUDA_CALL(cudaMemcpyAsync(q_len_h[s][it/BatchSize], q_len_d[s][it/BatchSize], BatchSize * sizeof(int), cudaMemcpyDeviceToHost,streams[s]));
                    // CUDA_CALL(cudaMemcpyAsync(s_len_h[s][it/BatchSize], s_len_d[s][it/BatchSize], BatchSize * sizeof(int), cudaMemcpyDeviceToHost,streams[s]));
                    // CUDA_CALL(cudaMemcpyAsync(q_res_h[s][it/BatchSize], q_res_d[s][it/BatchSize], BatchSize * sizeof(size_t) * MaxAlignLen, cudaMemcpyDeviceToHost,streams[s]));
                    // CUDA_CALL(cudaMemcpyAsync(s_res_h[s][it/BatchSize], s_res_d[s][it/BatchSize], BatchSize * sizeof(size_t) * MaxAlignLen, cudaMemcpyDeviceToHost,streams[s]));
                    
                    CUDA_CALL(cudaMemcpyAsync(q_end_h[s][it/BatchSize], q_end_d[s][it/BatchSize], BatchSize * sizeof(size_t), cudaMemcpyDeviceToHost,streams[s]));
                    CUDA_CALL(cudaMemcpyAsync(s_end_h[s][it/BatchSize], s_end_d[s][it/BatchSize], BatchSize * sizeof(size_t), cudaMemcpyDeviceToHost,streams[s]));
                    
                    CUDA_CALL(cudaMemcpyAsync(cigar_op_h[s][it/BatchSize], cigar_op_d[s][it/BatchSize], BatchSize * sizeof(char) * MaxAlignLen, cudaMemcpyDeviceToHost,streams[s]));
                    CUDA_CALL(cudaMemcpyAsync(cigar_cnt_h[s][it/BatchSize], cigar_cnt_d[s][it/BatchSize], BatchSize * sizeof(int) * MaxAlignLen, cudaMemcpyDeviceToHost,streams[s]));
                    CUDA_CALL(cudaMemcpyAsync(cigar_len_h[s][it/BatchSize], cigar_len_d[s][it/BatchSize], BatchSize * sizeof(int), cudaMemcpyDeviceToHost,streams[s]));

                    // CUDA_CALL(cudaStreamSynchronize(streams[s]));
                    CUDA_CALL(cudaEventRecord(sw_kernel_finished[s][it/BatchSize]));
                    cudaEventSynchronize(sw_kernel_finished[s][it/BatchSize]);
                    for (size_t i = 0; i < BatchSize; ++i) {

                        if(cigar_len_h[s][it/BatchSize][i] >= MaxAlignLen || cigar_len_h[s][it/BatchSize][i] < 0){
                            printf("## cigar_len: %d\n",cigar_len_h[s][it/BatchSize][i]);
                        }
                        // assert(cigar_len_h[i] < MaxAlignLen && cigar_len_h[i] < MaxAlignLen);
                        //TODO From cigar to index
                        cigar_to_index(cigar_len_h[s][it/BatchSize][i],cigar_op_h[s][it/BatchSize] + i * MaxAlignLen,cigar_cnt_h[s][it/BatchSize] + i * MaxAlignLen,
                                        q_end_h[s][it/BatchSize][i],s_end_h[s][it/BatchSize][i],
                                        res_s[g][s][it + i].q_res,res_s[g][s][it + i].s_res);
                        
                        res_s[g][s][it + i].score = score_h[s][it/BatchSize][i];
                        
                        // check
                        // SWResult sw_tmp;
                        // cpu_kernel(&sw_tmp,query,subj[s],s_length[s],\
                        //             q_groups[g].offset[task_host[g_idx][s][it + i].q_id],\
                        //             q_groups[g].length[task_host[g_idx][s][it + i].q_id],\
                        //             task_host[g_idx][s][it+i].key,band_width);
                        // assert(sw_tmp.q_res.size() == res_s[g][s][it + i].q_res.size() );             
                        // assert(sw_tmp.s_res.size() == res_s[g][s][it + i].s_res.size() );
                        // for(int cnt = 0; cnt < sw_tmp.q_res.size(); ++ cnt){
                        //     if(!(sw_tmp.q_res[cnt] == res_s[g][s][it + i].q_res[cnt])){
                        //         printf("## error %d\n", cnt);
                        //     }
                        //     assert(sw_tmp.q_res[cnt] == res_s[g][s][it + i].q_res[cnt]);
                        // }
                        // for(int cnt = 0; cnt < sw_tmp.s_res.size(); ++ cnt){
                        //     if(!(sw_tmp.s_res[cnt] == res_s[g][s][it + i].s_res[cnt])){
                        //         printf("## error %d\n", cnt);
                        //     }
                        //     assert(sw_tmp.s_res[cnt] == res_s[g][s][it + i].s_res[cnt]);
                        // }            
                        // assert(sw_tmp.score == res_s[g][s][it + i].score );             
                    }
                    // printf("done : %ld\n", it);
                }
                // printf("cuda kernel finished \n");
                for(size_t it=cpu_start; it < n; it ++){   
                    cpu_kernel(&res_s[g][s][it],query,subj[s],s_length[s],q_groups[g].offset[task_host[g_idx][s][it].q_id],q_groups[g].length[task_host[g_idx][s][it].q_id],task_host[g_idx][s][it].key,band_width);
                    generate_report(&res_s[g][s][it],query, subj[s]);
                }
                // printf("kernel finished \n");
                for(size_t it = 0; it < cpu_start; it ++){
                    generate_report(&res_s[g][s][it],query, subj[s]);
                    // printf("generate done : %ld\n", it);
                }

                    // banded_sw_kernel<<<blocks,threadsPerBlock,0,s>>>(q_lengths_dev[g_idx], q_idx_dev[g_idx],diags_d,query_dev,subj_dev,tasks.c_len,rd,band_width,res_d,BLOSUM62_d);
        
#else
                result_threads[g_idx][s] = thread(handle_results, ref(seeding_finished[g_idx][s]), query, subj[s], task_host[g_idx][s], task_num_host[g_idx][s], ref(q_groups[g]), s_length[s], s, ref(res_s[g][s]), ref(sw_tasks[g][s]), pool, ref(rs[g_idx][s]));
#endif 
            }
            // CUDA_CALL(cudaStreamSynchronize(streams[s]));
            destroy_hashtable_async(pHashTable[s],streams[s]);
            s_begin += s_length_stream_byte;
            cout << "=";
        }

        CUDA_CALL(cudaDeviceSynchronize());

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

        for (int s = 0; s < NUM_STREAM; s++)
        {
            #ifdef USE_GUP_DIFFUSE
                CUDA_CALL(cudaFreeAsync(rd[s], streams[s]));
                CUDA_CALL(cudaFreeAsync(rt[s], streams[s]));
                CUDA_CALL(cudaFreeAsync(BLOSUM62_d[s], streams[s]));
                CUDA_CALL(cudaStreamSynchronize(streams[s]));
                
                for(int i = 0; i < 200; ++ i){
                    
                    CUDA_CALL(cudaFreeAsync(score_d[s][i], streams[s]));
                    CUDA_CALL(cudaFreeAsync(s_end_d[s][i], streams[s]));
                    CUDA_CALL(cudaFreeAsync(q_end_d[s][i], streams[s]));
                    CUDA_CALL(cudaFreeAsync(cigar_op_d[s][i], streams[s]));
                    CUDA_CALL(cudaFreeAsync(cigar_cnt_d[s][i], streams[s]));
                    CUDA_CALL(cudaFreeAsync(cigar_len_d[s][i], streams[s]));

                    
                    free(score_h[s][i]);
                    
                    free(s_end_h[s][i]);
                    free(q_end_h[s][i]);
                    free(cigar_op_h[s][i]);
                    free(cigar_cnt_h[s][i]);
                    free(cigar_len_h[s][i]);
                }
            #endif
            CUDA_CALL(cudaStreamDestroy(streams[s]));
        }

        int hsp_count = 0;

        for (int s = 0; s < NUM_STREAM; s++)
        {
            for (int g = g_begin; g < g_begin + n_groups; g++)
            {
                int g_idx = g - g_begin;
#ifndef USE_GUP_DIFFUSE
                result_threads[g_idx][s].join();
#endif
                CUDA_CALL(cudaEventDestroy(seeding_finished[g_idx][s]));
                hsp_count += res_s[g][s].size();
                cout << "=";
#ifndef USE_GPU_SW
                for (auto &r : rs[g_idx][s])
                    r.get();
                proceed_result(res, res_s[g][s], query, subj[s], q_groups[g], s_name[s], s_offsets[s], sn_offsets[s], s_num[s], total_db_size);
                cout << "=";
#endif
            }
        }

        g_begin += MAX_GROUPS_PER_ROUND;

        gettimeofday(&t_end, NULL);
        time_prof.cpu_time += timeuse(tt_start, t_end);
        group_time += timeuse(tt_start, t_end);
        cout << "] " << group_time << "s, " << hsp_count << " HSPs" << endl;
    }

    gettimeofday(&t_start, NULL);

    n_groups = q_groups.size();

    if (n_groups > MAX_GROUPS_PER_ROUND)
        n_groups = MAX_GROUPS_PER_ROUND;

    for (int s = 0; s < NUM_STREAM; s++)
    {
        // destroy_hashtable(pHashTable[s]);
        CUDA_CALL(cudaFree(task_dev[s]));
        CUDA_CALL(cudaFree(task_num_dev[s]));
    }

    for (int g = 0; g < n_groups; g++)
    {
        CUDA_CALL(cudaFree(q_num_dev[g]));

        CUDA_CALL(cudaFree(q_idx_dev[g]));

        CUDA_CALL(cudaFree(q_lengths_dev[g]));
        CUDA_CALL(cudaFree(q_offset_dev[g]));

        CUDA_CALL(cudaFree(index_size_dev[g]));

        CUDA_CALL(cudaFree(threshold_dev[g]));
    }

    gettimeofday(&t_end, NULL);
    time_prof.mem_time += timeuse(t_start, t_end);

#ifdef USE_GPU_SW

    gettimeofday(&t_start, NULL);
    // char* query_dev;
    // CUDA_CALL(cudaMalloc((void **)&query_dev, total_len_query));
    // CUDA_CALL(cudaMemcpy(query_dev, query, total_len_query, cudaMemcpyHostToDevice));
    // sw_tasks_total.q = query;
    for (int s = 0; s < NUM_STREAM; s++)
    {
        sw_tasks_total.c_all[s] = subj[s];
        sw_tasks_total.c_offs[s] = s==0? 0: sw_tasks_total.c_offs[s-1] +s_length[s-1];
    }
    kernel_run(ref(sw_tasks_total), query_dev, subj_dev, res_s, band_width);
    // gasal_run(sw_tasks_total, res_s, query_dev, subj_dev, q_groups.size(), band_width);
    cout << "Done.\t[";

    gettimeofday(&t_end, NULL);
    time_prof.gpu_time += timeuse(t_start, t_end);
    gettimeofday(&t_start, NULL);

    CUDA_CALL(cudaFree(query_dev));
    for (int s = 0; s < NUM_STREAM; s++)
    {
        for (int g = 0; g < q_groups.size(); g++)
        {
            proceed_result(res, res_s[g][s], query, subj[s], q_groups[g], s_name[s], s_offsets[s], sn_offsets[s], s_num[s], total_db_size);
        }
        cout << "=";
    }
    cout << "] ";
    gettimeofday(&t_end, NULL);
    cout << timeuse(t_start, t_end) <<"s" << endl;
    time_prof.cpu_time += timeuse(t_start, t_end);
#endif

    gettimeofday(&t_start, NULL);

    CUDA_CALL(cudaFree(subj_dev));

    for (int s = 0; s < NUM_STREAM; s++)
    {
        free(s_name[s]);
        free(sn_offsets[s]);
        free(s_offsets[s]);
    }

    gettimeofday(&t_end, NULL);
    time_prof.mem_time += timeuse(t_start, t_end);
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

    gettimeofday(&t_end, NULL);

    cout << "Output Time:\t" << timeuse(t_start, t_end) << endl;

    cout << "Finished." << endl;
}