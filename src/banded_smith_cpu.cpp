#include "smith.h"
#include <nvtx3/nvToolsExt.h>
// #include <libunwind.h>
// #include <gperftools/tcmalloc.h>

void cigar_to_index(size_t idx, int begin, int* cigar_len, char* cigar_op, int* cigar_cnt,
               size_t* q_start,
               size_t* c_start,
               std::vector<SWResult>& res_s)
{
    assert(cigar_len[idx] > 0);
    size_t cur_q = q_start[idx];
    size_t cur_c = c_start[idx];
    for(int i = 0; i < cigar_len[idx]; i ++){
        int cur = (cigar_cnt + idx * MaxAlignLen)[i];
        char op = (cigar_op + idx * MaxAlignLen)[i];
        int d = ((op=='M')?DIAG:((op=='D')?TOP:LEFT));
        

        for(int j = 0; j < cur; ++ j){

            int tmp_q = (d&0x01) ? (cur_q) : -1;
            int tmp_c = (d&0x02) ? (cur_c) : -1;
            res_s[begin + idx].q_res.push_back(tmp_q);
            res_s[begin + idx].s_res.push_back(tmp_c);

            //TOP 01b, left 10b, diag 11b
            //DIAG : cur_q -= 1, cur_c -= 1
            //TOP : cur_q -= 1, 
            //LEFT : cur_c -= 1
            cur_q -= (d == DIAG || d == TOP);
            cur_c -= (d == LEFT || d == DIAG); // Decrement cur_c if LEFT (10b)
        }
    
    }
    reverse(res_s[begin + idx].q_res.begin(),res_s[begin + idx].q_res.end());
    reverse(res_s[begin + idx].s_res.begin(),res_s[begin + idx].s_res.end());
    assert(res_s[begin + idx].q_res.size() && res_s[begin + idx].s_res.size());
}

void generate_report(SWResult *res, const char* q, const char* c){
    size_t len = res->s_res.size();
    res->align_length = len;
    res->bitscore = (E_lambda * res->score - log(E_k)) / (0.69314718055995);
    char s[len + 1] = {0};
    char q_seq[len + 1] = {0};
    char s_ori[len + 1] = {0};
    char match[len + 1] = {0};
    int s_ori_len = 0;
    res->gap_open = 0;
    res->gaps = 0;
    bool ga = false;
    for (int i = 0; i < len; i++)
    {
        // cout<<c_res[t][i]<<" ";
        if (res->s_res[i] != (size_t)(-1))
        {
            ga = false;
            // printf("res->s_res[i] = %d\n",res->s_res[i]);
            s[i] = get_char(c, res->s_res[i]) + 65;
            if (s[i] == 95)
                s[i] = '*';
            s_ori[s_ori_len++] = s[i];
        }
        else
        {
            s[i] = '-';
            res->gaps++;
            if (!ga)
            {
                ga = true;
                res->gap_open++;
            }
        }
    }

    if (has_must_include)
    {
        string s_ori_str(s_ori, len);
        if (!check_include(s_ori_str))
        {
            res->report = false;
            return;
        }
    }
    ga = false;
    for (int i = 0; i < len; i++)
    {
        // cout<<q_res[t][i]<<" ";
        if (res->q_res[i] != (size_t)(-1))
        {
            ga = false;
            q_seq[i] = q[res->q_res[i]] + 65;
        }
        else
        {
            q_seq[i] = '-';
            res->gaps++;
            if (!ga)
            {
                ga = true;
                res->gap_open++;
            }
        }
    }
    res->mismatch = 0;
    res->positive = 0;
    for (int i = 0; i < len; i++)
    {
        match[i]=' ';
        if (BLOSUM62[(q_seq[i] - 65) * 26 + (s[i] - 65)] > 0 && q_seq[i]!='-' && s[i]!='-')
        {
            res->positive++;
            match[i]='+';
        }
        if (q_seq[i] != s[i])
        {
            res->mismatch++;
        }
        else
        {
            match[i]=q_seq[i];
        }
            
    }
    res->n_identity = res->align_length - res->mismatch;
    res->p_identity = (1 - (double)res->mismatch / res->align_length) * 100;
    if (detailed_alignment)
    {
        res->q = q_seq;
        res->s = s;
        res->s_ori = s_ori;
        res->match = match;
    }
}


void generate_report(size_t idx, int begin, std::vector<SWResult>& res_s, int* score, Task* task, const char* q, const char* c){
    res_s[begin + idx].score = score[idx];
    res_s[begin + idx].num_q = task[idx].q_id;
    size_t len = res_s[begin + idx].s_res.size();
    res_s[begin + idx].align_length = len;
    res_s[begin + idx].bitscore = (E_lambda * res_s[begin + idx].score - log(E_k)) / (0.69314718055995);
    char s[len + 1] = {0};
    char q_seq[len + 1] = {0};
    char s_ori[len + 1] = {0};
    char match[len + 1] = {0};
    int s_ori_len = 0;
    res_s[begin + idx].gap_open = 0;
    res_s[begin + idx].gaps = 0;
    bool ga = false;
    for (int i = 0; i < len; i++)
    {
        // cout<<c_res[t][i]<<" ";
        if (res_s[begin + idx].s_res[i] != (size_t)(-1))
        {
            ga = false;
            // printf("res_s[begin + idx].s_res[i] = %d\n",res_s[begin + idx].s_res[i]);
            s[i] = get_char(c, res_s[begin + idx].s_res[i]) + 65;
            if (s[i] == 95)
                s[i] = '*';
            s_ori[s_ori_len++] = s[i];
        }
        else
        {
            s[i] = '-';
            res_s[begin + idx].gaps++;
            if (!ga)
            {
                ga = true;
                res_s[begin + idx].gap_open++;
            }
        }
    }

    if (has_must_include)
    {
        string s_ori_str(s_ori, len);
        if (!check_include(s_ori_str))
        {
            res_s[begin + idx].report = false;
            return;
        }
    }
    ga = false;
    for (int i = 0; i < len; i++)
    {
        // cout<<q_res[t][i]<<" ";
        if (res_s[begin + idx].q_res[i] != (size_t)(-1))
        {
            ga = false;
            q_seq[i] = q[res_s[begin + idx].q_res[i]] + 65;
        }
        else
        {
            q_seq[i] = '-';
            res_s[begin + idx].gaps++;
            if (!ga)
            {
                ga = true;
                res_s[begin + idx].gap_open++;
            }
        }
    }
    res_s[begin + idx].mismatch = 0;
    res_s[begin + idx].positive = 0;
    for (int i = 0; i < len; i++)
    {
        match[i]=' ';
        if (BLOSUM62[(q_seq[i] - 65) * 26 + (s[i] - 65)] > 0 && q_seq[i]!='-' && s[i]!='-')
        {
            res_s[begin + idx].positive++;
            match[i]='+';
        }
        if (q_seq[i] != s[i])
        {
            res_s[begin + idx].mismatch++;
        }
        else
        {
            match[i]=q_seq[i];
        }
            
    }
    res_s[begin + idx].n_identity = res_s[begin + idx].align_length - res_s[begin + idx].mismatch;
    res_s[begin + idx].p_identity = (1 - (double)res_s[begin + idx].mismatch / res_s[begin + idx].align_length) * 100;
    if (detailed_alignment)
    {
        res_s[begin + idx].q = q_seq;
        res_s[begin + idx].s = s;
        res_s[begin + idx].s_ori = s_ori;
        res_s[begin + idx].match = match;
    }
}
// CPU做一个分Batch的
void cpu_kernel(SWResult *res, 
                const char *q, const char* c, 
                size_t c_len, uint32_t q_idx, uint32_t n,
                // TODO pack 
                uint32_t diag, const int band_width)
{
    //TODO 把banded 放到scoreMatrix的接口里，sw里面不体现
    int64_t c_begin = (int64_t)diag - band_width - n + 2;
    size_t c_end = diag + band_width;
    if (has_must_include)
    {
        if (!check_include(c, c_begin, c_end))
        {
            res->report = false;
            return;
        }
    }

    size_t width = 2 * band_width + 1;
    size_t height = n + 1;

    // short *s = (short *)malloc(width * height * sizeof(short));
    // char *p = (char *)malloc(width * height * sizeof(char));

    int tileSize = 2;
    size_t t_height = tileSize + 1;
    
    record *rt = (record *)malloc(width * t_height * sizeof(record));
    int *rd = (int *)malloc(width * height * sizeof(int));

    if (rd == nullptr || rt == nullptr)
    {
        printf("CPU out of memory!\n");
        exit(-1);
        return;
    }

    // memset(s, 0, width * height * sizeof(short));
    // memset(p, 0, width * height * sizeof(char));
    memset(rt, 0, width * t_height * sizeof(record));
    memset(rd, 0, width * height * sizeof(int));

    size_t max_q = 0;
    size_t max_c = 0;
    int score = 0, Score = 0;
    // TODO index映射到tile里的index
    // TODO 行列的tile?
    // cal maxScore and it's position
    for(size_t it = 0; it * tileSize < n; it ++){
        
        size_t q_offset = it * tileSize;

        for(size_t _q = 0; _q < t_height-1 && q_offset + _q < height-1; ++_q){
            for(size_t _c = 0; _c < width-2; ++_c){
                
                if(c_begin + _c+ q_offset + _q < 0) continue;
                if(c_begin + _c+ q_offset + _q >= c_len) break;

                char chq = q[q_idx + q_offset + _q];
                char chc = get_char(c, c_begin + q_offset + _c + _q);
                
                if (chq == END_SIGNAL || chc == END_SIGNAL)
                {
                    continue;
                }
                //rt(_q,_c) -> (_q+1) * width + _c + 1
                // logical m(_q,_c).x = max(m(_q-1,_c).x + SCORE_GAP_EXT, m(_q-1,_c).m +SCORE_GAP, 0 );
                // logical m(_q,_c).y = max(m(_q,_c-1).y + SCORE_GAP_EXT, m(_q,_c-1).m +SCORE_GAP, 0 );
                // logical m(_q,_c).m = max(m(_q-1,_c-1).y,m(_q-1,_c-1).x,m(_q-1,_c-1).m, 0 );
                
                rt[calIndex(_q,_c,width)].x = max3(rt[calTop(_q,_c,width)].x + SCORE_GAP_EXT,  rt[calTop(_q,_c,width)].m + SCORE_GAP, 0);
                rt[calIndex(_q,_c,width)].y = max3(rt[calLeft(_q,_c,width)].y + SCORE_GAP_EXT, rt[calLeft(_q,_c,width)].m + SCORE_GAP, 0);

                if (chq == ILLEGAL_WORD || chc == ILLEGAL_WORD)
                {
                    // illegal word
                    rt[calIndex(_q,_c,width)].m = 0;
                }
                else
                {
                    rt[calIndex(_q,_c,width)].m = max2(max3(rt[calDiag(_q,_c,width)].x, rt[calDiag(_q,_c,width)].y, rt[calDiag(_q,_c,width)].m) + BLOSUM62[chq * 26 + chc], 0);
                }

                score = max3(rt[calIndex(_q,_c,width)].x, rt[calIndex(_q,_c,width)].y, rt[calIndex(_q,_c,width)].m);
                
                if(score)
                    rd[calIndex(_c, _q+q_offset, height)] = \
                        (score == rt[calIndex(_q,_c,width)].m) ? DIAG : \
                        ((score == rt[calIndex(_q,_c,width)].y) ? LEFT :TOP );
                    
                if (Score < score)
                {
                    Score = score;
                    max_c = _c;
                    max_q = _q + q_offset;
                }
                // printf("(q = %c,c = %c) score = %d maxScore = %d direction = %d\n", chq+65,chc+65,r[_q*width + _c].s,r[max_c * height + max_q].s,r[_q * width + _c].d);
            }
        }
        memcpy(rt,rt + (t_height - 1) * width ,width * sizeof(record));
        // Hit when target is not long enough, there are some cells should be zero
        memset(rt + width, 0, (t_height - 1) *width * sizeof(record));

    }

    res->score = Score;
    assert(res->score != 0);

    size_t cur_q = max_q;
    size_t cur_c = max_c;

    while (rd[calIndex(cur_c,cur_q,height)])
    {
        int d = rd[calIndex(cur_c,cur_q,height)];
        int64_t res_q = (d&0x01) ? (cur_q + q_idx) : -1;
        int64_t res_c = (d&0x02) ? (c_begin + cur_c + cur_q) : -1;
        
        res->q_res.push_back(res_q);
        res->s_res.push_back(res_c);
        //TOP 01b, left 10b, diag 11b
        //DIAG : cur_q -= 1
        //TOP : cur_q -= 1, cur_c += 1;
        //LEFT : cur_c -= 1
        (cur_q) -= (d == DIAG || d == TOP);
        (cur_c) += (d == TOP);
        (cur_c) -= (d == LEFT);
    }

    
    // string Cigar = "";
    // vector<int> cigar_len;
    // cur_q = max_q;
    // cur_c = max_c;
    // while (rd[calIndex(cur_c,cur_q,height)])
    // {
    //     int d = rd[calIndex(cur_c,cur_q,height)];
    //     int cur_cigar_cnt = 0;
    //     while (rd[calIndex(cur_c,cur_q,height)] && rd[calIndex(cur_c,cur_q,height)]==d){
    //         cur_cigar_cnt ++;
    //         //TOP 01b, left 10b, diag 11b
    //         //DIAG : cur_q -= 1
    //         //TOP : cur_q -= 1, cur_c += 1;
    //         //LEFT : cur_c -= 1
    //         cur_q -= (d == DIAG || d == TOP);
    //         cur_c += (d == TOP); // Increment cur_c if TOP (01b)
    //         cur_c -= (d == LEFT); // Decrement cur_c if LEFT (10b)
    //     }
    //     cigar_len.push_back(cur_cigar_cnt);
    //     Cigar += ((d==DIAG)?'M':((d==TOP)?'D':'I'));
    // }
    
    // cur_q = max_q + q_idx;
    // cur_c = c_begin + max_c + max_q;
    // vector<int>q_res, s_res;
    // for(int i = 0; i < cigar_len.size(); i ++){
    //     int cur = cigar_len[i];
    //     char op = Cigar[i];
    //     int d = ((op=='M')?DIAG:((op=='D')?TOP:LEFT)); 
    //     for(int j = 0; j < cur; ++ j){
    //         int tmp_q = (d&0x01) ? (cur_q) : -1;
    //         int tmp_c = (d&0x02) ? (cur_c) : -1;
        
    //         q_res.push_back(tmp_q);
    //         s_res.push_back(tmp_c);

    //         //TOP 01b, left 10b, diag 11b
    //         //DIAG : cur_q -= 1, cur_c -= 1
    //         //TOP : cur_q -= 1, 
    //         //LEFT : cur_c -= 1
    //         cur_q -= (d == DIAG || d == TOP);
    //         cur_c -= (d == LEFT || d == DIAG); // Decrement cur_c if LEFT (10b)
    //     }
    
    // }

    free(rd);
    free(rt);
    // assert(q_res.size() == res->q_res.size());
    // assert(s_res.size() == res->s_res.size());
    
    // for(int i = 0; i < q_res.size(); ++ i){
    //     if(!(q_res[i] == res->q_res[i])){
    //         printf("## error %d\n", i);
    //     }
    // }
    // for(int i = 0; i < s_res.size(); ++ i){
    //     if(!(s_res[i] == res->s_res[i])){
    //         printf("## error %d\n", i);
    //     }
    // }
    // assert(s_res[i] == res->s_res[i]);
    // assert(q_res[i] == res->q_res[i]);

    reverse(res->q_res.begin(), res->q_res.end());
    reverse(res->s_res.begin(), res->s_res.end());
    generate_report(res,q,c);
}

void cpu_kernel(SWResult *res, 
                const char *q, const char* c, 
                size_t c_len, 
                int64_t c_begin, int64_t c_end,
                uint32_t q_begin, uint32_t q_len,
                const int band_width)
{
    uint32_t diag = (c_begin + c_end) >> 1;
    size_t n = q_len;
    
    if (has_must_include)
    {
        if (!check_include(c, c_begin, c_end))
        {
            res->report = false;
            return;
        }
    }

    size_t width = 2 * band_width + 1;
    size_t height = n + 1;

    // short *s = (short *)malloc(width * height * sizeof(short));
    // char *p = (char *)malloc(width * height * sizeof(char));

    int tileSize = 2;
    size_t t_height = tileSize + 1;
    
    record *rt = (record *)malloc(width * t_height * sizeof(record));
    int *rd = (int *)malloc(width * height * sizeof(int));

    if (rd == nullptr || rt == nullptr)
    {
        printf("CPU out of memory!\n");
        exit(-1);
        return;
    }

    // memset(s, 0, width * height * sizeof(short));
    // memset(p, 0, width * height * sizeof(char));
    memset(rt, 0, width * t_height * sizeof(record));
    memset(rd, 0, width * height * sizeof(int));

    size_t max_q = 0;
    size_t max_c = 0;
    int score = 0, Score = 0;
    // cal maxScore and it's position
    for(size_t it = 0; it * tileSize < n; it ++){
        
        size_t q_offset = it * tileSize;

        for(size_t _q = 0; _q < t_height-1 && q_offset + _q < height-1; ++_q){
            for(size_t _c = 0; _c < width-2; ++_c){
                
                if(c_begin + _c+ q_offset + _q < 0) continue;
                if(c_begin + _c+ q_offset + _q >= c_len) break;

                char chq = q[q_begin + q_offset + _q];
                char chc = get_char(c, c_begin + q_offset + _c + _q);
                
                if (chq == END_SIGNAL || chc == END_SIGNAL)
                {
                    continue;
                }
                //rt(_q,_c) -> (_q+1) * width + _c + 1
                // logical m(_q,_c).x = max(m(_q-1,_c).x + SCORE_GAP_EXT, m(_q-1,_c).m +SCORE_GAP, 0 );
                // logical m(_q,_c).y = max(m(_q,_c-1).y + SCORE_GAP_EXT, m(_q,_c-1).m +SCORE_GAP, 0 );
                // logical m(_q,_c).m = max(m(_q-1,_c-1).y,m(_q-1,_c-1).x,m(_q-1,_c-1).m, 0 );
                
                rt[calIndex(_q,_c,width)].x = max3(rt[calTop(_q,_c,width)].x + SCORE_GAP_EXT,  rt[calTop(_q,_c,width)].m + SCORE_GAP, 0);
                rt[calIndex(_q,_c,width)].y = max3(rt[calLeft(_q,_c,width)].y + SCORE_GAP_EXT, rt[calLeft(_q,_c,width)].m + SCORE_GAP, 0);

                if (chq == ILLEGAL_WORD || chc == ILLEGAL_WORD)
                {
                    // illegal word
                    rt[calIndex(_q,_c,width)].m = 0;
                }
                else
                {
                    rt[calIndex(_q,_c,width)].m = max2(max3(rt[calDiag(_q,_c,width)].x, rt[calDiag(_q,_c,width)].y, rt[calDiag(_q,_c,width)].m) + BLOSUM62[chq * 26 + chc], 0);
                }

                score = max3(rt[calIndex(_q,_c,width)].x, rt[calIndex(_q,_c,width)].y, rt[calIndex(_q,_c,width)].m);
                
                if(score)
                    rd[calIndex(_c, _q+q_offset, height)] = \
                        (score == rt[calIndex(_q,_c,width)].m) ? DIAG : \
                        ((score == rt[calIndex(_q,_c,width)].y) ? LEFT :TOP );
                
                if (Score < score)
                {
                    Score = score;
                    max_c = _c;
                    max_q = _q + q_offset;
                }
                // printf("(q = %c,c = %c) score = %d maxScore = %d direction = %d\n", chq+65,chc+65,r[_q*width + _c].s,r[max_c * height + max_q].s,r[_q * width + _c].d);
            }
        }
        memcpy(rt,rt + (t_height - 1) * width ,width * sizeof(record));
        // Hit when target is not long enough, there are some cells should be zero
        memset(rt + width, 0, (t_height - 1) *width * sizeof(record));

    }

    res->score = Score;
    assert(res->score != 0);

    size_t cur_q = max_q;
    size_t cur_c = max_c;
    
    while (rd[calIndex(cur_c,cur_q,height)])
    {
        int d = rd[calIndex(cur_c,cur_q,height)];
        int64_t res_q = (d&0x01) ? (cur_q + q_begin) : -1;
        int64_t res_c = (d&0x02) ? (c_begin + cur_c + cur_q) : -1;
        
        res->q_res.push_back(res_q);
        res->s_res.push_back(res_c);
        //TOP 01b, left 10b, diag 11b
        //DIAG : cur_q -= 1
        //TOP : cur_q -= 1, cur_c += 1;
        //LEFT : cur_c -= 1
        (cur_q) -= (d == DIAG || d == TOP);
        (cur_c) += (d == TOP);
        (cur_c) -= (d == LEFT);
    }

    free(rd);
    free(rt);

    reverse(res->q_res.begin(), res->q_res.end());
    reverse(res->s_res.begin(), res->s_res.end());
    generate_report(res,q,c);
}

void smith_waterman_kernel(const int idx, SWResult *res, SWTasks *sw_task)
{
    const char *q = sw_task->q;
    const char *c = sw_task->c;
    size_t c_len = sw_task->c_len;
    size_t q_idx = sw_task->q_idxs[idx];
    size_t n = sw_task->q_lens[idx];
    size_t diag = sw_task->diags[idx]; //  pos of c at end of q
    int64_t c_begin = (int64_t)diag - band_width - n + 2;
    size_t c_end = diag + band_width;
    cpu_kernel(res,q,c,c_len,q_idx,n,diag,band_width);
    // cpu_kernel(res,q,c,c_len,c_begin,c_end,q_idx,n,band_width);

}

// void banded_smith_waterman(const char *q, const char *c, vector<uint32_t> &q_idxs, vector<uint32_t> &q_lens, vector<size_t> &diags, size_t c_len, size_t num_task, vector<SWResult> &res, ThreadPool *pool, vector<future<int>> &rs)
// {
// mu1.lock();

// vector<future<int>> rs;
// std::thread threads[num_task];

// mu1.lock();

// for (int i = 0; i < num_task; ++i)
// {
//     rs.emplace_back((*pool).enqueue([=, &res]
//                                  {
//         banded_smith_waterman_kernel(i,q,c,q_idxs[i],q_lens[i],diags[i],c_len,res);
//         return i; }));
//     // threads[i] = std::thread(banded_smith_waterman_kernel, i, q, c, q_idxs, q_lens, diags, c_len, ref(res));
//     // boost::asio::post(*pool, [=,&res](){
//     //     banded_smith_waterman_kernel(i, q, c, q_idxs, q_lens, diags, c_len, res);
//     // });
// }

// mu1.unlock();

// for (auto &r : rs)
// {
//     r.get();
// }

// for (auto &thread : threads)
// {
//     thread.join();
// }

// mu1.unlock();
// }
void cigar_to_index_and_report(size_t idx, int begin, int* cigar_len, char* cigar_op, int* cigar_cnt,
               size_t* q_start, size_t* c_start,
               std::vector<SWResult>& res_s,
               int* score, Task* task, const char* query, const char* target)
{
    SWResult& result = res_s[begin + idx];
    result.score = score[idx];
    result.num_q = task[idx].q_id;
    result.align_length = 0;
    result.gaps = 0;
    result.gap_open = 0;
    result.mismatch = 0;
    result.positive = 0;

    // std::vector<size_t> q_res, s_res;
    size_t cur_q = q_start[idx];
    size_t cur_c = c_start[idx];
    
    // Process CIGAR string
    for (int i = 0; i < cigar_len[idx]; ++i) {
        int count = cigar_cnt[idx * MaxAlignLen + i];
        char op = cigar_op[idx * MaxAlignLen + i];
        int d = ((op=='M')?DIAG:((op=='D')?TOP:LEFT));

        result.align_length += count;
        
        for(int j = 0; j < count; ++ j){

            int tmp_q = (d&0x01) ? (cur_q) : -1;
            int tmp_c = (d&0x02) ? (cur_c) : -1;
            res_s[begin + idx].q_res.push_back(tmp_q);
            res_s[begin + idx].s_res.push_back(tmp_c);

            //TOP 01b, left 10b, diag 11b
            //DIAG : cur_q -= 1, cur_c -= 1
            //TOP : cur_q -= 1, 
            //LEFT : cur_c -= 1
            cur_q -= (d == DIAG || d == TOP);
            cur_c -= (d == LEFT || d == DIAG); // Decrement cur_c if LEFT (10b)
        }
        
        if (op != 'M') {
            result.gap_open += count;
        }
    }
    
    // Reverse vectors as in original implementation
    reverse(res_s[begin + idx].q_res.begin(),res_s[begin + idx].q_res.end());
    reverse(res_s[begin + idx].s_res.begin(),res_s[begin + idx].s_res.end());
    
    
    // Generate alignment strings and calculate statistics
    std::string q_seq, s_seq, s_ori, match;
    for (size_t i = 0; i < result.align_length; ++i) {
        char q_char = (res_s[begin + idx].q_res[i] != (size_t)-1) ? (query[res_s[begin + idx].q_res[i]] + 65) : '-';
        char s_char = (res_s[begin + idx].s_res[i] != (size_t)-1) ? (get_char(target, res_s[begin + idx].s_res[i]) + 65) : '-';
        
        q_seq += q_char;
        s_seq += s_char;
        
        if (s_char != '-') {
            if (s_char == 95) s_char = '*';
            s_ori += s_char;
        }
        
        if (q_char != '-' && s_char != '-') {
            if (q_char == s_char) {
                match += q_char;
            } else {
                match += ' ';
                result.mismatch++;
            }
            
            if (BLOSUM62[(q_char - 65) * 26 + (s_char - 65)] > 0) {
                result.positive++;
                if (q_char != s_char) match[match.length() - 1] = '+';
            }
        } else {
            match += ' ';
        }
    }
    
    result.n_identity = result.align_length - result.mismatch;
    result.p_identity = (1 - (double)result.mismatch / result.align_length) * 100;
    result.bitscore = (E_lambda * result.score - log(E_k)) / 0.69314718055995;

    if (has_must_include && !check_include(s_ori)) {
        result.report = false;
        return;
    }

    if (detailed_alignment) {
        result.q = q_seq;
        result.s = s_seq;
        result.s_ori = s_ori;
        result.match = match;
    }
}

// void cigar_to_index_and_report(size_t idx, int begin, int* cigar_len, char* cigar_op, int* cigar_cnt,
//                size_t* q_start,
//                size_t* c_start,
//                std::vector<SWResult>& res_s,
//             //    uint32_t* num_task,
//                int* score, Task* task, const char* query, const char* target)
// {
//     // res_s.resize(*num_task);
//     cigar_to_index(idx, begin, cigar_len, cigar_op, cigar_cnt, q_start, c_start, res_s);
//     generate_report(idx, begin, res_s, score, task, query, target);
//     assert(res_s.size());
// }



void banded_sw_cpu_kernel(
                uint32_t max_len_query,
                int num_task,
                uint32_t* q_lens, uint32_t* q_idxs, Task* task,
                const char* query, const char* target, size_t target_len,
                int * max_score,
                size_t* q_end_idx, size_t* s_end_idx,
                char* cigar_op, int* cigar_cnt,int* cigar_len,
                int *direct_matrix, record* tile_matrix,int band_width,
                const int* BLOSUM62){
double start_time = omp_get_wtime();
nvtxRangePushA("OpenMP Parallel Region");
#pragma omp parallel for schedule(dynamic)
    for(int idx = 0; idx < num_task; ++ idx){
        int* rd = direct_matrix + idx *  (max_len_query+1) * MaxBW;
        record* tile = tile_matrix + idx *  MaxBW * (TILE_SIZE + 1); 
        memset(rd,0,sizeof(int) *  (max_len_query+1) * MaxBW);
        memset(tile,0,sizeof(record) *  MaxBW * (TILE_SIZE + 1));
        size_t query_len = q_lens[task[idx].q_id];
        assert( query_len < max_len_query);

        size_t q_idx  = q_idxs[task[idx].q_id];
        size_t diag  = task[idx].key;

        int64_t c_begin = (int64_t)diag - band_width - query_len + 2;
        // size_t c_end = diag + band_width;

        size_t width = 2 * band_width + 1;
        size_t height = max_len_query + 1;
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
                    char chc = get_char(target, c_begin + q_offset + _c + _q);
                    
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
                    // if(score)
                    //     rd[calIndex(_c, _q+q_offset, height)] = \
                    //         (score == rt[calIndex(_q,_c,width)].m) ? DIAG : \
                    //         ((score == rt[calIndex(_q,_c,width)].y) ? LEFT :TOP );
                        
                    rd[calIndex(_c, _q+q_offset,height)] = (score?( \
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
        assert(Score != 0 && Score < 1e8);

        size_t cur_q= max_q;
        size_t cur_c = max_c;

        q_end_idx[idx] = cur_q + q_idx;
        s_end_idx[idx] = c_begin + cur_c + cur_q;

        // int cnt_q = 0, cnt_c = 0;
        int cigar_cur_len = 0;
        while (rd[calIndex(cur_c,cur_q,height)])
        {
            int d = rd[calIndex(cur_c,cur_q,height)];
            // size_t res_q = (d&0x01) ? (cur_q + q_idx) : (size_t)-1;
            // size_t res_c = (d&0x02) ? (c_begin + cur_c + cur_q) : (size_t)-1;
            
            // q_res_d[idx* MaxAlignLen + (cnt_q)] = (res_q);
            // s_res_d[idx* MaxAlignLen + (cnt_c)] = (res_c);
            int cur_cigar_cnt = 0;
            while (rd[calIndex(cur_c,cur_q,height)] && rd[calIndex(cur_c,cur_q,height)]==d){
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
        if(cigar_cur_len >= MaxAlignLen)
            printf("@@ cigar_len %d %d\n", idx,cigar_cur_len);
        assert(cigar_cur_len > 0 && cigar_cur_len < MaxAlignLen);
        cigar_len[idx] = cigar_cur_len;
    }
    double end_time = omp_get_wtime();
    double parallel_time = end_time - start_time;
    nvtxRangePop();

    // printf("omp time %f, task = %d\n",parallel_time, num_task);
}

void banded_sw_cpu_kernel_per_task(
                uint32_t max_len_query,
                size_t idx,
                uint32_t* q_lens, uint32_t* q_idxs, Task* task,
                const char* query, const char* target, size_t target_len,
                int * max_score,
                size_t* q_end_idx, size_t* s_end_idx,
                char* cigar_op, int* cigar_cnt,int* cigar_len,
                int band_width,
                const int* BLOSUM62){
        size_t query_len = q_lens[task[idx].q_id];
        assert( query_len <= max_len_query);
        int width = 2 * band_width + 1;
        size_t height = query_len + 1;
        // printf("band_width = %d width = %d\t MaxBw = %d\n",band_width, width, MaxBW);
        assert(width < MaxBW);

        // int* rd = direct_matrix + idx *  (max_len_query+1) * MaxBW;
        // record* tile = tile_matrix + idx *  MaxBW * (TILE_SIZE + 1); 
        int* rd = (int*)malloc(sizeof(int) *  (query_len+1) * width);
        record* tile = (record*)malloc(sizeof(record) *  MaxBW * (TILE_SIZE + 1));
        memset(rd,0,sizeof(int) *  (query_len+1) * width);
        memset(tile,0,sizeof(record) *  MaxBW * (TILE_SIZE + 1));

        size_t q_idx  = q_idxs[task[idx].q_id];
        size_t diag  = task[idx].key;

        int64_t c_begin = (int64_t)diag - band_width - query_len + 2;
        // size_t c_end = diag + band_width;


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
                    char chc = get_char(target, c_begin + q_offset + _c + _q);
                    
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
                    // if(score)
                    //     rd[calIndex(_c, _q+q_offset, height)] = \
                    //         (score == rt[calIndex(_q,_c,width)].m) ? DIAG : \
                    //         ((score == rt[calIndex(_q,_c,width)].y) ? LEFT :TOP );
                        
                    rd[calIndex(_c, _q+q_offset,height)] = (score?( \
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
        assert(Score != 0 && Score < 1e8);

        size_t cur_q= max_q;
        size_t cur_c = max_c;

        q_end_idx[idx] = cur_q + q_idx;
        s_end_idx[idx] = c_begin + cur_c + cur_q;

        int cigar_cur_len = 0;
        assert(rd[calIndex(max_c, max_q,height)] != 0);
        assert(max_q == cur_q && max_c == cur_c);
        assert(rd[calIndex(cur_c, cur_q,height)] != 0);
        while (rd[calIndex(cur_c,cur_q,height)])
        {
            int d = rd[calIndex(cur_c,cur_q,height)];
            // size_t res_q = (d&0x01) ? (cur_q + q_idx) : (size_t)-1;
            // size_t res_c = (d&0x02) ? (c_begin + cur_c + cur_q) : (size_t)-1;
            
            // q_res_d[idx* MaxAlignLen + (cnt_q)] = (res_q);
            // s_res_d[idx* MaxAlignLen + (cnt_c)] = (res_c);
            int cur_cigar_cnt = 0;
            while (rd[calIndex(cur_c,cur_q,height)] && rd[calIndex(cur_c,cur_q,height)]==d){
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
            if(cigar_cur_len >= MaxAlignLen ){
                cigar_cur_len --;
                break;
            }
        }

        assert(cigar_cur_len > 0 && cigar_cur_len < MaxAlignLen);
        cigar_len[idx] = cigar_cur_len;
        free(tile);
        free(rd);
}

void banded_sw_cpu_kernel_thread_pool(
                uint32_t max_len_query,
                int num_task,
                uint32_t* q_lens, uint32_t* q_idxs, Task* task,
                const char* query, const char* target, size_t target_len,
                int * max_score,
                size_t* q_end_idx, size_t* s_end_idx,
                char* cigar_op, int* cigar_cnt,int* cigar_len,
                const int band_width,
                const int* BLOSUM62, ThreadPool* sw_pool){
    for(size_t idx = 0; idx < num_task; ++ idx){
        sw_pool->enqueue([=] {
            banded_sw_cpu_kernel_per_task(
                max_len_query,
                idx, q_lens, q_idxs, task,
                query, target, target_len,
                max_score, q_end_idx, s_end_idx,
                cigar_op, cigar_cnt, cigar_len,
                band_width,
                BLOSUM62);
        });
    }
}

void banded_sw_cpu_kernel_api(
                uint32_t max_len_query,
                int num_task,
                uint32_t* q_lens, uint32_t* q_idxs, Task* task,
                const char* query, const char* target, size_t target_len,
                int * max_score,
                size_t* q_end_idx, size_t* s_end_idx,
                char* cigar_op, int* cigar_cnt,int* cigar_len,
                int *direct_matrix, record* tile_matrix,int band_width,
                const int* BLOSUM62){
// #pragma omp parallel for
    for(int idx = 0; idx < num_task; ++ idx){
        int* rd = direct_matrix + idx *  (max_len_query+1) * MaxBW;
        record* tile = tile_matrix + idx *  MaxBW * (TILE_SIZE + 1); 
        memset(rd,0,sizeof(int) *  (max_len_query+1) * MaxBW);
        memset(tile,0,sizeof(record) *  MaxBW *(TILE_SIZE + 1));
        size_t query_len = q_lens[task[idx].q_id];
        assert( query_len < max_len_query);

        size_t q_idx  = q_idxs[task[idx].q_id];
        size_t diag  = task[idx].key;

        int64_t c_begin = (int64_t)diag - band_width - query_len + 2;
        // size_t c_end = diag + band_width;
 
        size_t width = 2 * band_width + 1;
        size_t height = max_len_query + 1;
        assert(width < MaxBW);

        //init:
        max_score[idx] = 0;

        // size_t t_height = TILE_SIZE + 1;

        // record *rt = (record *)malloc(width * t_height * sizeof(record));
        // memset(rt, 0, width * t_height * sizeof(record));

        size_t max_q = 0;
        size_t max_c = 0;
        int score = 0, Score = 0;
        // cal maxScore and it's position
        size_t query_begin = q_idx;
        int64_t target_begin = c_begin;
        size_t logic_target_len = 2 * band_width + query_len - 2;
        // tile 2 * width
        // record *tile = (record*) malloc(2 * MaxBW * sizeof(record));
        // memset(tile, 0, 2 * MaxBW * sizeof(record));
        
        for(int64_t _q = 0; _q < query_len; ++ _q){
            size_t current = _q % 2;
            size_t previous = 1 - current;
            // memset(tile + previous * MaxBW,0,sizeof(record) *  MaxBW);
            for(int64_t _c = 0; _c < logic_target_len; ++ _c){
                
                if(_c < _q) continue;
                if(_c > _q + 2*band_width -2) break;

                int64_t ch_query_pos = query_begin + _q;
                int64_t ch_target_pos = target_begin + _c;
                
                // TODO Judge:
                // 1. ch_target_pos < 0 continue;
                // 2. ch_target_pos > total_target_len break;
                // 3. _c 是否属于 [_q, _q + 2 * bw -1) 不属于 break;
                // _c < _q ||  _c > _q + 2*band_width -2 : 不在tile内
                // ch_target_pos < 0 && _c >= _q tile 内失效
                if(ch_target_pos < 0 ) {
                    
                    tile[calIndex2D(current, _c-_q, MaxBW)].x = 0;
                    tile[calIndex2D(current, _c-_q, MaxBW)].y = 0;
                    tile[calIndex2D(current, _c-_q, MaxBW)].m = 0;
                    continue;
                }
                
                if(ch_target_pos >= target_len ){
                    for(int k = _c; k <=  _q + 2*band_width -2; ++ k){

                        tile[calIndex2D(current, k-_q, MaxBW)].x = 0;
                        tile[calIndex2D(current, k-_q, MaxBW)].y = 0;
                        tile[calIndex2D(current, k-_q, MaxBW)].m = 0;
                    }
                    break;
                }
                
                char chq = query[ch_query_pos];
                char chc = get_char(target, ch_target_pos);
                if (chq == END_SIGNAL || chc == END_SIGNAL)
                {
                    continue;
                }
                
                // 获取依赖关系
                if(_q > 0 && _c  < _q + 2 * band_width -2){

                    tile[calIndex2D(current, _c-_q, MaxBW)].x = max3(tile[calIndex2D(previous,_c - _q+1,MaxBW)].x + SCORE_GAP_EXT,  tile[calIndex2D(previous,_c - _q+1,MaxBW)].m + SCORE_GAP, 0);
                }
                else{
                    tile[calIndex2D(current, _c-_q, MaxBW)].x = 0;

                }
                
                if(_c-1 >= _q && target_begin + _c - 1 >= 0 && _c-1 >= _q){

                    tile[calIndex2D(current, _c-_q, MaxBW)].y = max3(tile[calIndex2D(current,_c - _q-1,MaxBW)].y + SCORE_GAP_EXT, tile[calIndex2D(current,_c - _q-1,MaxBW)].m + SCORE_GAP, 0);
                }
                else{

                    tile[calIndex2D(current, _c-_q, MaxBW)].y = 0;
                }
                
                
                if (chq == ILLEGAL_WORD || chc == ILLEGAL_WORD)
                {
                   tile[calIndex2D(current, _c-_q, MaxBW)].m = 0;
                }
                else
                {
                    if(_q > 0 && target_begin + _c - 1 >= 0 && _c-1 >= _q){

                        tile[calIndex2D(current, _c-_q, MaxBW)].m = max2(max3(tile[calIndex2D(previous,_c - _q,MaxBW)].x, tile[calIndex2D(previous,_c - _q,MaxBW)].y, tile[calIndex2D(previous,_c - _q,MaxBW)].m) + BLOSUM62[chq * 26 + chc], 0);
                    }
                    else{

                        tile[calIndex2D(current, _c-_q, MaxBW)].m = max2(max3(0,0, 0)+ BLOSUM62[chq * 26 + chc], 0);
                    } 
                }

                score = max3(tile[calIndex2D(current,_c - _q,MaxBW)].x, tile[calIndex2D(current,_c - _q,MaxBW)].y, tile[calIndex2D(current,_c - _q,MaxBW)].m);
                        
                
                // cal score
               rd[calIndex(_c - _q, _q,height)] = (score?( \
                        (score == tile[calIndex2D(current,_c - _q,MaxBW)].m) ? DIAG : \
                        ((score == tile[calIndex2D(current,_c - _q,MaxBW)].y) ? LEFT :TOP )):0);
                    
                if (Score < score)
                {
                    Score = score;
                    max_c = _c;
                    max_q = _q;
                }
            }
        }
        // free(tile);
        max_score[idx] = Score;
        // res[idx].score = Score;
        assert(Score != 0 && Score < 1e8);

        size_t cur_q= max_q;
        size_t cur_c = max_c;

        q_end_idx[idx] = max_q + query_begin;
        s_end_idx[idx] = max_c + target_begin;

        // int cnt_q = 0, cnt_c = 0;
        int cigar_cur_len = 0;
        assert(rd[calIndex(cur_c-cur_q,cur_q,height)] != 0);
        while (rd[calIndex(cur_c-cur_q,cur_q,height)])
        {
            int d = rd[calIndex(cur_c-cur_q,cur_q,height)];
            // size_t res_q = (d&0x01) ? (cur_q + q_idx) : (size_t)-1;
            // size_t res_c = (d&0x02) ? (c_begin + cur_c + cur_q) : (size_t)-1;
            
            // q_res_d[idx* MaxAlignLen + (cnt_q)] = (res_q);
            // s_res_d[idx* MaxAlignLen + (cnt_c)] = (res_c);
            int cur_cigar_cnt = 0;
            while (rd[calIndex(cur_c-cur_q,cur_q,height)] && rd[calIndex(cur_c-cur_q,cur_q,height)]==d){
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
        if(cigar_cur_len >= MaxAlignLen || cigar_cur_len <= 0)
            printf("@@ cigar_len %d %d\n", idx,cigar_cur_len);
        assert(cigar_cur_len > 0 && cigar_cur_len < MaxAlignLen);
        cigar_len[idx] = cigar_cur_len;
    }
}
