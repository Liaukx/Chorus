#include "smith.h"
// #include <libunwind.h>
// #include <gperftools/tcmalloc.h>

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
    int d;
    int x; // top
    int y; // left
    int m; // left top
    int s;
} record;

inline char get_char(const char *s, size_t offset)
{
    size_t n_bit = offset * 5;
    return MASK5((unsigned)((*((uint16_t *)&(s[n_bit >> 3]))) >> (n_bit & 7)));
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

        for(size_t _q = 1; _q < t_height && q_offset + _q < height; ++_q){
            for(size_t _c = 1; _c < width-1; ++_c){
                
                if(c_begin + _c-1+ q_offset + _q-1 < 0) continue;
                if(c_begin + _c-1+ q_offset + _q-1 >= c_len) break;

                char chq = q[q_idx + q_offset + _q-1];
                char chc = get_char(c, c_begin + q_offset + _c-1 + _q-1);
                
                if (chq == END_SIGNAL || chc == END_SIGNAL)
                {
                    continue;
                }

                rt[_q * width + _c].x = max3(rt[(_q - 1) * width + (_c + 1)].x + SCORE_GAP_EXT,  rt[(_q - 1) * width + (_c + 1)].m + SCORE_GAP, 0);
                rt[_q * width + _c].y = max3(rt[_q * width + (_c - 1)].y + SCORE_GAP_EXT, rt[_q * width + (_c - 1)].m + SCORE_GAP, 0);

                if (chq == ILLEGAL_WORD || chc == ILLEGAL_WORD)
                {
                    // illegal word
                    rt[_q * width + _c].m = 0;
                }
                else
                {
                    rt[_q * width + _c].m = max2(max3(rt[(_q - 1) * width + _c].x, rt[(_q - 1) * width + _c].y, rt[(_q - 1) * width + _c].m) + BLOSUM62[chq * 26 + chc], 0);
                }

                score = max3(rt[_q * width + _c].x, rt[_q * width + _c].y, rt[_q * width + _c].m);
                // printf("(q = %c,c = %c) BLOSUM62 = %d rt[_q * width + _c].s = %d\n", chq+65,chc+65,BLOSUM62[chq * 26 + chc], rt[_q * width + _c].s);
                
                if (score != 0)
                {
                    if (score == rt[_q * width + _c].x)
                        rd[_c * height + _q + q_offset] = TOP;
                    if (score == rt[_q * width + _c].y)
                        rd[_c * height + _q + q_offset] = LEFT;
                    if (score == rt[_q * width + _c].m)
                        rd[_c * height + _q + q_offset] = DIAG;
                }
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

    // for (int i = 0; i < height; i++)
    // {
    //     for (int j = 0; j < width; j++)
    //     {
    //         cout << s[j * height + i] << "\t";
    //     }
    //     cout << endl;
    // }

    res->score = Score;
    assert(res->score != 0);

    size_t cur_q= max_q;
    size_t cur_c = max_c;
    
    while (rd[cur_c * height + cur_q])
    {
        int d = rd[cur_c * height + cur_q];
        int64_t res_q = (d&0x01) ? ((cur_q-1) + q_idx) : -1;
        int64_t res_c = (d&0x02) ? (c_begin + cur_c-1 + cur_q-1) : -1;
        
        res->q_res.push_back(res_q);
        res->s_res.push_back(res_c);
        //TOP 01b, left 10b, diag 11b
        //DIAG : cur_q -= 1
        //TOP : cur_q -= 1, cur_c += 1;
        //LEFT : cur_c -= 1
        cur_q -= (d == DIAG || d == TOP);
        cur_c += (d == TOP);
        cur_c -= (d == LEFT);
    }

    free(rd);
    free(rt);

    reverse(res->q_res.begin(), res->q_res.end());
    reverse(res->s_res.begin(), res->s_res.end());
    generate_report(res,q, c);

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