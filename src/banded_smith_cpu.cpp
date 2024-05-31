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
    record *r = (record *)malloc(width * height * sizeof(record));

    if (r == nullptr)
    {
        printf("CPU out of memory!\n");
        exit(-1);
        return;
    }

    // memset(s, 0, width * height * sizeof(short));
    // memset(p, 0, width * height * sizeof(char));
    memset(r, 0, width * height * sizeof(record));

    size_t max_q = 0;
    size_t max_c = 0;
    
    size_t offset_q = q_idx, offset_c = c_begin;

    for(size_t _q = 1; _q < height; ++_q){
        for(size_t _c = 1; _c < width-1; ++_c){
            if (offset_c + _c-1+_q-1 < 0 || offset_c + _c-1+_q-1 >= c_len)
            {
                continue;
            }

            char chq = q[offset_q + _q-1];
            char chc = get_char(c, offset_c + _c-1+_q-1);
            
            if (chq == END_SIGNAL || chc == END_SIGNAL)
            {
                continue;
            }

            assert(_q * width + _c < width * height);
            assert(_q - 1 >= 0);
            assert(_c + 1 < height);
            assert(_c - 1 >= 0);

            r[_c * height + _q].x = max3(r[(_c + 1) * height + (_q - 1)].x + SCORE_GAP_EXT,  r[(_c + 1) * height + (_q - 1)].m + SCORE_GAP, 0);
            r[_c * height + _q].y = max3(r[(_c - 1) * height + _q].y + SCORE_GAP_EXT, r[(_c - 1) * height + _q].m + SCORE_GAP, 0);
            // r[_q * width + _c].x = max3(r[(_q - 1) * width + (_c + 1)].x + SCORE_GAP_EXT,  r[(_q - 1) * width + (_c + 1)].m + SCORE_GAP, 0);
            // r[_q * width + _c].y = max3(r[_q * width + (_c - 1)].y + SCORE_GAP_EXT, r[_q * width + (_c - 1)].m + SCORE_GAP, 0);

            if (chq == ILLEGAL_WORD || chc == ILLEGAL_WORD)
            {
                // illegal word
                // r[_q * width + _c].m = 0;
                r[_c * height + _q].m = 0;
            }
            else
            {
                // r[_q * width + _c].m = max2(max3(r[(_q - 1) * width + _c].x, r[(_q - 1) * width + _c].y, r[(_q - 1) * width + _c].m) + BLOSUM62[chq * 26 + chc], 0);
                r[_c * height+ _q].m = max2(max3(r[_c * height + (_q - 1)].x, r[_c * height + (_q - 1)].y, r[_c * height + (_q - 1)].m) + BLOSUM62[chq * 26 + chc], 0);
            }

            // r[_q * width + _c].s = max3(r[_q * width + _c].x, r[_q * width + _c].y, r[_q * width + _c].m);
            r[_c * height + _q].s = max3(r[_c * height + _q].x, r[_c * height + _q].y, r[_c * height + _q].m);
            // printf("(q = %c,c = %c) BLOSUM62 = %d r[_q * width + _c].s = %d\n", chq+65,chc+65,BLOSUM62[chq * 26 + chc], r[_q * width + _c].s);
            
            if (r[_c * height + _q].s != 0)
            {
                if (r[_c * height + _q].s == r[_c * height + _q].x)
                    r[_c * height + _q].d = TOP;
                if (r[_c * height + _q].s == r[_c * height + _q].y)
                    r[_c * height + _q].d = LEFT;
                if (r[_c * height + _q].s == r[_c * height + _q].m)
                    r[_c * height + _q].d = DIAG;
            }
            // if (r[_q * width + _c].s != 0)
            // {
            //     if (r[_q * width + _c].s == r[_q * width + _c].x)
            //         r[_q * width + _c].d = TOP;
            //     if (r[_q * width + _c].s == r[_q * width + _c].y)
            //         r[_q * width + _c].d = LEFT;
            //     if (r[_q * width + _c].s == r[_q * width + _c].m)
            //         r[_q * width + _c].d = DIAG;
            // }
            // if (r[_q * width + _c].s > r[max_q * width + max_c].s)
            // {
            //     max_c = _c;
            //     max_q = _q;
            // }

            if (r[_c * height + _q].s > r[max_c * height + max_q].s)
            {
                max_c = _c;
                max_q = _q;
            }
            // printf("(q = %c,c = %c) score = %d maxScore = %d direction = %d\n", chq+65,chc+65,r[_q*width + _c].s,r[max_j * width + max_i].s,r[_q * width + _c].d);
        }
    }

    // for (int i = 0; i < height; i++)
    // {
    //     for (int j = 0; j < width; j++)
    //     {
    //         cout << s[j * height + i] << "\t";
    //     }
    //     cout << endl;
    // }

    // res->score = r[max_q * width + max_c].s;
    res->score = r[max_c * height + max_q].s;
    assert(res->score != 0);

    size_t cur_q= max_q;
    size_t cur_c = max_c;
    
    while (r[cur_c * height + cur_q].s > 0)
    {
        switch (r[cur_c * height + cur_q].d)
    // while (r[cur_q * width + cur_c].s > 0)
    // {
    //     switch (r[cur_q * width + cur_c].d)
        {
        case DIAG:
            res->q_res.push_back((cur_q-1) + q_idx);
            res->s_res.push_back(c_begin + cur_c-1 + cur_q-1);
            cur_q -= 1;
            break;

        case TOP:
            res->q_res.push_back((cur_q-1) + q_idx);
            res->s_res.push_back(-1);
            cur_q -= 1;
            cur_c += 1;
            break;

        case LEFT:
            res->q_res.push_back(-1);
            res->s_res.push_back(c_begin + cur_c-1 + cur_q-1);
            cur_c -= 1;
            break;

        default:
            printf("err\n");
            break;
        }
    }

    free(r);

    reverse(res->q_res.begin(), res->q_res.end());
    reverse(res->s_res.begin(), res->s_res.end());

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