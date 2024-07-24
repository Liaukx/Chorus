#pragma once

// #include "cuda_runtime.h"
#include "argparse.hpp"
#include <vector>
#include <string>

#define NUM_STREAM 16
#define STREAM_SYNC 0
#define MAX_QUERY_PER_GROUP 4096
#define MAX_NUM_GROUPS 256
#define MAX_GROUPS_PER_ROUND 1 //larger more cpu memory
#define FILTER_ALL_SAME_AA_SEED true
#define MAX_FILTER_TASK 800000

#define D_SEED_LENGTH 5
#define D_QIT_WIDTH 4
#define D_MAX_DISPLAY_ALIGN 0
#define D_BAND_WIDTH_CPU 8
#define D_BAND_WIDTH_GPU 16
#define D_OUTFMT 0
#define D_FILTER_LEVEL 1
#define D_MIN_SCORE 0
#define D_MAX_EVALUE 1e1
#define D_HASH_RATIO 2

struct Condition
{
    int t = 0;
    std::string re;
};

extern ArgumentParser arg_parser;
extern int seed_length;
extern int qit_length;
extern int qit_width;
extern int max_display_align;
extern int band_width;
extern std::vector<Condition> must_include;
extern bool has_must_include;
extern int num_threads;
extern int min_score;
extern double max_evalue;
extern bool detailed_alignment;

// A-Z: 0-25, end of seq: 0b11111(31), illegal word: 0b11110(30)
#define ILLEGAL_WORD 30
#define END_SIGNAL 31

// static const int CTOI[128] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,-1,1,2,3,4,5,6,7,-1,8,9,10,11,-1,12,13,14,15,16,-1,17,18,-1,19,-1,-1,-1,-1,-1,-1,-1,0,-1,1,2,3,4,5,6,7,-1,8,9,10,11,-1,12,13,14,15,16,-1,17,18,-1,19,-1,-1,-1,-1,-1,-1};

// static const char ITOC[] = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#'};
#define SCORE_GAP -11
#define SCORE_GAP_EXT -1
#define E_k 0.041
#define E_lambda 0.267
static const int BLOSUM62[] = {4, -2, 0, -2, -1, -2, 0, -2, -1, -1, -1, -1, -1, -2, 0, -1, -1, -1, 1, 0, 0, 0, -3, -1, -2, -1,
                               -2, 4, -3, 4, 1, -3, -1, 0, -3, -3, 0, -4, -3, 4, 0, -2, 0, -1, 0, -1, 0, -3, -4, -1, -3, 0,
                               0, -3, 9, -3, -4, -2, -3, -3, -1, -1, -3, -1, -1, -3, 0, -3, -3, -3, -1, -1, 0, -1, -2, -1, -2, -3,
                               -2, 4, -3, 6, 2, -3, -1, -1, -3, -3, -1, -4, -3, 1, 0, -1, 0, -2, 0, -1, 0, -3, -4, -1, -3, 1,
                               -1, 1, -4, 2, 5, -3, -2, 0, -3, -3, 1, -3, -2, 0, 0, -1, 2, 0, 0, -1, 0, -2, -3, -1, -2, 4,
                               -2, -3, -2, -3, -3, 6, -3, -1, 0, 0, -3, 0, 0, -3, 0, -4, -3, -3, -2, -2, 0, -1, 1, -1, 3, -3,
                               0, -1, -3, -1, -2, -3, 6, -2, -4, -4, -2, -4, -3, 0, 0, -2, -2, -2, 0, -2, 0, -3, -2, -1, -3, -2,
                               -2, 0, -3, -1, 0, -1, -2, 8, -3, -3, -1, -3, -2, 1, 0, -2, 0, 0, -1, -2, 0, -3, -2, -1, 2, 0,
                               -1, -3, -1, -3, -3, 0, -4, -3, 4, 3, -3, 2, 1, -3, 0, -3, -3, -3, -2, -1, 0, 3, -3, -1, -1, -3,
                               -1, -3, -1, -3, -3, 0, -4, -3, 3, 3, -3, 3, 2, -3, 0, -3, -2, -2, -2, -1, 0, 2, -2, -1, -1, -3,
                               -1, 0, -3, -1, 1, -3, -2, -1, -3, -3, 5, -2, -1, 0, 0, -1, 1, 2, 0, -1, 0, -2, -3, -1, -2, 1,
                               -1, -4, -1, -4, -3, 0, -4, -3, 2, 3, -2, 4, 2, -3, 0, -3, -2, -2, -2, -1, 0, 1, -2, -1, -1, -3,
                               -1, -3, -1, -3, -2, 0, -3, -2, 1, 2, -1, 2, 5, -2, 0, -2, 0, -1, -1, -1, 0, 1, -1, -1, -1, -1,
                               -2, 4, -3, 1, 0, -3, 0, 1, -3, -3, 0, -3, -2, 6, 0, -2, 0, 0, 1, 0, 0, -3, -4, -1, -2, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               -1, -2, -3, -1, -1, -4, -2, -2, -3, -3, -1, -3, -2, -2, 0, 7, -1, -2, -1, -1, 0, -2, -4, -1, -3, -1,
                               -1, 0, -3, 0, 2, -3, -2, 0, -3, -2, 1, -2, 0, 0, 0, -1, 5, 1, 0, -1, 0, -2, -2, -1, -1, 4,
                               -1, -1, -3, -2, 0, -3, -2, 0, -3, -2, 2, -2, -1, 0, 0, -2, 1, 5, -1, -1, 0, -3, -3, -1, -2, 0,
                               1, 0, -1, 0, 0, -2, 0, -1, -2, -2, 0, -2, -1, 1, 0, -1, 0, -1, 4, 1, 0, -2, -3, -1, -2, 0,
                               0, -1, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, 1, 5, 0, 0, -2, -1, -2, -1,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, -3, -1, -3, -2, -1, -3, -3, 3, 2, -2, 1, 1, -3, 0, -2, -2, -3, -2, 0, 0, 4, -3, -1, -1, -2,
                               -3, -4, -2, -4, -3, 1, -2, -2, -3, -2, -3, -2, -1, -4, 0, -4, -2, -3, -3, -2, 0, -3, 11, -1, 2, -2,
                               -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1,
                               -2, -3, -2, -3, -2, 3, -3, 2, -1, -1, -2, -1, -1, -2, 0, -3, -1, -2, -2, -2, 0, -1, 2, -1, 7, -2,
                               -1, 0, -3, 1, 4, -3, -2, 0, -3, -3, 1, -3, -1, 0, 0, -1, 4, 0, 0, -1, 0, -2, -2, -1, -2, 4};