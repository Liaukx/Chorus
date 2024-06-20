#pragma once

#include "params.h"
#include "util.h"
#include <cuda_runtime.h>

struct KeyValue
{
    uint32_t key;
    uint32_t value;
};

struct Task
{
    uint32_t key;
    uint16_t value;
    uint16_t q_id;
};

const uint32_t kEmpty = 0xffffffff;

KeyValue *create_hashtable(size_t size);
KeyValue *create_hashtable_async(size_t size, cudaStream_t& s);

void destroy_hashtable(KeyValue *hashtable);
void destroy_hashtable_async(KeyValue *hashtable, cudaStream_t& s);
