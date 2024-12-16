#include <bsg_manycore.h>
#include <bsg_cuda_lite_barrier.h>
#include <stdint.h>

struct KeyValue {
    int32_t key;
    int32_t value;
};

#define CHUNK_SIZE 256

// Simple insertion sort for small chunks
void insertionSort(struct KeyValue* arr, int start, int count) {
    for (int i = start + 1; i < start + count; i++) {
        struct KeyValue key = arr[i];
        int j = i - 1;
        
        while (j >= start && arr[j].value > key.value) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// Merge function using stack allocation
void merge(struct KeyValue* arr, int start, int mid, int end) {
    struct KeyValue temp[CHUNK_SIZE];
    int offset = 0;
    
    while (offset < end - start) {
        int current_chunk = ((end - start - offset) < CHUNK_SIZE) ? 
                           (end - start - offset) : CHUNK_SIZE;
        
        int left = start + offset;
        int right = mid;
        int k = 0;
        
        // Merge chunks
        while (left < mid && right < end && k < current_chunk) {
            if (arr[left].value <= arr[right].value) {
                temp[k++] = arr[left++];
            } else {
                temp[k++] = arr[right++];
            }
        }
        
        while (left < mid && k < current_chunk) {
            temp[k++] = arr[left++];
        }
        
        while (right < end && k < current_chunk) {
            temp[k++] = arr[right++];
        }
        
        // Copy back the merged chunk
        for (int i = 0; i < k; i++) {
            arr[start + offset + i] = temp[i];
        }
        
        offset += current_chunk;
    }
}

void warmCache(struct KeyValue* arr, int size) {
    int tid = __bsg_id;
    int tiles = bsg_tiles_X * bsg_tiles_Y;
    
    for (int i = tid; i < size; i += tiles) {
        volatile int temp = arr[i].value;
        asm volatile("" : : "r" (temp));
    }
    bsg_fence();
}

extern "C" __attribute__ ((noinline))
int kernel_sort(struct KeyValue* input, int size) {
    bsg_barrier_hw_tile_group_init();
    
    // Warm cache
    warmCache(input, size);
    bsg_barrier_hw_tile_group_sync();
    
    bsg_cuda_print_stat_kernel_start();
    
    int tiles = bsg_tiles_X * bsg_tiles_Y;
    int tid = __bsg_id;
    int chunk_size = size / tiles;
    int start = tid * chunk_size;
    
    // Adjust last tile's chunk size
    if (tid == tiles - 1) {
        chunk_size += size % tiles;
    }
    
    // Local sort
    insertionSort(input, start, chunk_size);
    bsg_barrier_hw_tile_group_sync();
    
    // Merge phase
    for (int step = 1; step < tiles; step *= 2) {
        if (tid % (2 * step) == 0) {
            int merge_start = tid * chunk_size;
            int merge_mid = merge_start + step * chunk_size;
            int merge_end = merge_mid + step * chunk_size;
            
            if (merge_mid < size) {
                if (merge_end > size) merge_end = size;
                merge(input, merge_start, merge_mid, merge_end);
            }
        }
        bsg_barrier_hw_tile_group_sync();
    }
    
    bsg_fence();
    bsg_cuda_print_stat_kernel_end();
    bsg_fence();
    
    bsg_barrier_hw_tile_group_sync();
    return 0;
}