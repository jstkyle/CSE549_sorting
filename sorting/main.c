#include <bsg_manycore_tile.h>
#include <bsg_manycore_errno.h>
#include <bsg_manycore_loader.h>
#include <bsg_manycore_cuda.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <bsg_manycore_regression.h>

#define ALLOC_NAME "default_allocator"

#ifndef SIZE
#error "Please define SIZE in Makefile"
#endif

typedef struct {
    uint32_t key;
    uint32_t value;
} KeyValue;

static int compare(const void* a, const void* b) {
    return ((KeyValue*)a)->value - ((KeyValue*)b)->value;
}

int kernel_sort(int argc, char **argv) {
    char *bin_path, *test_name;
    struct arguments_path args = {NULL, NULL};
    
    argp_parse(&argp_path, argc, argv, 0, 0, &args);
    bin_path = args.path;
    test_name = args.name;
    
    bsg_pr_test_info("Running simple quicksort kernel\n");
    srand(time(NULL));
    
    // Initialize Device
    hb_mc_device_t device;
    BSG_CUDA_CALL(hb_mc_device_init(&device, test_name, 0));
    
    hb_mc_pod_id_t pod;
    hb_mc_device_foreach_pod_id(&device, pod) {
        BSG_CUDA_CALL(hb_mc_device_set_default_pod(&device, pod));
        BSG_CUDA_CALL(hb_mc_device_program_init(&device, bin_path, ALLOC_NAME, 0));
        
        // Allocate arrays
        KeyValue* input_host = (KeyValue*)malloc(SIZE * sizeof(KeyValue));
        KeyValue* output_host = (KeyValue*)malloc(SIZE * sizeof(KeyValue));
        KeyValue* verify_host = (KeyValue*)malloc(SIZE * sizeof(KeyValue));
        
        // Initialize arrays
        for (int i = 0; i < SIZE; i++) {
            uint32_t val = rand() % SIZE;
            input_host[i].key = i;
            input_host[i].value = val;
            verify_host[i].key = i;
            verify_host[i].value = val;
        }
        
        // Sort verification array
        qsort(verify_host, SIZE, sizeof(KeyValue), compare);
        
        // Allocate device memory
        eva_t input_device;
        BSG_CUDA_CALL(hb_mc_device_malloc(&device, SIZE * sizeof(KeyValue), &input_device));
        
        // Transfer to device
        hb_mc_dma_htod_t htod_job[] = {
            {
                .d_addr = input_device,
                .h_addr = input_host,
                .size = SIZE * sizeof(KeyValue)
            }
        };
        BSG_CUDA_CALL(hb_mc_device_dma_to_device(&device, htod_job, 1));
        
        // Launch kernel
        hb_mc_dimension_t tg_dim = {.x = bsg_tiles_X, .y = bsg_tiles_Y};
        hb_mc_dimension_t grid_dim = {.x = 1, .y = 1};
        uint32_t cuda_argv[] = {input_device, SIZE};
        
        BSG_CUDA_CALL(hb_mc_kernel_enqueue(&device, grid_dim, tg_dim, "kernel_sort", 2, cuda_argv));
        BSG_CUDA_CALL(hb_mc_device_tile_groups_execute(&device));
        
        // Copy result back
        hb_mc_dma_dtoh_t dtoh_job[] = {
            {
                .d_addr = input_device,
                .h_addr = output_host,
                .size = SIZE * sizeof(KeyValue)
            }
        };
        BSG_CUDA_CALL(hb_mc_device_dma_to_host(&device, dtoh_job, 1));
        
        // Verify results
        for (int i = 0; i < SIZE; i++) {
            if (verify_host[i].value != output_host[i].value) {
                printf("FAIL: Mismatch at %d (expected %u, got %u)\n",
                       i, verify_host[i].value, output_host[i].value);
                return HB_MC_FAIL;
            }
        }
        
        printf("PASS: Sort completed successfully\n");
        
        // Cleanup
        free(input_host);
        free(output_host);
        free(verify_host);
        BSG_CUDA_CALL(hb_mc_device_program_finish(&device));
    }
    
    BSG_CUDA_CALL(hb_mc_device_finish(&device));
    return HB_MC_SUCCESS;
}

declare_program_main("quicksort", kernel_sort);