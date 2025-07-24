#ifndef PROPOSAL_CUH
#define PROPOSAL_CUH
#include "common.cuh"

#ifndef MAKE_TW_BEFORE_VALUE
#define MAKE_TW_BEFORE_VALUE
__global__ void make_tw_before_value(int *op_type, int *tw_before_value,
                                     long long all_ops_count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= all_ops_count) {
        return;
    }
    tw_before_value[idx] = (int)(op_type[idx] == TEMP_VER_WRITE);
}
#endif

#ifndef REVERSE_ARRAY
#define REVERSE_ARRAY
__global__ void reverse_array(int *input, int *output,
                              long long all_ops_count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= all_ops_count) {
        return;
    }
    output[all_ops_count - 1 - idx] = input[idx];
}
#endif

#ifndef MAKE_KEY_VALUE_FOR_WB_AND_WA_FOR_PROPOSAL
#define MAKE_KEY_VALUE_FOR_WB_AND_WA_FOR_PROPOSAL
__global__ void make_key_value_for_wb_and_wa_for_proposal(
    OperationForProposal *sorted_op, int *key_for_wb, int *value_for_wb,
    int *key_for_wa, int *value_for_wa, long long all_ops_count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= all_ops_count) {
        return;
    }
    OperationForProposal op = sorted_op[idx];
    key_for_wb[idx] = op.record_id;
    value_for_wb[idx] = op.type;
    key_for_wa[all_ops_count - 1 - idx] = op.record_id;
    value_for_wa[all_ops_count - 1 - idx] = op.type;
}
#endif

#ifndef GET_RW_LOCATION
#define GET_RW_LOCATION
__global__ void get_rw_location(int *op_type, int *tw_before,
                                long long all_ops_count, RWLoc *rw_loc) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= all_ops_count) {
        return;
    }

    if (op_type[idx] == CURR_VER_READ || op_type[idx] == CURR_VER_WRITE) {
        rw_loc[idx].version = CURR_VER;
    } else if (op_type[idx] == PREV_VER_READ) {
        rw_loc[idx].version = PREV_VER;
    } else {
        if (op_type[idx] == TEMP_VER_READ) {
            rw_loc[idx].version = TEMP_VER;
            rw_loc[idx].index = tw_before[idx] - 1;
        } else {
            rw_loc[idx].version = TEMP_VER;
            rw_loc[idx].index = tw_before[idx];
        }
    }
}
#endif

#ifndef GET_OP_TYPE_FOR_PROPOSAL
#define GET_OP_TYPE_FOR_PROPOSAL
__global__ void get_op_type_for_proposal(OperationForProposal *sorted_op,
                                         int *write_before, int *write_after,
                                         int *op_type,
                                         long long all_ops_count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= all_ops_count) {
        return;
    }

    if (sorted_op[idx].type == 1) {
        if (write_after[idx] == 0) {
            op_type[idx] = CURR_VER_WRITE;
        } else {
            op_type[idx] = TEMP_VER_WRITE;
        }
    } else {
        if (write_before[idx] == 0) {
            op_type[idx] = PREV_VER_READ;
        } else if (write_after[idx] == 0) {
            op_type[idx] = CURR_VER_READ;
        } else {
            op_type[idx] = TEMP_VER_READ;
        }
    }
}
#endif

#ifndef SCATTER_FOR_PROPOSAL
#define SCATTER_FOR_PROPOSAL
__global__ void scatter_for_proposal(OperationForProposal *sorted_op,
                                     RWLoc *result, RWLoc *rw_loc,
                                     int all_ops_count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= all_ops_count)
        return;

    int result_idx = sorted_op[idx].idx;
    result[result_idx].version = rw_loc[idx].version;
    result[result_idx].index = rw_loc[idx].index;
}
#endif

#ifndef PROPOSAL_INIT
#define PROPOSAL_INIT
void proposal_init(std::vector<Transaction> &h_txs,
                   std::vector<OperationForProposal> &h_sorted_ops,
                   std::vector<Operation> &h_ops, int transaction_count,
                   int operation_count, std::vector<int> &h_op_id_to_idx,
                   long long all_ops_count) {
    std::vector<
        std::pair<std::string, std::chrono::high_resolution_clock::time_point>>
        times = {{"Start", std::chrono::high_resolution_clock::now()}};
    // // GPU用のメモリを確保する
    // Transaction *d_txs;
    // Operation *d_ops;

    // cudaMalloc(&d_txs, sizeof(Transaction) * h_txs.size());
    // cudaMalloc(&d_ops, sizeof(Operation) * h_ops.size());

    // // // データをGPUに転送
    // cudaMemcpy(d_txs, h_txs.data(), sizeof(Transaction) * h_txs.size(),
    //            cudaMemcpyHostToDevice);
    // cudaMemcpy(d_ops, h_ops.data(), sizeof(Operation) * h_ops.size(),
    //            cudaMemcpyHostToDevice);

    // times.push_back(
    //     {"Data transfer to GPU", std::chrono::high_resolution_clock::now()});
    //      all_ops用のメモリ
    // int *op_id_to_idx;
    // cudaMalloc(&op_id_to_idx, sizeof(int) * all_ops_count);
    // cudaMemcpy(op_id_to_idx, h_op_id_to_idx.data(), sizeof(int) *
    // all_ops_count,
    //            cudaMemcpyHostToDevice);

    //      sorted_op用のメモリ
    auto *tmp = h_sorted_ops.data();
    OperationForProposal *sorted_op;
    // printf("OperationForAllOps Data Size: %d\n",
    //        sizeof(OperationForProposal) * h_sorted_ops.size());
    CHECK_CUDA(
        cudaMalloc(&sorted_op, sizeof(OperationForProposal) * all_ops_count));
    CHECK_CUDA(cudaMemcpy(sorted_op, tmp,
                          sizeof(OperationForProposal) * all_ops_count,
                          cudaMemcpyHostToDevice));

    times.push_back({"Sorted Data Transfer Done\t",
                     std::chrono::high_resolution_clock::now()});
    // write_before, write_after
    int *write_before, *write_after_rev, *write_after, *key_for_wb,
        *value_for_wb, *key_for_wa, *value_for_wa;
    CHECK_CUDA(cudaMalloc(&write_before, sizeof(int) * all_ops_count));
    CHECK_CUDA(cudaMalloc(&write_after_rev, sizeof(int) * all_ops_count));
    CHECK_CUDA(cudaMalloc(&write_after, sizeof(int) * all_ops_count));
    CHECK_CUDA(cudaMalloc(&key_for_wb, sizeof(int) * all_ops_count));
    CHECK_CUDA(cudaMalloc(&value_for_wb, sizeof(int) * all_ops_count));
    CHECK_CUDA(cudaMalloc(&key_for_wa, sizeof(int) * all_ops_count));
    CHECK_CUDA(cudaMalloc(&value_for_wa, sizeof(int) * all_ops_count));

    make_key_value_for_wb_and_wa_for_proposal<<<(all_ops_count + 1023) / 1024,
                                                1024>>>(
        sorted_op, key_for_wb, value_for_wb, key_for_wa, value_for_wa,
        all_ops_count);

    cudaDeviceSynchronize();
    // write_beforeのデータを作成する
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSumByKey(d_temp_storage, temp_storage_bytes,
                                       key_for_wb, value_for_wb, write_before,
                                       all_ops_count);
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    cub::DeviceScan::ExclusiveSumByKey(d_temp_storage, temp_storage_bytes,
                                       key_for_wb, value_for_wb, write_before,
                                       all_ops_count);

    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSumByKey(
        d_temp_storage, temp_storage_bytes, key_for_wa, value_for_wa,
        write_after_rev, all_ops_count);
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    cub::DeviceScan::ExclusiveSumByKey(
        d_temp_storage, temp_storage_bytes, key_for_wa, value_for_wa,
        write_after_rev, all_ops_count);

    reverse_array<<<(all_ops_count + 1023) / 1024, 1024>>>(
        write_after_rev, write_after, all_ops_count);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(d_temp_storage);

    times.push_back(
        {"wb/wa creation\t", std::chrono::high_resolution_clock::now()});
    // op_type
    int *op_type;
    CHECK_CUDA(cudaMalloc(&op_type, sizeof(int) * all_ops_count));
    get_op_type_for_proposal<<<(all_ops_count + 1023) / 1024, 1024>>>(
        sorted_op, write_before, write_after, op_type, all_ops_count);
    times.push_back(
        {"op_type creation", std::chrono::high_resolution_clock::now()});
    //      tw_before
    int *tw_before, *tw_before_value;
    CHECK_CUDA(cudaMalloc(&tw_before, sizeof(int) * all_ops_count));
    CHECK_CUDA(cudaMalloc(&tw_before_value, sizeof(int) * all_ops_count));

    make_tw_before_value<<<(all_ops_count + 1023) / 1024, 1024>>>(
        op_type, tw_before_value, all_ops_count);

    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  tw_before_value, tw_before, all_ops_count);
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  tw_before_value, tw_before, all_ops_count);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(d_temp_storage));
    times.push_back(
        {"tw_before creation", std::chrono::high_resolution_clock::now()});
    // rw_loc用のメモリ
    RWLoc *rw_loc;
    CHECK_CUDA(cudaMalloc(&rw_loc, sizeof(RWLoc) * all_ops_count));

    get_rw_location<<<(all_ops_count + 1023) / 1024, 1024>>>(
        op_type, tw_before, all_ops_count, rw_loc);
    CHECK_CUDA(cudaDeviceSynchronize());
    times.push_back(
        {"rw_loc creation\t", std::chrono::high_resolution_clock::now()});
    // 振り分ける
    RWLoc *result;
    CHECK_CUDA(cudaMalloc(&result, sizeof(RWLoc) * all_ops_count));
    scatter_for_proposal<<<(all_ops_count + 1023) / 1024, 1024>>>(
        sorted_op, result, rw_loc, all_ops_count);
    CHECK_CUDA(cudaDeviceSynchronize());
    times.push_back(
        {"scatter creation", std::chrono::high_resolution_clock::now()});

    std::chrono::duration<double, std::milli> all_time =
        (std::chrono::high_resolution_clock::now() - times[0].second);
    // for (int i = 1; i < times.size(); i++) {
    //     std::chrono::duration<double, std::milli> duration =
    //         (times[i].second - times[i - 1].second);
    //     std::cout << times[i].first << "\t: " << duration.count()
    //               << " milliseconds\t "
    //               << (double)(duration.count()) / (double)all_time.count() *
    //               100
    //               << "% of total time" << std::endl;
    // }

    // 実行時間をCSVファイルに保存（1つのファイルに追記）
    static bool first_run = true;
    write_times_to_csv("Proposal", times, "proposal_times.csv", first_run);
    first_run = false;

    // メモリ開放
    // cudaFree(op_id_to_idx);
    cudaFree(sorted_op);
    cudaFree(d_temp_storage);
    cudaFree(write_before);
    cudaFree(write_after_rev);
    cudaFree(write_after);
    cudaFree(key_for_wb);
    cudaFree(value_for_wb);
    cudaFree(key_for_wa);
    cudaFree(value_for_wa);
    cudaFree(op_type);
    cudaFree(tw_before);
    cudaFree(tw_before_value);
    cudaFree(rw_loc);
    cudaFree(result);
    cudaDeviceSynchronize();
}
#endif

#endif