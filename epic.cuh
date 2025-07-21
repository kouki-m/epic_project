#ifndef EPIC_CUH
#define EPIC_CUH

#include "common.cuh"
#include "proposal.cuh"

#ifndef MAKE_ALL_OPS
#define MAKE_ALL_OPS
/* トランザクションセットをオペレーション毎に切り分ける */
__global__ void make_all_ops(OperationForAllOps *all_ops, Transaction *txns,
                             Operation *ops, int transaction_count,
                             int operation_count, int *op_id_to_idx,
                             uint64_t *sort_key) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= transaction_count) {
        return;
    }

    Transaction tx = txns[idx];
    int op_id = 0;
    for (int i = 0; i < tx.op_count; ++i) {
        if (ops[tx.op_start_index + i].type == 0) {
            all_ops[tx.op_start_index + i] =
                OperationForAllOps(ops[tx.op_start_index + i].record_id,
                                   ops[tx.op_start_index + i].type, tx.id,
                                   tx.op_start_index, op_id);
            sort_key[tx.op_start_index + i] =
                (static_cast<uint64_t>(ops[tx.op_start_index + i].record_id)
                 << 32) |
                static_cast<uint32_t>(tx.id);
            op_id_to_idx[tx.op_start_index + op_id] = tx.op_start_index + i;
            op_id++;
        }
    }
    for (int i = 0; i < tx.op_count; ++i) {
        if (ops[tx.op_start_index + i].type == 1) {
            all_ops[tx.op_start_index + i] =
                OperationForAllOps(ops[tx.op_start_index + i].record_id,
                                   ops[tx.op_start_index + i].type, tx.id,
                                   tx.op_start_index, op_id);
            sort_key[tx.op_start_index + i] =
                (static_cast<uint64_t>(ops[tx.op_start_index + i].record_id)
                 << 32) |
                static_cast<uint32_t>(tx.id);
            op_id_to_idx[tx.op_start_index + op_id] = tx.op_start_index + i;
            op_id++;
        }
    }
}
#endif

#ifndef MAKE_KEY_VALUE_FOR_WB_AND_WA
#define MAKE_KEY_VALUE_FOR_WB_AND_WA
/* write before と write afterを計算するためのkeyとvalueの配列を作成 */
__global__ void make_key_value_for_wb_and_wa(OperationForAllOps *sorted_op,
                                             int *key_for_wb, int *value_for_wb,
                                             int *key_for_wa, int *value_for_wa,
                                             long long all_ops_count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= all_ops_count) {
        return;
    }
    OperationForAllOps op = sorted_op[idx];
    key_for_wb[idx] = op.record_id;
    value_for_wb[idx] = op.type;
    key_for_wa[all_ops_count - 1 - idx] = op.record_id;
    value_for_wa[all_ops_count - 1 - idx] = op.type;
}
#endif

#ifndef REVERSE_ARRAY
#define REVERSE_ARRAY
/* 配列を反転 */
__global__ void reverse_array(int *input, int *output,
                              long long all_ops_count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= all_ops_count) {
        return;
    }
    output[all_ops_count - 1 - idx] = input[idx];
}
#endif

#ifndef GET_OP_TYPE
#define GET_OP_TYPE
/* オペレーションタイプを決定 */
__global__ void get_op_type(OperationForAllOps *sorted_op, int *write_before,
                            int *write_after, int *op_type,
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

#ifndef MAKE_TW_BEFORE_VALUE
#define MAKE_TW_BEFORE_VALUE
/* temp writeを01で配列に格納 */
__global__ void make_tw_before_value(int *op_type, int *tw_before_value,
                                     long long all_ops_count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= all_ops_count) {
        return;
    }
    tw_before_value[idx] = (int)(op_type[idx] == TEMP_VER_WRITE);
}
#endif

#ifndef GET_RW_LOCATION
#define GET_RW_LOCATION
/* Read writeロケーションを決定 */
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

#ifndef SCATTER
#define SCATTER
/* 計算結果をトランザクション毎に振り分け */
__global__ void scatter(OperationForAllOps *sorted_op, RWLoc *result,
                        RWLoc *rw_loc, int *op_id_to_idx, int all_ops_count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= all_ops_count)
        return;

    int op_id = sorted_op[idx].op_id;
    int txn_start_idx = sorted_op[idx].txn_start_idx;
    int result_idx = op_id_to_idx[txn_start_idx + op_id];
    result[result_idx].version = rw_loc[idx].version;
    result[result_idx].index = rw_loc[idx].index;
}
#endif

#ifndef EPIC_INIT
#define EPIC_INIT
/* Epicの初期化フェーズ */
void epic_init(std::vector<Transaction> &h_txs, std::vector<Operation> &h_ops,
               int transaction_count, int operation_count,
               long long all_ops_count) {
    std::vector<
        std::pair<std::string, std::chrono::high_resolution_clock::time_point>>
        times = {{"Start", std::chrono::high_resolution_clock::now()}};
    // GPU用のメモリを確保する
    Transaction *d_txs;
    Operation *d_ops;

    CHECK_CUDA(cudaMalloc(&d_txs, sizeof(Transaction) * h_txs.size()));
    CHECK_CUDA(cudaMalloc(&d_ops, sizeof(Operation) * h_ops.size()));

    // データをGPUに転送
    CHECK_CUDA(cudaMemcpy(d_txs, h_txs.data(),
                          sizeof(Transaction) * h_txs.size(),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ops, h_ops.data(), sizeof(Operation) * h_ops.size(),
                          cudaMemcpyHostToDevice));

    times.push_back(
        {"Data transfer to GPU", std::chrono::high_resolution_clock::now()});
    //      all_ops用のメモリ
    OperationForAllOps *all_ops;
    int *op_id_to_idx;
    CHECK_CUDA(
        cudaMalloc(&all_ops, sizeof(OperationForAllOps) * all_ops_count));
    CHECK_CUDA(cudaMalloc(&op_id_to_idx, sizeof(int) * all_ops_count));

    //      sort用のキー
    uint64_t *sort_key, *sort_key_out;
    CHECK_CUDA(cudaMalloc(&sort_key, sizeof(uint64_t) * all_ops_count));
    CHECK_CUDA(cudaMalloc(&sort_key_out, sizeof(uint64_t) * all_ops_count));

    //      all_opsのデータを作成
    make_all_ops<<<(transaction_count + 1023) / 1024, 1024>>>(
        all_ops, d_txs, d_ops, transaction_count, operation_count, op_id_to_idx,
        sort_key);
    CHECK_CUDA(cudaDeviceSynchronize());
    times.push_back(
        {"All ops creation", std::chrono::high_resolution_clock::now()});
    //      sorted_op用のメモリ
    OperationForAllOps *sorted_op;
    CHECK_CUDA(
        cudaMalloc(&sorted_op, sizeof(OperationForAllOps) * all_ops_count));
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    sort_key, sort_key_out, all_ops, sorted_op,
                                    all_ops_count);

    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    CHECK_CUDA(cudaDeviceSynchronize());

    // sortする
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    sort_key, sort_key_out, all_ops, sorted_op,
                                    all_ops_count);

    CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(d_temp_storage);
    times.push_back(
        {"Sorting Done\t", std::chrono::high_resolution_clock::now()});
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

    make_key_value_for_wb_and_wa<<<(all_ops_count + 1023) / 1024, 1024>>>(
        sorted_op, key_for_wb, value_for_wb, key_for_wa, value_for_wa,
        all_ops_count);

    CHECK_CUDA(cudaDeviceSynchronize());
    // write_beforeのデータを作成する
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSumByKey(d_temp_storage, temp_storage_bytes,
                                       key_for_wb, value_for_wb, write_before,
                                       all_ops_count, cub::Equality());
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    cub::DeviceScan::ExclusiveSumByKey(d_temp_storage, temp_storage_bytes,
                                       key_for_wb, value_for_wb, write_before,
                                       all_ops_count, cub::Equality());

    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSumByKey(
        d_temp_storage, temp_storage_bytes, key_for_wa, value_for_wa,
        write_after_rev, all_ops_count, cub::Equality());
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    cub::DeviceScan::ExclusiveSumByKey(
        d_temp_storage, temp_storage_bytes, key_for_wa, value_for_wa,
        write_after_rev, all_ops_count, cub::Equality());

    reverse_array<<<(all_ops_count + 1023) / 1024, 1024>>>(
        write_after_rev, write_after, all_ops_count);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(d_temp_storage);

    times.push_back(
        {"wb/wa creation\t", std::chrono::high_resolution_clock::now()});
    // op_type
    int *op_type;
    CHECK_CUDA(cudaMalloc(&op_type, sizeof(int) * all_ops_count));
    get_op_type<<<(all_ops_count + 1023) / 1024, 1024>>>(
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
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  tw_before_value, tw_before, all_ops_count);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(d_temp_storage);
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
    scatter<<<(all_ops_count + 1023) / 1024, 1024>>>(
        sorted_op, result, rw_loc, op_id_to_idx, all_ops_count);
    CHECK_CUDA(cudaDeviceSynchronize());
    times.push_back(
        {"scatter creation", std::chrono::high_resolution_clock::now()});

    std::chrono::duration<double, std::milli> all_time =
        (std::chrono::high_resolution_clock::now() - times[0].second);

    // 実行時間をCSVファイルに保存（1つのファイルに追記）
    static bool first_run = true;
    write_times_to_csv("Epic", times, "epic_times.csv", first_run);
    first_run = false;

    // メモリ開放
    cudaFree(d_txs);
    cudaFree(d_ops);
    cudaFree(all_ops);
    cudaFree(op_id_to_idx);
    cudaFree(sorted_op);
    cudaFree(sort_key);
    cudaFree(sort_key_out);
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