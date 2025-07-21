#ifndef COMMON_CUH
#define COMMON_CUH

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cub/cub.cuh>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/* CUDA API 呼び出しエラーチェック用マクロ */
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(                                                           \
                stderr,                                                        \
                "CUDA error in file '%s' in line %i: %s (error code: %d).\n",  \
                __FILE__, __LINE__, cudaGetErrorString(err), (int)err);        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* バージョンの定義 */
const int PREV_VER_READ = 0;
const int CURR_VER_READ = 1;
const int CURR_VER_WRITE = 2;
const int TEMP_VER_READ = 3;
const int TEMP_VER_WRITE = 4;

const int PREV_VER = 0;
const int TEMP_VER = 1;
const int CURR_VER = 2;

/* 定数の管理 */
const int ROOP_COUNT = 200;
const long long RECORD_COUNT = 100000000;
const int SEED = 1234;

/* 構造体定義 */
// 読み書きのロケーション
struct RWLoc {
    int version;
    int index;
};

// 提案手法用のオペレーション
struct OperationForProposal {
    int record_id;
    int idx;
    bool type; // 0: Read, 1: Write

    __host__ __device__ OperationForProposal(int id, bool t, int i)
        : record_id(id), idx(i), type(t) {}
};

// 全オペレーションを1つのリストに入れるためのオペレーション構造体
struct OperationForAllOps {
    int record_id;
    int txn_id;
    int op_id;
    int type; // 0: Read, 1: Write
    int txn_start_idx;

    __host__ __device__ OperationForAllOps()
        : record_id(0), txn_id(0), op_id(0), txn_start_idx(0), type(0) {}
    __host__ __device__ OperationForAllOps(int id, int t, int txn,
                                           int txn_start_idx, int op)
        : record_id(id), txn_id(txn), op_id(op), txn_start_idx(txn_start_idx),
          type(t) {}
};

// オペレーション
struct Operation {
    int record_id;
    int type; // 0: Read, 1: Write

    __host__ __device__ Operation() : record_id(0), type(0) {}

    __host__ __device__ Operation(int id, int t) : record_id(id), type(t) {}
};

// トランザクション
struct Transaction {
    int id;
    int op_start_index;
    int op_count;
};

// GPUでトランザクションをプリント（デバッグ用）
__global__ void print_transactions(Transaction *txs, Operation *ops,
                                   int tx_count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= 20) {
        return;
    }
    Transaction tx = txs[idx];
    for (int i = 0; i < tx.op_count; ++i) {
        printf("  Op %d: record_id=%d, type=%d\n", i,
               ops[tx.op_start_index + i].record_id,
               ops[tx.op_start_index + i].type);
    }
}

// GPUで配列をプリント（デバッグ用）
__global__ void print_array(int *arr) {
    printf("Array: ");
    for (int i = 0; i < 20; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n\n");
}

// GPUでRW_Locをプリント（デバッグ用）
__global__ void print_result(RWLoc *result) {
    printf("Result: ");
    for (int i = 0; i < 20; ++i) {
        printf("(%d, %d) ", result[i].version, result[i].index);
    }
    printf("\n\n");
}

// GPUでソートされたオペレーションをプリント（デバッグ用）
__global__ void print_sorted_op(OperationForAllOps *sorted_op) {
    printf("Sorted Operations: ");
    for (int i = 0; i < 20; ++i) {
        printf("(%d, %d, %d, %d) ", sorted_op[i].record_id, sorted_op[i].txn_id,
               sorted_op[i].op_id, sorted_op[i].type);
    }
    printf("\n\n");
}

// CSVファイルに実行時間を書き込むためのヘルパー関数
void write_times_to_csv(
    const std::string &prefix,
    const std::vector<
        std::pair<std::string, std::chrono::high_resolution_clock::time_point>>
        &times,
    const std::string &filename, bool write_header = false) {

    bool file_exists = access(filename.c_str(), F_OK) != -1;

    std::ofstream csvFile;
    if (write_header || !file_exists) {
        // 新規作成またはヘッダー書き込みが必要な場合
        csvFile.open(filename);

        // CSVヘッダー - 各処理ステップをカラムとして設定
        csvFile << "Run,";
        for (int i = 1; i < times.size(); i++) {
            csvFile << times[i].first << ",";
        }
        csvFile << "Total" << std::endl;
    } else {
        // 既存ファイルに追記
        csvFile.open(filename, std::ios::app);
    }

    if (!csvFile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // 実行番号を管理（static変数でファイルごとに独立してカウント）
    static int run_count = 0;

    // 行の始めに実行番号を記録
    csvFile << run_count << ",";

    // 各処理ステップの実行時間を記録
    double total_time = 0.0;
    for (int i = 1; i < times.size(); i++) {
        std::chrono::duration<double, std::milli> duration =
            (times[i].second - times[i - 1].second);
        total_time += duration.count();
        csvFile << duration.count() << ",";
    }

    // 合計時間を最後のカラムに記録
    csvFile << total_time << std::endl;

    run_count++;
    csvFile.close();
}

// トランザクションリストを作成
long long create_transactions(std::vector<Transaction> &h_txs,
                              std::vector<Operation> &h_ops,
                              int transaction_count, int operation_count,
                              int record_count, int seed) {
    std::random_device rd;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> prob_dist(0.0, 1.0);
    long long all_ops_count = 0;
    // トランザクションを作成
    for (int i = 0; i < transaction_count; i++) {
        Transaction tx;
        tx.id = i;
        tx.op_start_index = h_ops.size();
        tx.op_count = operation_count;

        // オペレーションを作成
        for (int j = 0; j < operation_count / 2; j++) {
            int record_id = rand() % record_count;
            h_ops.push_back(Operation(record_id, 1));
            all_ops_count++;
        }
        for (int j = 0; j < operation_count / 2; j++) {
            int record_id = rand() % record_count;
            h_ops.push_back(Operation(record_id, 0));
            all_ops_count++;
        }
        h_txs.push_back(tx);
    }
    return all_ops_count;
}

// 提案手法用のトランザクションリストを作成
long long create_transactions_for_proposal(
    std::vector<Transaction> &h_txs,
    std::vector<OperationForProposal> &h_sorted_ops,
    std::vector<Operation> &h_ops, long long transaction_count,
    int operation_count, long long record_count, int seed) {
    std::random_device rd;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> prob_dist(0.0, 1.0);
    long long all_ops_count = 0;
    std::unordered_map<int, int> idx_to_tid;
    // トランザクションを作成
    for (int i = 0; i < transaction_count; i++) {
        Transaction tx;
        tx.id = i;
        tx.op_start_index = h_ops.size();
        tx.op_count = operation_count;

        // オペレーションを作成
        int op_id = 0;
        for (int j = 0; j < operation_count / 2; j++) {
            int record_id = rand() % record_count;
            h_ops.push_back(Operation(record_id, 0));
            h_sorted_ops.push_back(
                OperationForProposal(record_id, 0, h_ops.size() - 1));
            idx_to_tid[h_ops.size() - 1] = i;
            all_ops_count++;
            op_id++;
        }
        for (int j = 0; j < operation_count / 2; j++) {
            int record_id = rand() % record_count;
            h_ops.push_back(Operation(record_id, 1));
            h_sorted_ops.push_back(
                OperationForProposal(record_id, 1, h_ops.size() - 1));
            idx_to_tid[h_ops.size() - 1] = i;
            all_ops_count++;
            op_id++;
        }
        h_txs.push_back(tx);
    }
    // sort (record_id, txn_id)でソート
    std::sort(h_sorted_ops.begin(), h_sorted_ops.end(),
              [&idx_to_tid](const OperationForProposal &a,
                            const OperationForProposal &b) {
                  return (a.record_id == b.record_id)
                             ? idx_to_tid[a.idx] < idx_to_tid[b.idx]
                             : a.record_id < b.record_id;
              });
    printf("Operation Create Done\n");
    return all_ops_count;
}

__global__ void warmUp() {}

#endif