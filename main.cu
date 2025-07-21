#include "common.cuh"
#include "epic.cuh"
#include "proposal.cuh"

void epic(long long transaction_count, int operation_count,
          long long record_count, int seed) {
    long long all_ops_count;
    std::vector<Transaction> h_txs;
    std::vector<Operation> h_ops;
    all_ops_count = create_transactions(h_txs, h_ops, transaction_count,
                                        operation_count, record_count, seed);
    for (int i = 0; i < ROOP_COUNT; i++) {
        std::chrono::high_resolution_clock::time_point start =
            std::chrono::high_resolution_clock::now();
        epic_init(h_txs, h_ops, transaction_count, operation_count,
                  all_ops_count);
        std::chrono::high_resolution_clock::time_point end =
            std::chrono::high_resolution_clock::now();
    }
    h_txs.clear();
    h_ops.clear();
}

void proposal(long long transaction_count, int operation_count,
              long long record_count, int seed) {
    long long all_ops_count;
    std::vector<Transaction> h_txs;
    std::vector<Operation> h_ops;
    std::vector<OperationForProposal> h_sorted_ops;
    std::vector<int> h_op_id_to_idx;
    all_ops_count = create_transactions_for_proposal(
        h_txs, h_sorted_ops, h_ops, transaction_count, operation_count,
        record_count, seed);
    for (int i = 0; i < ROOP_COUNT; i++) {
        std::chrono::high_resolution_clock::time_point start =
            std::chrono::high_resolution_clock::now();
        proposal_init(h_txs, h_sorted_ops, h_ops, transaction_count,
                      operation_count, h_op_id_to_idx, all_ops_count);
        std::chrono::high_resolution_clock::time_point end =
            std::chrono::high_resolution_clock::now();
    }
    h_txs.clear();
    h_ops.clear();
}

int main() {
    warmUp<<<1, 1>>>();
    cudaDeviceSynchronize();

    // // タイムスタンプ付きのディレクトリを作成
    std::time_t t = std::time(nullptr);
    char timestamp[100];
    std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S",
                  std::localtime(&t));

    std::vector<long long> transaction_counts = {1000, 10000};
    std::vector<int> operation_counts = {10, 100, 1000};
    for (int i = 0; i < 2; i++) {
        long long transaction_count = transaction_counts[1];
        int operation_count = operation_counts[i];

        std::cout << "Transaction Count: " << transaction_count
                  << ", Operation Count: " << operation_count << std::endl;

        std::string dir_name = "results_" + std::to_string(transaction_count) +
                               "_" + std::to_string(operation_count) + "_" +
                               timestamp;
        // ディレクトリ作成
        std::string cmd = "mkdir -p " + dir_name;
        system(cmd.c_str());

        // カレントディレクトリを変更
        chdir(dir_name.c_str());

        std::cout << "Saving results to directory: " << dir_name << std::endl;

        epic(transaction_count, operation_count, RECORD_COUNT, SEED);
        // proposal(transaction_count, operation_count, RECORD_COUNT, SEED);

        // 元のディレクトリに戻る
        chdir("..");
    }

    return 0;
}
