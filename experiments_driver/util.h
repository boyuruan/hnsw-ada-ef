#pragma once

#include "../hnswlib/adaptive_ef.h"
#include "fvecs_io.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <queue>
#include <stdexcept>
#include <vector>

void load_fvecs(const std::string &path, hnswdis::MatrixXf &vectors)
{
    load_texmex_vecs<float, hnswdis::MatrixXf>(path, vectors);
}

void load_ivecs(const std::string &path, hnswdis::MatrixXi &vectors)
{
    load_texmex_vecs<int, hnswdis::MatrixXi>(path, vectors);
}

void normalize_matrix(hnswdis::MatrixXf &matrix)
{
    for (int i = 0; i < matrix.rows(); ++i)
    {
        float norm = matrix.row(i).norm();
        norm = 1.0f / (norm + 1e-30f);
        matrix.row(i) *= norm;
    }
}

void build_index_from_matrix(
    hnswdis::MatrixXf &data_vectors,
    const std::string &index_path,
    const int M,
    const int ef_construction,
    const std::string &metric,
    const int num_threads)
{
    const int dim = data_vectors.cols();
    const int max_elements = data_vectors.rows();
    if (max_elements <= 0 || dim <= 0)
    {
        throw std::runtime_error("Cannot build index from empty data");
    }

    if (metric == "cd")
    {
        normalize_matrix(data_vectors);
    }

    std::shared_ptr<hnswlib::SpaceInterface<float>> space = hnswdis::init_space(metric, dim);
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw =
        std::make_shared<hnswlib::HierarchicalNSW<float>>(space.get(), max_elements, M, ef_construction);

    auto start = std::chrono::high_resolution_clock::now();
    hnswdis::ParallelFor(0, max_elements, num_threads, [&](size_t row_id, size_t threadId)
                         { alg_hnsw->addPoint((void *)(data_vectors.row(row_id).data()), row_id); });
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Index built in " << duration.count() << " ms" << std::endl;

    const int probe = std::min(1000, max_elements);
    float correct = 0;
    for (int i = 0; i < probe; ++i)
    {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
            alg_hnsw->searchKnn(data_vectors.row(i).data(), 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i)
            correct++;
    }
    std::cout << "First " << probe << " Data Points Recall: " << correct / probe << "\n";

    if (max_elements > probe)
    {
        correct = 0;
        for (int i = max_elements - probe; i < max_elements; ++i)
        {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
                alg_hnsw->searchKnn(data_vectors.row(i).data(), 1);
            hnswlib::labeltype label = result.top().second;
            if (label == i)
                correct++;
        }
        std::cout << "Last " << probe << " Data Points Recall: " << correct / probe << "\n";
    }

    alg_hnsw->saveIndex(index_path);
}

void build_index_from_fvecs(
    const std::string &base_path,
    const std::string &index_path,
    const int M,
    const int ef_construction,
    const std::string &metric,
    const int num_threads)
{
    hnswdis::MatrixXf data_vectors;
    load_fvecs(base_path, data_vectors);
    std::cout << "[Data vectors]  rows: " << data_vectors.rows()
              << ", cols: " << data_vectors.cols() << std::endl;
    build_index_from_matrix(data_vectors, index_path, M, ef_construction, metric, num_threads);
}

std::tuple<
    std::shared_ptr<hnswlib::HierarchicalNSW<float>>,
    std::shared_ptr<hnswdis::MatrixXf>,
    std::shared_ptr<hnswdis::MatrixXf>,
    std::shared_ptr<hnswdis::MatrixXi>,
    std::shared_ptr<hnswlib::SpaceInterface<float>>>
load_index_and_data_fvecs(
    const std::string &base_path,
    const std::string &query_path,
    const std::string &neighbors_path,
    const std::string &index_path,
    const std::string &metric)
{
    auto query_vectors_ptr = std::make_shared<hnswdis::MatrixXf>();
    auto data_vectors_ptr = std::make_shared<hnswdis::MatrixXf>();
    auto neighbors_ptr = std::make_shared<hnswdis::MatrixXi>();

    load_fvecs(base_path, *data_vectors_ptr);
    load_fvecs(query_path, *query_vectors_ptr);
    load_ivecs(neighbors_path, *neighbors_ptr);

    if (query_vectors_ptr->rows() != neighbors_ptr->rows())
    {
        throw std::runtime_error("Query count does not match neighbors count");
    }
    if (query_vectors_ptr->cols() != data_vectors_ptr->cols())
    {
        throw std::runtime_error("Query dimension does not match base dimension");
    }

    if (metric == "cd")
    {
        std::cout << "Normalize the data vectors" << std::endl;
        normalize_matrix(*data_vectors_ptr);
        normalize_matrix(*query_vectors_ptr);
    }

    std::cout << "Data vectors dimensions: " << data_vectors_ptr->rows() << " x " << data_vectors_ptr->cols() << std::endl;
    std::cout << "Query vectors dimensions: " << query_vectors_ptr->rows() << " x " << query_vectors_ptr->cols() << std::endl;
    std::cout << "Neighbors dimensions: " << neighbors_ptr->rows() << " x " << neighbors_ptr->cols() << std::endl;

    std::shared_ptr<hnswlib::SpaceInterface<float>> space =
        hnswdis::init_space(metric, query_vectors_ptr->cols());

    std::shared_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw =
        std::make_shared<hnswlib::HierarchicalNSW<float>>(space.get(), index_path);

    std::cout << "Index loaded" << std::endl;

    std::cout << "Dimension of space:" << *(size_t *)(space->get_dist_func_param()) << std::endl;
    std::cout << "Data size of space:" << space->get_data_size() << std::endl;

    return std::make_tuple(alg_hnsw, query_vectors_ptr, data_vectors_ptr, neighbors_ptr, space);
}

std::tuple<
    std::shared_ptr<hnswlib::HierarchicalNSW<float>>,
    std::shared_ptr<hnswdis::MatrixXf>,
    std::shared_ptr<hnswdis::MatrixXi>,
    std::shared_ptr<hnswlib::SpaceInterface<float>>>
load_index_query_gt(
    const std::string &query_path,
    const std::string &neighbors_path,
    const std::string &index_path,
    const std::string &metric)
{
    auto query_vectors_ptr = std::make_shared<hnswdis::MatrixXf>();
    auto neighbors_ptr = std::make_shared<hnswdis::MatrixXi>();

    load_fvecs(query_path, *query_vectors_ptr);
    load_ivecs(neighbors_path, *neighbors_ptr);

    if (query_vectors_ptr->rows() != neighbors_ptr->rows())
    {
        throw std::runtime_error("Query count does not match neighbors count");
    }
    if (neighbors_ptr->cols() <= 0)
    {
        throw std::runtime_error("Ground truth has no neighbor columns");
    }

    if (metric == "cd")
    {
        std::cout << "Normalize the query vectors" << std::endl;
        normalize_matrix(*query_vectors_ptr);
    }

    std::cout << "Query vectors dimensions: " << query_vectors_ptr->rows() << " x " << query_vectors_ptr->cols() << std::endl;
    std::cout << "Neighbors dimensions: " << neighbors_ptr->rows() << " x " << neighbors_ptr->cols() << std::endl;

    std::shared_ptr<hnswlib::SpaceInterface<float>> space =
        hnswdis::init_space(metric, query_vectors_ptr->cols());
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw =
        std::make_shared<hnswlib::HierarchicalNSW<float>>(space.get(), index_path);

    std::cout << "Index loaded, n=" << alg_hnsw->cur_element_count.load() << std::endl;
    return std::make_tuple(alg_hnsw, query_vectors_ptr, neighbors_ptr, space);
}

void print_score_distribution(const std::vector<float> &score_list)
{
    std::vector<int> score_ranges(10, 0);

    for (const auto &score : score_list)
    {
        int range_index = static_cast<int>(score / 10);
        if (range_index >= 0 && range_index < 10)
        {
            ++score_ranges[range_index];
        }
        else if (range_index == 10)
        {
            ++score_ranges[9];
        }
        else
        {
            std::cerr << "Invalid score: " << score << std::endl;
        }
    }

    for (int i = 0; i < score_ranges.size(); ++i)
    {
        std::cout << "[" << i * 10 << "," << (i + 1) * 10 << "]: " << score_ranges[i] << std::endl;
    }
}

void search_and_score(
    const hnswlib::HierarchicalNSW<float> &alg_hnsw,
    const hnswdis::MatrixXf &query_vectors,
    const hnswdis::MatrixXf &data_vectors,
    const std::string &metric,
    const size_t k,
    const size_t statics_length,
    const float quantile_step)
{

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < query_vectors.rows(); i++)
    {
        auto ret = alg_hnsw.searchKnn(
            query_vectors.row(i).data(), k);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Search time: " << duration.count() << " ms" << std::endl;

    std::shared_ptr<hnswdis::Estimator> estimator = hnswdis::init_estimator(metric, data_vectors);
    hnswdis::ApproximatedScoreCalculator score_cal(estimator, quantile_step);
    std::vector<float> score_list;
    start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < query_vectors.rows(); i++)
    {
        auto ret = alg_hnsw.adaptiveSearchKnn(
            query_vectors.row(i).data(), k, statics_length, score_cal);
        score_list.push_back(ret.second);
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Search time: " << duration.count() << " ms" << std::endl;
    print_score_distribution(score_list);
}

void print_recall_distribution(std::vector<float> &recalls)
{
    std::vector<int> recall_distribution(101, 0);
    for (const auto &recall : recalls)
    {
        int range_index = static_cast<int>(recall * 100);
        if (range_index >= 0 && range_index <= 100)
        {
            ++recall_distribution[range_index];
        }
        else
        {
            std::cerr << "Invalid recall value: " << recall << std::endl;
        }
    }

    for (int i = 0; i < recall_distribution.size(); ++i)
    {
        if (recall_distribution[i] == 0)
            continue;
        std::cout << "[" << i * 0.01 << "," << (i + 1) * 0.01 << "]: " << recall_distribution[i] << std::endl;
    }
}

void adaptive_search(
    const std::string &dataset,
    const int repeat,
    const hnswlib::HierarchicalNSW<float> &alg_hnsw,
    const hnswdis::MatrixXf &query_vectors,
    const hnswdis::MatrixXf &data_vectors,
    const hnswdis::MatrixXi &ground_truth,
    const hnswdis::ApproximatedScoreCalculator &score_cal,
    const size_t k,
    hnswdis::Sketch &sketch,
    const size_t statics_length,
    const float expected_recall)
{
    std::vector<int64_t> time;
    time.reserve(repeat);

    std::tuple<size_t, size_t, float, float, float, int, int, int> exp_record =
        std::make_tuple(statics_length, 0, 0.0f, 0.0f, 0.0f, 0, 0, 0);

    for (int i = 0; i < repeat; ++i)
    {
        std::vector<std::vector<size_t>> result;
        for (int j = 0; j < query_vectors.rows(); ++j)
        {
            result.push_back(std::vector<size_t>(k, 0));
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < query_vectors.rows(); ++j)
        {
            auto pq = alg_hnsw.adaptiveSearchKnnTest(
                query_vectors.row(j).data(), k, statics_length, score_cal, &sketch);
            std::vector<size_t> &labels = result[j];
            {
                size_t count = pq.size();
                while (!pq.empty())
                {
                    labels[--count] = pq.top().second;
                    pq.pop();
                }
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "Search time: " << duration.count() << " ms" << std::endl;

        time.push_back(duration.count());

        auto recalls = hnswdis::compute_recall(ground_truth, result, k, false);
        // std::cout << "Average recall: " << std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size() << std::endl;

        // print_recall_distribution(recalls);

        if (i == repeat - 1)
        {
            // auto recalls = hnswdis::compute_recall(ground_truth, result, false);
            // std::cout << "Average recall: " << std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size() << std::endl;

            auto avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size();
            std::cout << "Average Recall: " << avg_recall << std::endl;

            // print_recall_distribution(recalls);
            std::sort(recalls.begin(), recalls.end());
            size_t index_5 = static_cast<size_t>(recalls.size() * 0.05);
            size_t index_1 = static_cast<size_t>(recalls.size() * 0.01);
            float percentile_5 = recalls[index_5];
            float percentile_1 = recalls[index_1];
            std::cout << "5th percentile recall: " << percentile_5 << std::endl;
            std::cout << "1st percentile recall: " << percentile_1 << std::endl;

            int count_high_recall_99 = 0, count_high_recall_95 = 0, count_high_recall_90 = 0;
            for (const auto &recall : recalls)
            {
                if (recall >= 0.99)
                    count_high_recall_99++;
                if (recall >= 0.95)
                    count_high_recall_95++;
                if (recall >= 0.90)
                    count_high_recall_90++;
            }

            int num_queries = recalls.size();

            std::get<2>(exp_record) = avg_recall;
            std::get<3>(exp_record) = percentile_5;
            std::get<4>(exp_record) = percentile_1;

            std::get<5>(exp_record) = count_high_recall_99;
            std::get<6>(exp_record) = count_high_recall_95;
            std::get<7>(exp_record) = count_high_recall_90;
        }
    }
    std::sort(time.begin(), time.end());
    int64_t median_time = time[time.size() / 2];
    std::cout << "Median search time: " << median_time << " ms" << std::endl;
    std::cout << "Search times: ";
    for (const auto &t : time)
    {
        std::cout << t << " ms, ";
    }
    std::cout << std::endl;

    std::get<1>(exp_record) = median_time;

    std::cout << dataset << " experiment results:" << std::endl;
    std::cout << "statisc_length, time, avg_recall, 5th_percentile_recall, 1st_percentile_recall, recall_above_99, recall_above_95, recall_above_90" << std::endl;

    std::cout << std::get<0>(exp_record) << ", "
              << std::get<1>(exp_record) << ", "
              << std::get<2>(exp_record) << ", "
              << std::get<3>(exp_record) << ", "
              << std::get<4>(exp_record) << ", "
              << std::get<5>(exp_record) << ", "
              << std::get<6>(exp_record) << ", "
              << std::get<7>(exp_record) << std::endl;

    std::cout << "Experiment finished" << std::endl;
}

void adaptive_search_per_query_result(
    const std::string &dataset,
    const hnswlib::HierarchicalNSW<float> &alg_hnsw,
    const hnswdis::MatrixXf &query_vectors,
    const hnswdis::MatrixXf &data_vectors,
    const hnswdis::MatrixXi &ground_truth,
    const hnswdis::ApproximatedScoreCalculator &score_cal,
    const size_t k,
    hnswdis::Sketch &sketch,
    const size_t statics_length,
    const float expected_recall)
{
    const int num_queries = query_vectors.rows();
    std::vector<std::vector<size_t>> result(num_queries, std::vector<size_t>(k, 0));

    std::vector<int64_t> latencies_ns(num_queries);
    std::vector<float> recalls(num_queries);

    for (int j = 0; j < num_queries; ++j)
    {
        auto start = std::chrono::high_resolution_clock::now();

        auto pq = alg_hnsw.adaptiveSearchKnnTest(
            query_vectors.row(j).data(), k, statics_length, score_cal, &sketch);

        auto end = std::chrono::high_resolution_clock::now();
        auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        latencies_ns[j] = latency_ns;

        // Extract top-k results
        size_t count = pq.size();
        while (!pq.empty())
        {
            result[j][--count] = pq.top().second;
            pq.pop();
        }

        // Compute recall for this query
        int correct = 0;
        for (size_t id : result[j])
        {
            for (int gt_idx = 0; gt_idx < k; ++gt_idx)
            {
                if (id == ground_truth(j, gt_idx))
                {
                    correct++;
                    break;
                }
            }
        }
        recalls[j] = static_cast<float>(correct) / k;
    }

    // === Summary statistics ===
    double total_latency_seconds = std::accumulate(latencies_ns.begin(), latencies_ns.end(), 0.0) / 1e9;
    double avg_latency = std::accumulate(latencies_ns.begin(), latencies_ns.end(), 0.0) / num_queries;
    double avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / num_queries;

    // Output per-query results to a CSV file
    std::string csv_filename = "per_query_results_" + dataset + ".csv";
    std::ofstream csv_file(csv_filename);
    if (csv_file.is_open()) {
        csv_file << "QueryID,Latency(ns),Recall\n"; // Write the header
        for (int j = 0; j < num_queries; ++j) {
            csv_file << j << "," << latencies_ns[j] << "," << recalls[j] << "\n"; // Write each query's results
        }
        csv_file.close();
        std::cout << "Per-query results have been written to " << csv_filename << std::endl;
    } else {
        std::cerr << "Error: Unable to open file for writing." << std::endl;
    }

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Average Latency: " << avg_latency << " ns" << std::endl;
    std::cout << "Average Recall: " << avg_recall << std::endl;
    std::cout << "Total Latency: " << total_latency_seconds << " seconds" << std::endl;

    std::vector<int64_t> sorted_latencies = latencies_ns;
    std::sort(sorted_latencies.begin(), sorted_latencies.end());
    std::cout << "95th percentile latency: " << sorted_latencies[(int)(num_queries * 0.95)] << " ns" << std::endl;
    std::cout << "99th percentile latency: " << sorted_latencies[(int)(num_queries * 0.99)] << " ns" << std::endl;

    std::cout << "Experiment finished." << std::endl;
}


void adaptive_ef_analysis(
    const std::string &dataset,
    const hnswlib::HierarchicalNSW<float> &alg_hnsw,
    const hnswdis::MatrixXf &query_vectors,
    const hnswdis::ApproximatedScoreCalculator &score_cal,
    const size_t k,
    hnswdis::Sketch &sketch,
    const size_t statics_length)
{
    std::cout << "Adaptive EF Analysis for dataset: " << dataset << std::endl;
    for (int j = 0; j < query_vectors.rows(); ++j)
    {
        alg_hnsw.adaptiveSearchKnn(
            query_vectors.row(j).data(), k, statics_length, score_cal, &sketch);
    }
    std::cout << std::endl;
}

void baseline_search(
    const std::string &dataset,
    const int repeat,
    hnswlib::HierarchicalNSW<float> &hnsw,
    const hnswdis::MatrixXf &query_vectors,
    const hnswdis::MatrixXi &ground_truth,
    const size_t k,
    const size_t ef_upper_bound)
{
    size_t ef = k;
    float avg_recall = 0.0f;
    // scheme for storing the results
    // ef, time, avg_recall, 5th percentile recall, 1st percentile recall, recall_above_99, recall_above_95, recall_above_90
    std::vector<std::tuple<size_t, size_t, float, float, float, int, int, int>> exp_results;
    while (exp_results.size() < 3 || avg_recall < 0.99)
    {
        std::tuple<size_t, size_t, float, float, float, int, int, int> exp_record =
            std::make_tuple(ef, 0, 0.0f, 0.0f, 0.0f, 0, 0, 0);

        std::cout << "ef: " << ef << std::endl;
        hnsw.setEf(ef);

        std::vector<int64_t> hnsw_search_time;
        hnsw_search_time.reserve(repeat);
        for (int i = 0; i < repeat; ++i)
        {
            std::vector<std::vector<size_t>> result;
            result.reserve(query_vectors.rows());
            auto start_time = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < query_vectors.rows(); i++)
            {
                auto ret = hnsw.searchKnn(
                    query_vectors.row(i).data(), k); // transforming to clser first

                size_t count = ret.size();
                std::vector<size_t> labels(count);
                while (!ret.empty())
                {
                    labels[--count] = ret.top().second;
                    ret.pop();
                }

                result.push_back(std::move(labels));
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            hnsw_search_time.push_back(duration.count());

            if (i == repeat - 1)
            {
                auto recalls = hnswdis::compute_recall(ground_truth, result, k, false);
                avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size();
                std::cout << "Average Recall: " << avg_recall << std::endl;

                // print_recall_distribution(recalls);
                std::sort(recalls.begin(), recalls.end());
                size_t index_5 = static_cast<size_t>(recalls.size() * 0.05);
                size_t index_1 = static_cast<size_t>(recalls.size() * 0.01);
                float percentile_5 = recalls[index_5];
                float percentile_1 = recalls[index_1];
                std::cout << "5th percentile recall: " << percentile_5 << std::endl;
                std::cout << "1st percentile recall: " << percentile_1 << std::endl;

                int count_high_recall_99 = 0, count_high_recall_95 = 0, count_high_recall_90 = 0;
                for (const auto &recall : recalls)
                {
                    if (recall >= 0.99)
                        count_high_recall_99++;
                    if (recall >= 0.95)
                        count_high_recall_95++;
                    if (recall >= 0.90)
                        count_high_recall_90++;
                }

                int num_queries = recalls.size();

                std::get<0>(exp_record) = ef;

                std::get<2>(exp_record) = avg_recall;
                std::get<3>(exp_record) = percentile_5;
                std::get<4>(exp_record) = percentile_1;

                std::get<5>(exp_record) = count_high_recall_99;
                std::get<6>(exp_record) = count_high_recall_95;
                std::get<7>(exp_record) = count_high_recall_90;
            }
        }

        std::sort(hnsw_search_time.begin(), hnsw_search_time.end());
        int64_t median_time = hnsw_search_time[hnsw_search_time.size() / 2];
        std::cout << "Median search time: " << median_time << " ms" << std::endl;

        std::get<1>(exp_record) = median_time;
        exp_results.push_back(exp_record);

        std::cout << "Search times: ";
        for (const auto &t : hnsw_search_time)
        {
            std::cout << t << " ms, ";
        }
        std::cout << std::endl;

        if (ef > ef_upper_bound)
        {
            break;
        }

        if (ef >= 1600)
        {
            ef += 400;
        }
        else
        {
            ef *= 2;
        }
    }

    std::cout << dataset << " experiment results:" << std::endl;
    std::cout << "ef, time, avg_recall, 5th_percentile_recall, 1st_percentile_recall, recall_above_99, recall_above_95, recall_above_90" << std::endl;
    for (const auto &result : exp_results)
    {
        std::cout << std::get<0>(result) << ", "
                  << std::get<1>(result) << ", "
                  << std::get<2>(result) << ", "
                  << std::get<3>(result) << ", "
                  << std::get<4>(result) << ", "
                  << std::get<5>(result) << ", "
                  << std::get<6>(result) << ", "
                  << std::get<7>(result) << std::endl;
    }
    std::cout << "Experiment finished" << std::endl;
}

void baseline_search_range(
    const std::string &dataset,
    const int repeat,
    hnswlib::HierarchicalNSW<float> &hnsw,
    const hnswdis::MatrixXf &query_vectors,
    const hnswdis::MatrixXi &ground_truth,
    const size_t k,
    const size_t ef_min,
    const size_t ef_max,
    const size_t ef_step)
{
    if (ef_min == 0 || ef_step == 0 || ef_min > ef_max)
    {
        throw std::runtime_error("invalid ef range: need 0 < ef-min <= ef-max and ef-step > 0");
    }

    std::vector<std::tuple<size_t, size_t, float, float, float, int, int, int>> exp_results;
    for (size_t ef = ef_min; ef <= ef_max; ef += ef_step)
    {
        std::tuple<size_t, size_t, float, float, float, int, int, int> exp_record =
            std::make_tuple(ef, 0, 0.0f, 0.0f, 0.0f, 0, 0, 0);

        std::cout << "ef: " << ef << std::endl;
        hnsw.setEf(ef);

        std::vector<int64_t> hnsw_search_time;
        hnsw_search_time.reserve(repeat);
        std::vector<float> recalls;
        for (int i = 0; i < repeat; ++i)
        {
            std::vector<std::vector<size_t>> result;
            result.reserve(query_vectors.rows());
            auto start_time = std::chrono::high_resolution_clock::now();
            for (int q = 0; q < query_vectors.rows(); q++)
            {
                auto ret = hnsw.searchKnn(query_vectors.row(q).data(), k);
                size_t count = ret.size();
                std::vector<size_t> labels(count);
                while (!ret.empty())
                {
                    labels[--count] = ret.top().second;
                    ret.pop();
                }
                result.push_back(std::move(labels));
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            hnsw_search_time.push_back(duration.count());

            if (i == repeat - 1)
            {
                recalls = hnswdis::compute_recall(ground_truth, result, k, false);
            }
        }

        float avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size();
        std::cout << "Average Recall: " << avg_recall << std::endl;
        std::sort(recalls.begin(), recalls.end());
        size_t index_5 = static_cast<size_t>(recalls.size() * 0.05);
        size_t index_1 = static_cast<size_t>(recalls.size() * 0.01);
        float percentile_5 = recalls[index_5];
        float percentile_1 = recalls[index_1];
        std::cout << "5th percentile recall: " << percentile_5 << std::endl;
        std::cout << "1st percentile recall: " << percentile_1 << std::endl;

        int count_high_recall_99 = 0, count_high_recall_95 = 0, count_high_recall_90 = 0;
        for (const auto &recall : recalls)
        {
            if (recall >= 0.99)
                count_high_recall_99++;
            if (recall >= 0.95)
                count_high_recall_95++;
            if (recall >= 0.90)
                count_high_recall_90++;
        }

        std::sort(hnsw_search_time.begin(), hnsw_search_time.end());
        int64_t median_time = hnsw_search_time[hnsw_search_time.size() / 2];
        std::cout << "Median search time: " << median_time << " ms" << std::endl;
        std::cout << "Search times: ";
        for (const auto &t : hnsw_search_time)
        {
            std::cout << t << " ms, ";
        }
        std::cout << std::endl;

        std::get<0>(exp_record) = ef;
        std::get<1>(exp_record) = median_time;
        std::get<2>(exp_record) = avg_recall;
        std::get<3>(exp_record) = percentile_5;
        std::get<4>(exp_record) = percentile_1;
        std::get<5>(exp_record) = count_high_recall_99;
        std::get<6>(exp_record) = count_high_recall_95;
        std::get<7>(exp_record) = count_high_recall_90;
        exp_results.push_back(exp_record);
    }

    std::cout << dataset << " experiment results:" << std::endl;
    std::cout << "ef, time, avg_recall, 5th_percentile_recall, 1st_percentile_recall, recall_above_99, recall_above_95, recall_above_90" << std::endl;
    for (const auto &result : exp_results)
    {
        std::cout << std::get<0>(result) << ", "
                  << std::get<1>(result) << ", "
                  << std::get<2>(result) << ", "
                  << std::get<3>(result) << ", "
                  << std::get<4>(result) << ", "
                  << std::get<5>(result) << ", "
                  << std::get<6>(result) << ", "
                  << std::get<7>(result) << std::endl;
    }
    std::cout << "Experiment finished" << std::endl;
}

void search_with_patience_in_proximity(
    const std::string &dataset,
    const int repeat,
    hnswlib::HierarchicalNSW<float> &hnsw,
    const hnswdis::MatrixXf &query_vectors,
    const hnswdis::MatrixXi &ground_truth,
    const size_t k)
{

    std::cout << "Search with Patience in Proximity for dataset: " << dataset << std::endl;

    size_t ef = 3 * k;
    if (k >= 1000)
    {
        ef = 2 * k;
    }

    float avg_recall = 0.0f;
    // scheme for storing the results
    // ef, time, avg_recall, 5th percentile recall, 1st percentile recall, recall_above_99, recall_above_95, recall_above_90
    std::tuple<size_t, size_t, float, float, float, int, int, int> exp_record =
        std::make_tuple(ef, 0, 0.0f, 0.0f, 0.0f, 0, 0, 0);

    std::cout << "ef: " << ef << std::endl;
    hnsw.setEf(ef);

    std::vector<int64_t> hnsw_search_time;
    hnsw_search_time.reserve(repeat);
    for (int i = 0; i < repeat; ++i)
    {
        std::vector<std::vector<size_t>> result;
        result.reserve(query_vectors.rows());
        auto start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < query_vectors.rows(); i++)
        {
            auto ret = hnsw.searchKnnWithPatienceInProximity(
                query_vectors.row(i).data(), k); // transforming to clser first

            size_t count = ret.size();
            std::vector<size_t> labels(count);
            while (!ret.empty())
            {
                labels[--count] = ret.top().second;
                ret.pop();
            }

            result.push_back(std::move(labels));
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        hnsw_search_time.push_back(duration.count());

        if (i == repeat - 1)
        {
            auto recalls = hnswdis::compute_recall(ground_truth, result, k, false);
            avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size();
            std::cout << "Average Recall: " << avg_recall << std::endl;

            // print_recall_distribution(recalls);
            std::sort(recalls.begin(), recalls.end());
            size_t index_5 = static_cast<size_t>(recalls.size() * 0.05);
            size_t index_1 = static_cast<size_t>(recalls.size() * 0.01);
            float percentile_5 = recalls[index_5];
            float percentile_1 = recalls[index_1];
            std::cout << "5th percentile recall: " << percentile_5 << std::endl;
            std::cout << "1st percentile recall: " << percentile_1 << std::endl;

            int count_high_recall_99 = 0, count_high_recall_95 = 0, count_high_recall_90 = 0;
            for (const auto &recall : recalls)
            {
                if (recall >= 0.99)
                    count_high_recall_99++;
                if (recall >= 0.95)
                    count_high_recall_95++;
                if (recall >= 0.90)
                    count_high_recall_90++;
            }

            int num_queries = recalls.size();

            std::get<0>(exp_record) = ef;

            std::get<2>(exp_record) = avg_recall;
            std::get<3>(exp_record) = percentile_5;
            std::get<4>(exp_record) = percentile_1;

            std::get<5>(exp_record) = count_high_recall_99;
            std::get<6>(exp_record) = count_high_recall_95;
            std::get<7>(exp_record) = count_high_recall_90;
        }
    }

    std::sort(hnsw_search_time.begin(), hnsw_search_time.end());
    int64_t median_time = hnsw_search_time[hnsw_search_time.size() / 2];
    std::cout << "Median search time: " << median_time << " ms" << std::endl;

    std::get<1>(exp_record) = median_time;

    std::cout << "Search times: ";
    for (const auto &t : hnsw_search_time)
    {
        std::cout << t << " ms, ";
    }
    std::cout << std::endl;

    std::cout << dataset << " experiment results:" << std::endl;
    std::cout << "ef, time, avg_recall, 5th_percentile_recall, 1st_percentile_recall, recall_above_99, recall_above_95, recall_above_90" << std::endl;

    std::cout << std::get<0>(exp_record) << ", "
              << std::get<1>(exp_record) << ", "
              << std::get<2>(exp_record) << ", "
              << std::get<3>(exp_record) << ", "
              << std::get<4>(exp_record) << ", "
              << std::get<5>(exp_record) << ", "
              << std::get<6>(exp_record) << ", "
              << std::get<7>(exp_record) << std::endl;

    std::cout << "Experiment finished" << std::endl;
}
void run_online_search(
    const std::filesystem::path &root,
    const std::string &dataset,
    float quantile_step,
    int k,
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> hnsw,
    std::shared_ptr<hnswdis::MatrixXf> query,
    std::shared_ptr<hnswdis::MatrixXf> data,
    std::shared_ptr<hnswdis::MatrixXi> ground_truth,
    int repeat = 1,
    float expected_recall = 0.95f,
    bool ada_only = false)
{
    std::string ef_adaptor_path = (root / "estimation_table" / (dataset + "-ef_adaptor-" + "-k" + std::to_string(k) + "-ef.bin")).string();
    std::string estimator_path = (root / "statistics" / (dataset + "-estimator-" + "-k-" + std::to_string(k) + ".bin")).string();

    if (dataset == "laion_text")
    {
        estimator_path = (root / "statistics" / ("laion_image-estimator--k-" + std::to_string(k) + ".bin")).string();
    }

    if (!std::filesystem::exists(estimator_path))
    {
        throw std::runtime_error("Missing estimator file: " + estimator_path);
    }
    if (!std::filesystem::exists(ef_adaptor_path))
    {
        throw std::runtime_error("Missing ef adaptor file: " + ef_adaptor_path);
    }

    std::shared_ptr<hnswdis::Estimator> estimator;
    estimator = hnswdis::load_estimator_from_file(estimator_path);
    hnswdis::ApproximatedScoreCalculator score_cal(estimator, quantile_step);

    std::shared_ptr<hnswdis::EfAdapter> ef_adapter_ptr;
    hnswdis::EfAdapter ef_adapter(ef_adaptor_path);
    ef_adapter_ptr = std::make_shared<hnswdis::EfAdapter>(ef_adapter);

    hnswdis::Sketch sketch(
        ef_adapter_ptr->get_ef_recall_estimators(),
        expected_recall);
    const float wae = ef_adapter_ptr->get_wae();
    std::cout << "****Weighted average ef: " << (size_t)wae << std::endl;
    size_t statics_length = 1 + 32 + 31 * 32; // 2-hop neighbors on the base layer: M = 16
    hnsw->setEf(wae);
    adaptive_search(dataset, repeat, *hnsw, *query, *data, *ground_truth, score_cal, k, sketch, statics_length, expected_recall);
    if (ada_only)
    {
        return;
    }
    adaptive_ef_analysis(dataset, *hnsw, *query, score_cal, k, sketch, statics_length);

    search_with_patience_in_proximity(dataset, repeat, *hnsw, *query, *ground_truth, k);
    baseline_search(dataset, repeat, *hnsw, *query, *ground_truth, k, 5000);
}

void run_offline_ada(
    const std::filesystem::path &root,
    const std::string &dataset,
    const std::string &base_path,
    const std::string &index_path,
    const std::string &metric,
    const std::vector<int> &ks,
    float expected_recall,
    float quantile_step,
    int sampling_size,
    int ef_upper_bound)
{
    std::filesystem::create_directories(root / "estimation_table");
    std::filesystem::create_directories(root / "sampling");
    std::filesystem::create_directories(root / "statistics");

    hnswdis::MatrixXf data_vectors;
    load_fvecs(base_path, data_vectors);
    if (metric == "cd")
    {
        std::cout << "Normalize the data vectors" << std::endl;
        normalize_matrix(data_vectors);
    }
    auto data = std::make_shared<hnswdis::MatrixXf>(std::move(data_vectors));

    auto space = hnswdis::init_space(metric, data->cols());
    auto hnsw = std::make_shared<hnswlib::HierarchicalNSW<float>>(space.get(), index_path);
    std::cout << "Index loaded, n=" << hnsw->cur_element_count.load() << std::endl;

    size_t statics_length = 1 + 32 + 31 * 32;
    int max_k = *std::max_element(ks.begin(), ks.end());

    auto estimator = hnswdis::init_estimator(metric, *data);
    for (int k : ks)
    {
        auto estimator_path = (root / "statistics" / (dataset + "-estimator-" + "-k-" + std::to_string(k) + ".bin")).string();
        hnswdis::save_estimator_to_file(*estimator, estimator_path);
        std::cout << "Wrote estimator " << estimator_path << std::endl;
    }

    auto sampling_pair = hnswdis::compute_samplings(data, metric, max_k, static_cast<size_t>(sampling_size));
    for (int k : ks)
    {
        auto samplings_path = (root / "sampling" / (dataset + "-samplings-" + "-k" + std::to_string(k) + "-ef.bin")).string();
        hnswdis::MatrixXi gt_k = sampling_pair.second.leftCols(k);
        hnswdis::serialize_samplings(samplings_path, sampling_pair.first, gt_k);
        std::cout << "Wrote samplings " << samplings_path << std::endl;
    }

    for (int k : ks)
    {
        auto samplings_path = (root / "sampling" / (dataset + "-samplings-" + "-k" + std::to_string(k) + "-ef.bin")).string();
        auto estimator_path = (root / "statistics" / (dataset + "-estimator-" + "-k-" + std::to_string(k) + ".bin")).string();
        auto ef_adaptor_path = (root / "estimation_table" / (dataset + "-ef_adaptor-" + "-k" + std::to_string(k) + "-ef.bin")).string();
        std::cout << "Computing ef adaptor for k=" << k << std::endl;
        hnswdis::EfAdapter ef_adapter(hnsw, data, static_cast<size_t>(k), metric, expected_recall, quantile_step,
                                      statics_length, samplings_path, estimator_path, ef_upper_bound);
        ef_adapter.serialize(ef_adaptor_path);
        std::cout << "Wrote ef adaptor " << ef_adaptor_path << " wae=" << ef_adapter.get_wae() << std::endl;
    }
}

