#include "util.h"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

void print_usage()
{
    std::cerr
        << "Usage:\n"
        << "  run_fvecs build --base BASE.fvecs --index OUT.hnsw [--metric l2|cd] [--M 16] [--efc 500] [--threads N]\n"
        << "  run_fvecs bench --query QUERY.fvecs --neighbors GT.ivecs --index INDEX.hnsw\n"
        << "                  [--metric l2|cd] [--k 10] [--repeat 3] [--ef-upper 5000]\n"
        << "                  [--ef-min N --ef-step N] [--dataset NAME]\n"
        << "  run_fvecs offline --base BASE.fvecs --index INDEX.hnsw --dataset NAME [--metric l2|cd]\n"
        << "                    [--k K] [--expected-recall 0.95] [--sampling-size 200] [--ef-upper 5000]\n"
        << "  run_fvecs ada --query QUERY.fvecs --neighbors GT.ivecs --index INDEX.hnsw --dataset NAME\n"
        << "                [--metric l2|cd] [--k 10] [--repeat 3] [--expected-recall 0.95]\n"
        << "  run_fvecs online --base BASE.fvecs --query QUERY.fvecs --neighbors GT.ivecs --index INDEX.hnsw\n"
        << "                   --dataset NAME [--metric l2|cd] [--k 100] [--quantile-step 1e-3]\n";
}

bool require_flag(const std::string &name, const std::string &value)
{
    if (!value.empty())
    {
        return true;
    }
    std::cerr << "Missing required flag: " << name << std::endl;
    print_usage();
    return false;
}

int require_experiments_root()
{
    char *root_path = std::getenv("EXPERIMENTS_ROOT");
    if (root_path == nullptr)
    {
        std::cerr << "Error: EXPERIMENTS_ROOT environment variable is not set." << std::endl;
        std::cerr << "Please set it to the root path of the experiments directory." << std::endl;
        std::cerr << "For example, in bash: export EXPERIMENTS_ROOT=/path/to/experiments" << std::endl;
        return 1;
    }
    std::cout << "EXPERIMENTS_ROOT: " << root_path << std::endl;
    return 0;
}

int main(int argc, char **argv)
{
    if (argc <= 1)
    {
        print_usage();
        return 1;
    }

    const std::string cmd = argv[1];
    if (cmd == "-h" || cmd == "--help" || cmd == "help")
    {
        print_usage();
        return 0;
    }

    std::string base, query, neighbors, index_path, metric = "l2", dataset;
    int M = 16;
    int efc = 500;
    int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency() / 4));
    int k = 10;
    bool k_set = false;
    int repeat = 3;
    int ef_upper = 5000;
    int ef_min = 0;
    int ef_step = 0;
    int sampling_size = 200;
    float quantile_step = 1e-3f;
    float expected_recall = 0.95f;

    try
    {
        for (int i = 2; i < argc; ++i)
        {
            const std::string flag = argv[i];
            auto take = [&](std::string &out)
            {
                if (i + 1 >= argc)
                {
                    throw std::runtime_error("Missing value for " + flag);
                }
                out = argv[++i];
            };
            auto take_int = [&](int &out)
            {
                std::string raw;
                take(raw);
                out = std::stoi(raw);
            };
            auto take_float = [&](float &out)
            {
                std::string raw;
                take(raw);
                out = std::stof(raw);
            };

            if (flag == "--base")
                take(base);
            else if (flag == "--query")
                take(query);
            else if (flag == "--neighbors")
                take(neighbors);
            else if (flag == "--index")
                take(index_path);
            else if (flag == "--metric")
                take(metric);
            else if (flag == "--dataset")
                take(dataset);
            else if (flag == "--M")
                take_int(M);
            else if (flag == "--efc")
                take_int(efc);
            else if (flag == "--threads")
                take_int(threads);
            else if (flag == "--k")
            {
                take_int(k);
                k_set = true;
            }
            else if (flag == "--repeat")
                take_int(repeat);
            else if (flag == "--ef-upper")
                take_int(ef_upper);
            else if (flag == "--ef-min")
                take_int(ef_min);
            else if (flag == "--ef-step")
                take_int(ef_step);
            else if (flag == "--sampling-size")
                take_int(sampling_size);
            else if (flag == "--quantile-step")
                take_float(quantile_step);
            else if (flag == "--expected-recall")
                take_float(expected_recall);
            else
            {
                std::cerr << "Unknown flag: " << flag << std::endl;
                print_usage();
                return 1;
            }
        }

        if (cmd == "build")
        {
            if (!require_flag("--base", base) || !require_flag("--index", index_path))
            {
                return 1;
            }
            build_index_from_fvecs(base, index_path, M, efc, metric, threads);
            return 0;
        }
        if (cmd == "bench")
        {
            if (!require_flag("--query", query) || !require_flag("--neighbors", neighbors) ||
                !require_flag("--index", index_path))
            {
                return 1;
            }
            if (dataset.empty())
            {
                dataset = "hnsw";
            }
            auto loaded = load_index_query_gt(query, neighbors, index_path, metric);
            auto hnsw = std::get<0>(loaded);
            auto query_mat = std::get<1>(loaded);
            auto gt = std::get<2>(loaded);
            auto space = std::get<3>(loaded);
            (void)space;
            if (k > gt->cols())
            {
                throw std::runtime_error("k is larger than ground-truth width");
            }
            std::cout << "Default HNSW search (no Ada-ef), k=" << k << ", repeat=" << repeat << std::endl;
            if (ef_min > 0)
            {
                if (ef_step <= 0)
                {
                    throw std::runtime_error("--ef-min requires --ef-step");
                }
                baseline_search_range(dataset, repeat, *hnsw, *query_mat, *gt, static_cast<size_t>(k),
                                      static_cast<size_t>(ef_min), static_cast<size_t>(ef_upper),
                                      static_cast<size_t>(ef_step));
            }
            else
            {
                baseline_search(dataset, repeat, *hnsw, *query_mat, *gt, static_cast<size_t>(k),
                                static_cast<size_t>(ef_upper));
            }
            return 0;
        }
        if (cmd == "offline")
        {
            if (!require_flag("--base", base) || !require_flag("--index", index_path) ||
                !require_flag("--dataset", dataset))
            {
                return 1;
            }
            if (require_experiments_root() != 0)
            {
                return 1;
            }
            std::vector<int> ks = k_set ? std::vector<int>{k} : std::vector<int>{10, 100};
            Eigen::setNbThreads(std::max(1, threads));
            run_offline_ada(std::filesystem::path(std::getenv("EXPERIMENTS_ROOT")), dataset, base, index_path,
                            metric, ks, expected_recall, quantile_step, sampling_size, ef_upper);
            return 0;
        }
        if (cmd == "ada")
        {
            if (!require_flag("--query", query) || !require_flag("--neighbors", neighbors) ||
                !require_flag("--index", index_path) || !require_flag("--dataset", dataset))
            {
                return 1;
            }
            if (require_experiments_root() != 0)
            {
                return 1;
            }
            auto loaded = load_index_query_gt(query, neighbors, index_path, metric);
            auto hnsw = std::get<0>(loaded);
            auto query_mat = std::get<1>(loaded);
            auto gt = std::get<2>(loaded);
            auto space = std::get<3>(loaded);
            (void)space;
            if (k > gt->cols())
            {
                throw std::runtime_error("k is larger than ground-truth width");
            }
            hnswdis::MatrixXf dummy_data(0, query_mat->cols());
            auto dummy_ptr = std::make_shared<hnswdis::MatrixXf>(dummy_data);
            std::cout << "Ada-ef search, k=" << k << ", expected_recall=" << expected_recall
                      << ", repeat=" << repeat << std::endl;
            run_online_search(std::filesystem::path(std::getenv("EXPERIMENTS_ROOT")), dataset, quantile_step, k,
                              hnsw, query_mat, dummy_ptr, gt, repeat, expected_recall, true);
            return 0;
        }
        if (cmd == "online")
        {
            if (!require_flag("--base", base) || !require_flag("--query", query) ||
                !require_flag("--neighbors", neighbors) || !require_flag("--index", index_path) ||
                !require_flag("--dataset", dataset))
            {
                return 1;
            }
            if (require_experiments_root() != 0)
            {
                return 1;
            }
            const char *root_path = std::getenv("EXPERIMENTS_ROOT");
            std::filesystem::path root(root_path);
            auto tuple = load_index_and_data_fvecs(base, query, neighbors, index_path, metric);
            run_online_search(root, dataset, quantile_step, k,
                              std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple), std::get<3>(tuple));
            return 0;
        }

        std::cerr << "Unknown command: " << cmd << std::endl;
        print_usage();
        return 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
