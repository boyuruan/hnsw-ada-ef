#pragma once

#include <H5Cpp.h>
#include "util.h"

void load_hdf5(const std::string &path,
               hnswdis::MatrixXf &query_vectors,
               hnswdis::MatrixXf &data_vectors,
               hnswdis::MatrixXi &neighbors)
{

    // Open HDF5 file in read-only mode
    H5::H5File file(path, H5F_ACC_RDONLY);

    // --------------------
    // 1. Read "train" dataset
    // --------------------
    {
        H5::DataSet train_dataset = file.openDataSet("train");
        H5::DataSpace train_dataspace = train_dataset.getSpace();

        hsize_t train_dims[2];
        train_dataspace.getSimpleExtentDims(train_dims, nullptr);

        data_vectors.resize(train_dims[0], train_dims[1]);
        // Read from HDF5 (row-major ordering)
        train_dataset.read(data_vectors.data(), H5::PredType::NATIVE_FLOAT);
    }

    // --------------------
    // 2. Read "test" dataset
    // --------------------
    {
        H5::DataSet test_dataset = file.openDataSet("test");
        H5::DataSpace test_dataspace = test_dataset.getSpace();

        hsize_t test_dims[2];
        test_dataspace.getSimpleExtentDims(test_dims, nullptr);

        query_vectors.resize(test_dims[0], test_dims[1]);
        // Read raw data into row-major
        test_dataset.read(query_vectors.data(), H5::PredType::NATIVE_FLOAT);
    }

    // --------------------
    // 3. Read "neighbors" dataset (int)
    // --------------------
    {
        H5::DataSet neighbors_dataset = file.openDataSet("neighbors");
        H5::DataSpace neighbors_dataspace = neighbors_dataset.getSpace();

        hsize_t neighbors_dims[2];
        neighbors_dataspace.getSimpleExtentDims(neighbors_dims, nullptr);

        neighbors.resize(neighbors_dims[0], neighbors_dims[1]);

        // Read raw data into row-major
        neighbors_dataset.read(neighbors.data(), H5::PredType::NATIVE_INT);
    }
}

void load_hdf5(const std::string &path,
               hnswdis::MatrixXf &query_vectors,
               hnswdis::MatrixXf &data_vectors)
{

    // Open HDF5 file in read-only mode
    H5::H5File file(path, H5F_ACC_RDONLY);

    // --------------------
    // 1. Read "train" dataset
    // --------------------
    {
        H5::DataSet train_dataset = file.openDataSet("train");
        H5::DataSpace train_dataspace = train_dataset.getSpace();

        hsize_t train_dims[2];
        train_dataspace.getSimpleExtentDims(train_dims, nullptr);

        data_vectors.resize(train_dims[0], train_dims[1]);
        // Read from HDF5 (row-major ordering)
        train_dataset.read(data_vectors.data(), H5::PredType::NATIVE_FLOAT);
    }

    // --------------------
    // 2. Read "test" dataset
    // --------------------
    {
        H5::DataSet test_dataset = file.openDataSet("test");
        H5::DataSpace test_dataspace = test_dataset.getSpace();

        hsize_t test_dims[2];
        test_dataspace.getSimpleExtentDims(test_dims, nullptr);

        query_vectors.resize(test_dims[0], test_dims[1]);
        // Read raw data into row-major
        test_dataset.read(query_vectors.data(), H5::PredType::NATIVE_FLOAT);
    }
}

void load_hdf5(const std::string &path,
               const std::string &first_name,
               hnswdis::MatrixXf &first_matrix,
               const std::string &second_name,
               hnswdis::MatrixXi &second_matrix)
{

    // Open HDF5 file in read-only mode
    H5::H5File file(path, H5F_ACC_RDONLY);

    // --------------------
    // 1. Read dataset with name "first_name"
    // --------------------
    {
        H5::DataSet first_dataset = file.openDataSet(first_name);
        H5::DataSpace first_dataspace = first_dataset.getSpace();

        hsize_t train_dims[2];
        first_dataspace.getSimpleExtentDims(train_dims, nullptr);

        first_matrix.resize(train_dims[0], train_dims[1]);
        // Read from HDF5 (row-major ordering)
        first_dataset.read(first_matrix.data(), H5::PredType::NATIVE_FLOAT);
    }

    // --------------------
    // 2. Read dataset with name "second_name"
    // --------------------
    {
        H5::DataSet second_dataset = file.openDataSet(second_name);
        H5::DataSpace test_dataspace = second_dataset.getSpace();

        hsize_t test_dims[2];
        test_dataspace.getSimpleExtentDims(test_dims, nullptr);

        second_matrix.resize(test_dims[0], test_dims[1]);
        // Read raw data into row-major
        second_dataset.read(second_matrix.data(), H5::PredType::NATIVE_INT);
    }
}

void load_hdf5(const std::string &path,
               const std::string &first_name,
               hnswdis::MatrixXf &first_matrix)
{

    // Open HDF5 file in read-only mode
    H5::H5File file(path, H5F_ACC_RDONLY);

    // --------------------
    // 1. Read dataset with name "first_name"
    // --------------------
    {
        H5::DataSet first_dataset = file.openDataSet(first_name);
        H5::DataSpace first_dataspace = first_dataset.getSpace();

        hsize_t train_dims[2];
        first_dataspace.getSimpleExtentDims(train_dims, nullptr);

        first_matrix.resize(train_dims[0], train_dims[1]);
        // Read from HDF5 (row-major ordering)
        first_dataset.read(first_matrix.data(), H5::PredType::NATIVE_FLOAT);
    }
}

void save_hdf5(const std::string &path,
               hnswdis::MatrixXf &query_vectors,
               hnswdis::MatrixXf &data_vectors,
               hnswdis::MatrixXi &neighbors)
{
    // Create an HDF5 file
    H5::H5File file(path, H5F_ACC_TRUNC);

    float *test_data = query_vectors.data();
    float *train_data = data_vectors.data();
    int *neighbors_data = neighbors.data();

    hsize_t query_dims[2] = {query_vectors.rows(), query_vectors.cols()};
    H5::DataSpace query_dataspace(2, query_dims);

    hsize_t data_dims[2] = {data_vectors.rows(), data_vectors.cols()};
    H5::DataSpace data_dataspace(2, data_dims);

    hsize_t neighbors_dims[2] = {neighbors.rows(), neighbors.cols()};
    H5::DataSpace neighbors_dataspace(2, neighbors_dims);

    // Create "test" dataset and write data
    H5::DataSet dataset_test = file.createDataSet("test", H5::PredType::NATIVE_FLOAT, query_dataspace);
    dataset_test.write(test_data, H5::PredType::NATIVE_FLOAT);

    // Create "train" dataset and write data
    H5::DataSet dataset_train = file.createDataSet("train", H5::PredType::NATIVE_FLOAT, data_dataspace);
    dataset_train.write(train_data, H5::PredType::NATIVE_FLOAT);

    // Create "neighbors" dataset and write data
    H5::DataSet dataset_neighbors = file.createDataSet("neighbors", H5::PredType::NATIVE_INT, neighbors_dataspace);
    dataset_neighbors.write(neighbors_data, H5::PredType::NATIVE_INT);

    std::cout << "HDF5 file created with Eigen matrices successfully!" << std::endl;
}

void save_hdf5(const std::string &path,
               hnswdis::MatrixXf &query_vectors,
               hnswdis::MatrixXi &neighbors)
{
    // Create an HDF5 file
    H5::H5File file(path, H5F_ACC_TRUNC);

    float *test_data = query_vectors.data();
    int *neighbors_data = neighbors.data();

    hsize_t query_dims[2] = {query_vectors.rows(), query_vectors.cols()};
    H5::DataSpace query_dataspace(2, query_dims);

    hsize_t neighbors_dims[2] = {neighbors.rows(), neighbors.cols()};
    H5::DataSpace neighbors_dataspace(2, neighbors_dims);

    // Create "test" dataset and write data
    H5::DataSet dataset_test = file.createDataSet("test", H5::PredType::NATIVE_FLOAT, query_dataspace);
    dataset_test.write(test_data, H5::PredType::NATIVE_FLOAT);

    // Create "neighbors" dataset and write data
    H5::DataSet dataset_neighbors = file.createDataSet("neighbors", H5::PredType::NATIVE_INT, neighbors_dataspace);
    dataset_neighbors.write(neighbors_data, H5::PredType::NATIVE_INT);

    std::cout << "HDF5 file created with Eigen matrices successfully!" << std::endl;
}

void compute_and_save_gound_truth(const std::string &query_data_path, const std::string &query_data_neighbour_path, const std::string &metric, const int k, bool save = true)
{
    auto query_vectors_ptr = std::make_shared<hnswdis::MatrixXf>();
    auto data_vectors_ptr = std::make_shared<hnswdis::MatrixXf>();

    load_hdf5(query_data_path, *query_vectors_ptr, *data_vectors_ptr);

    if (metric == "cd")
    {
        std::cout << "Normalize the data vectors" << std::endl;
        normalize_matrix(*data_vectors_ptr);
        normalize_matrix(*query_vectors_ptr);
    }

    std::cout << "Data vectors dimensions: " << data_vectors_ptr->rows() << " x " << data_vectors_ptr->cols() << std::endl;
    std::cout << "Query vectors dimensions: " << query_vectors_ptr->rows() << " x " << query_vectors_ptr->cols() << std::endl;

    auto ground_truth = hnswdis::compute_ground_truth(*query_vectors_ptr, *data_vectors_ptr, metric, k);

    if (save)
    {
        std::cout << "Saving ground truth to " << query_data_neighbour_path << std::endl;
        save_hdf5(query_data_neighbour_path, *query_vectors_ptr, *data_vectors_ptr, ground_truth);
    }
}

void build_index(
    const std::string &hdf5_path,
    const std::string &index_path,
    const int M,
    const int ef_construction,
    const std::string &metric,
    const int num_threads)
{
    hnswdis::MatrixXf query_vectors, data_vectors;
    hnswdis::MatrixXi neighbors;
    load_hdf5(hdf5_path, query_vectors, data_vectors, neighbors);
    std::cout << "[Query vectors] rows: " << query_vectors.rows()
              << ", cols: " << query_vectors.cols() << std::endl;

    std::cout << "[Data vectors]  rows: " << data_vectors.rows()
              << ", cols: " << data_vectors.cols() << std::endl;

    std::cout << "[Neighbors] rows: " << neighbors.rows()
              << ", cols: " << neighbors.cols() << std::endl;

    build_index_from_matrix(data_vectors, index_path, M, ef_construction, metric, num_threads);
}

std::tuple<
    std::shared_ptr<hnswlib::HierarchicalNSW<float>>,
    std::shared_ptr<hnswdis::MatrixXf>,
    std::shared_ptr<hnswdis::MatrixXf>,
    std::shared_ptr<hnswdis::MatrixXi>,
    std::shared_ptr<hnswlib::SpaceInterface<float>>>
load_index_and_data(const std::string &hdf5_path, const std::string &index_path, const std::string &metric)
{
    auto query_vectors_ptr = std::make_shared<hnswdis::MatrixXf>();
    auto data_vectors_ptr = std::make_shared<hnswdis::MatrixXf>();
    auto neighbors_ptr = std::make_shared<hnswdis::MatrixXi>();

    // Load the data
    load_hdf5(hdf5_path, *query_vectors_ptr, *data_vectors_ptr, *neighbors_ptr);

    if (metric == "cd")
    {
        std::cout << "Normalize the data vectors" << std::endl;
        normalize_matrix(*data_vectors_ptr);
        normalize_matrix(*query_vectors_ptr);
    }

    std::cout << "Data vectors dimensions: " << data_vectors_ptr->rows() << " x " << data_vectors_ptr->cols() << std::endl;
    std::cout << "Query vectors dimensions: " << query_vectors_ptr->rows() << " x " << query_vectors_ptr->cols() << std::endl;
    std::cout << "Neighbors dimensions: " << neighbors_ptr->rows() << " x " << neighbors_ptr->cols() << std::endl;

    std::shared_ptr<hnswlib::SpaceInterface<float>> space = hnswdis::init_space(metric, query_vectors_ptr->cols());

    std::shared_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw = std::make_shared<hnswlib::HierarchicalNSW<float>>(space.get(), index_path);

    std::cout << "Index loaded" << std::endl;

    std::cout << "Dimension of space:" << *(size_t *)(space->get_dist_func_param()) << std::endl;
    std::cout << "Data size of space:" << space->get_data_size() << std::endl;

    return std::make_tuple(alg_hnsw, query_vectors_ptr, data_vectors_ptr, neighbors_ptr, space);
}

