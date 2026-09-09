#include "experiments_driver/fvecs_io.h"
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

struct RowMajorMatrix
{
    int rows_ = 0;
    int cols_ = 0;
    std::vector<float> data_;

    void resize(int rows, int cols)
    {
        rows_ = rows;
        cols_ = cols;
        data_.assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0.0f);
    }

    struct Row
    {
        float *ptr;
        float *data() { return ptr; }
    };

    Row row(int i) { return Row{data_.data() + static_cast<size_t>(i) * static_cast<size_t>(cols_)}; }
};

int main()
{
    const auto dir = std::filesystem::temp_directory_path() / "hnsw-ada-fvecs-smoke";
    std::filesystem::create_directories(dir);
    const auto path = (dir / "tiny.fvecs").string();

    const int n = 3;
    const int dim = 4;
    const float src[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f};

    write_texmex_vecs<float>(path, src, n, dim);

    RowMajorMatrix loaded;
    load_texmex_vecs<float>(path, loaded);
    if (loaded.rows_ != n || loaded.cols_ != dim)
    {
        std::cerr << "shape mismatch: " << loaded.rows_ << "x" << loaded.cols_ << std::endl;
        return 1;
    }
    for (int i = 0; i < n * dim; ++i)
    {
        if (std::fabs(loaded.data_[i] - src[i]) > 1e-6f)
        {
            std::cerr << "value mismatch at " << i << std::endl;
            return 1;
        }
    }

    const auto bad = (dir / "bad.fvecs").string();
    {
        std::ofstream out(bad, std::ios::binary);
        int32_t d0 = 2, d1 = 3;
        float a[2] = {1.0f, 2.0f};
        float b[3] = {3.0f, 4.0f, 5.0f};
        out.write(reinterpret_cast<char *>(&d0), 4);
        out.write(reinterpret_cast<char *>(a), 8);
        out.write(reinterpret_cast<char *>(&d1), 4);
        out.write(reinterpret_cast<char *>(b), 12);
    }
    try
    {
        RowMajorMatrix ignored;
        load_texmex_vecs<float>(bad, ignored);
        std::cerr << "expected inconsistent-dimension error" << std::endl;
        return 1;
    }
    catch (const std::runtime_error &)
    {
    }

    std::cout << "fvecs_io_smoke ok n=" << n << " dim=" << dim << std::endl;
    return 0;
}
