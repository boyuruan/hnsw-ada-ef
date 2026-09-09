#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

template <typename Scalar>
void write_texmex_vecs(const std::string &path, const Scalar *data, int n, int dim)
{
    if (n <= 0 || dim <= 0)
    {
        throw std::runtime_error("Cannot write empty vecs file: " + path);
    }

    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    const int32_t dim32 = dim;
    for (int i = 0; i < n; ++i)
    {
        out.write(reinterpret_cast<const char *>(&dim32), sizeof(int32_t));
        out.write(reinterpret_cast<const char *>(data + static_cast<std::int64_t>(i) * dim),
                  static_cast<std::size_t>(dim) * sizeof(Scalar));
    }
    if (!out)
    {
        throw std::runtime_error("Failed to write vecs file: " + path);
    }
}

template <typename Scalar, typename MatrixT>
void load_texmex_vecs(const std::string &path, MatrixT &vectors)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error("Failed to open file: " + path);
    }

    in.seekg(0, std::ios::end);
    const std::streamoff file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    if (file_size < static_cast<std::streamoff>(sizeof(int32_t)))
    {
        throw std::runtime_error("Empty or invalid vecs file: " + path);
    }

    int32_t dim = 0;
    in.read(reinterpret_cast<char *>(&dim), sizeof(int32_t));
    if (!in || dim <= 0)
    {
        throw std::runtime_error("Invalid dimension in vecs file: " + path);
    }

    const std::int64_t record_bytes =
        static_cast<std::int64_t>(dim + 1) * static_cast<std::int64_t>(sizeof(int32_t));
    if (file_size % record_bytes != 0)
    {
        throw std::runtime_error("File size is not a multiple of record size: " + path);
    }

    const int n = static_cast<int>(file_size / record_bytes);
    vectors.resize(n, dim);

    in.seekg(0, std::ios::beg);
    std::vector<char> record(static_cast<size_t>(record_bytes));
    for (int i = 0; i < n; ++i)
    {
        in.read(record.data(), record_bytes);
        if (!in)
        {
            throw std::runtime_error("Unexpected EOF while reading: " + path);
        }
        int32_t rec_dim = 0;
        std::memcpy(&rec_dim, record.data(), sizeof(int32_t));
        if (rec_dim != dim)
        {
            throw std::runtime_error("Inconsistent dimension in vecs file: " + path);
        }
        std::memcpy(vectors.row(i).data(), record.data() + sizeof(int32_t),
                    static_cast<size_t>(dim) * sizeof(Scalar));
    }
}
