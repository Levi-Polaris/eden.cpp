#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <cstdio>

struct eden_file;
struct eden_mmap;
struct eden_mlock;

using eden_files  = std::vector<std::unique_ptr<eden_file>>;
using eden_mmaps  = std::vector<std::unique_ptr<eden_mmap>>;
using eden_mlocks = std::vector<std::unique_ptr<eden_mlock>>;

struct eden_file {
    eden_file(const char * fname, const char * mode, bool use_direct_io = false);
    eden_file(FILE * file);
    ~eden_file();

    size_t tell() const;
    size_t size() const;

    int file_id() const; // fileno overload

    void seek(size_t offset, int whence) const;

    void read_raw(void * ptr, size_t len);
    void read_raw_unsafe(void * ptr, size_t len);
    void read_aligned_chunk(void * dest, size_t size);
    uint32_t read_u32();

    void write_raw(const void * ptr, size_t len) const;
    void write_u32(uint32_t val) const;

    size_t read_alignment() const;
    bool has_direct_io() const;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct eden_mmap {
    eden_mmap(const eden_mmap &) = delete;
    eden_mmap(struct eden_file * file, size_t prefetch = (size_t) -1, bool numa = false);
    ~eden_mmap();

    size_t size() const;
    void * addr() const;

    void unmap_fragment(size_t first, size_t last);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct eden_mlock {
    eden_mlock();
    ~eden_mlock();

    void init(void * ptr);
    void grow_to(size_t target_size);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

size_t eden_path_max();
