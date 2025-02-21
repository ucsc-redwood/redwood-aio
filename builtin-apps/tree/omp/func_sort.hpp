#pragma once

#include <omp.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace tree::omp {

struct bucket {
  int n_elem;
  int index;  // [start : n_elem)
  int start;  // starting point in B array
};

inline int cmpfunc(const void *a, const void *b) { return (*(uint32_t *)a - *(uint32_t *)b); }

inline void bucket_sort(

    uint32_t *A,
    uint32_t *B,  // for temporary storage

    int *global_n_elem,
    int *global_starting_position,

    struct bucket *buckets,

    const int dim,
    const int n_buckets,
    const int num_threads

) {
  //   uint32_t limit = 100000;

  // I got this number from running my program
  uint32_t limit = 1'073'741'600;
  uint32_t w = limit / n_buckets;

  int j, k;
  int local_index;        // [0 : n_buckets)
  int real_bucket_index;  // [0 : n_buckets * num_threads)
  int my_id = omp_get_thread_num();
  // int workload = dim / num_threads;
  // int prevoius_index;

#pragma omp for private(local_index)
  for (int i = 0; i < dim; i++) {
    local_index = A[i] / w;
    if (local_index > n_buckets - 1) local_index = n_buckets - 1;
    real_bucket_index = local_index + my_id * n_buckets;
    buckets[real_bucket_index].n_elem++;
  }

  int local_sum = 0;
  for (j = my_id; j < n_buckets * num_threads; j = j + num_threads) {
    local_sum += buckets[j].n_elem;
  }
  global_n_elem[my_id] = local_sum;

#pragma omp barrier

#pragma omp master
  {
    for (j = 1; j < n_buckets; j++) {
      global_starting_position[j] = global_starting_position[j - 1] + global_n_elem[j - 1];
      buckets[j].start = buckets[j - 1].start + global_n_elem[j - 1];
      buckets[j].index = buckets[j - 1].index + global_n_elem[j - 1];
    }
  }

#pragma omp barrier
  for (j = my_id + n_buckets; j < n_buckets * num_threads; j = j + num_threads) {
    int prevoius_index = j - n_buckets;
    buckets[j].start = buckets[prevoius_index].start + buckets[prevoius_index].n_elem;
    buckets[j].index = buckets[prevoius_index].index + buckets[prevoius_index].n_elem;
  }
#pragma omp barrier

  int b_index;

#pragma omp for private(b_index)
  for (int i = 0; i < dim; i++) {
    j = A[i] / w;
    if (j > n_buckets - 1) j = n_buckets - 1;
    k = j + my_id * n_buckets;
    b_index = buckets[k].index++;
    B[b_index] = A[i];
  }

#pragma omp for
  for (int i = 0; i < n_buckets; i++)
    qsort(B + global_starting_position[i], global_n_elem[i], sizeof(uint32_t), cmpfunc);

  // // I am not going to copy this back. (02/04/2025)
  // #pragma omp master
  // memcpy(A, B, sizeof(uint32_t) * dim);
}

// ----------------------------------------------------------------------------
// Old working version
// ----------------------------------------------------------------------------

// struct TempStorage {
//   explicit TempStorage(const int n_buckets, const int num_threads) {
//     global_n_elem = (int *)malloc(sizeof(int) * n_buckets);
//     global_starting_position = (int *)malloc(sizeof(int) * n_buckets);
//     memset(global_n_elem, 0, sizeof(int) * n_buckets);
//     memset(global_starting_position, 0, sizeof(int) * n_buckets);

//     // local buckets, n_buckets for each thread
//     buckets = (struct bucket *)calloc(n_buckets * num_threads, sizeof(struct bucket));
//   }

//   ~TempStorage() {
//     free(global_n_elem);
//     free(global_starting_position);
//     free(buckets);
//   }

//   int *global_n_elem;
//   int *global_starting_position;
//   struct bucket *buckets;
// };

// ----------------------------------------------------------------------------
// New version
// ----------------------------------------------------------------------------

struct TmpStorage {
 public:
  TmpStorage() = default;
  ~TmpStorage() = default;

  // Disallow copy if needed
  TmpStorage(const TmpStorage &) = delete;
  TmpStorage &operator=(const TmpStorage &) = delete;

  // Allow move semantics
  TmpStorage(TmpStorage &&) = default;
  TmpStorage &operator=(TmpStorage &&) = default;

  // getters
  [[nodiscard]] int *global_n_elem() { return h_global_n_elem.data(); }
  [[nodiscard]] int *global_starting_position() { return h_global_starting_position.data(); }
  [[nodiscard]] bucket *buckets() { return h_buckets.data(); }

  [[nodiscard]] bool is_allocated() const { return m_n_buckets > 0; }

  // Allocate memory for storage
  void allocate(int n_buckets, int num_threads) {
    m_n_buckets = n_buckets;
    m_num_threads = num_threads;

    h_global_n_elem.assign(n_buckets, 0);
    h_global_starting_position.assign(n_buckets, 0);

    // Each thread gets n_buckets, so total = n_buckets * num_threads
    h_buckets.assign(n_buckets * num_threads, bucket{});  // Default-initialize buckets
  }

  // // Reset the contents
  // void reset() {
  //   std::fill(global_n_elem.begin(), global_n_elem.end(), 0);
  //   std::fill(global_starting_position.begin(), global_starting_position.end(), 0);

  //   // Reset all buckets
  //   for (auto &b : buckets) {
  //     b = bucket{};  // Re-initialize each bucket to default state
  //   }
  // }

 private:
  int m_n_buckets = 0;
  int m_num_threads = 0;

  std::vector<int> h_global_n_elem;
  std::vector<int> h_global_starting_position;
  std::vector<bucket> h_buckets;
};

}  // namespace tree::omp
