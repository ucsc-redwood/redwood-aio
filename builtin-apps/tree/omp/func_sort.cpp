#include "func_sort.hpp"

#include <omp.h>

#include <cstring>
#include <vector>

// #include "../../../../pipe/affinity.hpp"

namespace tree {

namespace omp {

struct bucket {
  int n_elem;
  int index;  // [start : n_elem)
  int start;  // starting point in B array
};

int cmpfunc(const void *a, const void *b) {
  return (*(uint32_t *)a - *(uint32_t *)b);
}

// std::vector<int> cores = {0, 1, 2, 3, 4, 5, 6, 7};

void bucket_sort(uint32_t *input_array, int dim, int n_buckets) {
  uint32_t *A = input_array;
  uint32_t *B = (uint32_t *)malloc(sizeof(uint32_t) * dim);
  struct bucket *buckets;
  uint32_t limit = 100000;
  uint32_t w = limit / n_buckets;
  int num_threads = n_buckets;
  int workload, b_index;

  // global buckets
  int *global_n_elem = (int *)malloc(sizeof(int) * n_buckets);
  int *global_starting_position = (int *)malloc(sizeof(int) * n_buckets);
  memset(global_n_elem, 0, sizeof(int) * n_buckets);
  memset(global_starting_position, 0, sizeof(int) * n_buckets);

  omp_set_num_threads(num_threads);

  // local buckets, n_buckets for each thread
  buckets =
      (struct bucket *)calloc(n_buckets * num_threads, sizeof(struct bucket));

#pragma omp parallel
  {
    // bind_thread_to_core(cores);

    num_threads = omp_get_num_threads();

    int j, k;
    int local_index;        // [0 : n_buckets)
    int real_bucket_index;  // [0 : n_buckets * num_threads)
    int my_id = omp_get_thread_num();
    workload = dim / num_threads;
    int prevoius_index;

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
        global_starting_position[j] =
            global_starting_position[j - 1] + global_n_elem[j - 1];
        buckets[j].start = buckets[j - 1].start + global_n_elem[j - 1];
        buckets[j].index = buckets[j - 1].index + global_n_elem[j - 1];
      }
    }

#pragma omp barrier
    for (j = my_id + n_buckets; j < n_buckets * num_threads;
         j = j + num_threads) {
      int prevoius_index = j - n_buckets;
      buckets[j].start =
          buckets[prevoius_index].start + buckets[prevoius_index].n_elem;
      buckets[j].index =
          buckets[prevoius_index].index + buckets[prevoius_index].n_elem;
    }
#pragma omp barrier

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
      qsort(B + global_starting_position[i],
            global_n_elem[i],
            sizeof(uint32_t),
            cmpfunc);
  }

  // Copy sorted result back to input array
  memcpy(input_array, B, sizeof(uint32_t) * dim);

  // Clean up
  free(B);
  free(buckets);
  free(global_n_elem);
  free(global_starting_position);
}

}  // namespace omp

}  // namespace tree
