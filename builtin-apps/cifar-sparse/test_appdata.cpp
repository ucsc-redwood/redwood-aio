#include <iostream>

#include "sparse_appdata.hpp"

int main(int argc, char** argv) {
  auto mr = std::pmr::new_delete_resource();

  cifar_sparse::AppData appdata(mr);

  std::cout << "good" << std::endl;

  return 0;
}