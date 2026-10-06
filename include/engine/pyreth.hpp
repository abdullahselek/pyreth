#include <iostream>

namespace pyreth {
struct Pyreth {
  int n;
  Pyreth() { std::cout << "static constructor\n"; }
  ~Pyreth() { std::cout << "static destructor\n"; }
};

void run();
} // namespace pyreth
