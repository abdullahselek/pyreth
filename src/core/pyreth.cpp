#include <iostream>
#include <pyreth/pyreth.hpp>

namespace pyreth {

Pyreth::Pyreth() { std::cout << "Pyreth Engine Initialized\n"; }

Pyreth::~Pyreth() { std::cout << "Pyreth Engine Destroyed\n"; }

void run() { Pyreth p; }

} // namespace pyreth
