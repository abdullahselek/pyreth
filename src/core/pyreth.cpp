#include <iostream>
#include <pyreth/pyreth.hpp>

namespace pyreth {

Pyreth::Pyreth() {
  std::cout << "Pyreth Engine Initialized (v" << PYRETH_VERSION << ")\n";
  std::cout << "ONNX Runtime Version: " << Backend::getRuntimeVersion() << "\n";
  std::cout << "Best Available Backend: "
            << executionProviderToString(Backend::getBestAvailableProvider())
            << "\n";
}

Pyreth::~Pyreth() { std::cout << "Pyreth Engine Destroyed\n"; }

void run() { Pyreth p; }

} // namespace pyreth
