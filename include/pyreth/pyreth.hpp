#pragma once

#include <pyreth/backend.hpp>
#include <pyreth/common.hpp>
#include <pyreth/memory_pool.hpp>
#include <pyreth/result.hpp>
#include <pyreth/thread_pool.hpp>
#include <pyreth/work_stealing_queue.hpp>
#include <pyreth/worker_parker.hpp>

namespace pyreth {

/**
 * @class Pyreth
 * @brief The main entry point and orchestrator for the Pyreth inference engine.
 *
 * This class handles the initialization of the runtime environment, hardware
 * acceleration backends, and the internal thread pool.
 */
class PYRETH_API Pyreth {
public:
  /**
   * @brief Construct a new Pyreth engine instance.
   *
   * Initializes internal resources and pre-allocates memory pools.
   */
  Pyreth();

  /**
   * @brief Destroy the Pyreth engine instance.
   *
   * Gracefully shuts down worker threads and releases memory pools.
   */
  ~Pyreth();
};

/**
 * @brief Convenience function to run a demonstration of the engine.
 */
void PYRETH_API run();

}  // namespace pyreth
