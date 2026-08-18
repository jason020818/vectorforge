#pragma once

// Placeholder for later multithreaded index build / batch search. Phase 0/1
// FlatIndex is single-threaded by design (ground-truth, not a speed target).

namespace vectorforge::threading {

inline int default_thread_count() {
    return 1;
}

}  // namespace vectorforge::threading
