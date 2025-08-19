/* Runtime Shutdown Header File

Here's the complete `include/lm/runtime/shutdown.hpp` file:

```cpp */
#pragma once

#include <nlohmann/json.hpp>
#include <filesystem>

namespace lm::runtime {

class ShutdownHandler {
public:
    // Serialize state to JSON
    static void save_state(
        const std::filesystem::path& output_path,
        bool include_model_weights = false
    );
    
    // Cleanup hooks
    static void register_cleanup(void (*func)());
    static void execute_cleanup();
};

} // namespace lm::runtime
/*```

This header provides the interface for the framework shutdown system with:

1. **State serialization** to save the current system state to JSON
2. **Cleanup hook registration** for orderly resource release
3. **Ordered cleanup execution** for proper shutdown sequencing

The implementation (in the corresponding `.cpp` file) handles:
- Serialization of tokenizer state and model parameters
- Thread-safe cleanup function management
- Proper resource deallocation order
- Error handling during shutdown process

This shutdown system ensures the framework can:
- Persist state between sessions
- Release resources properly
- Handle graceful termination
- Support state restoration on next startup*/
