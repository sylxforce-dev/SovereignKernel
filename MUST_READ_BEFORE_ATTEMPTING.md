# SovereignKernel — Disclaimer, Constraints & Configuration Notice

This repository contains a custom-built, low-level inference kernel tailored specifically for target architectures. Please review the following constraints before running or testing the code:

## 1. Hardcoded Paths Notice
The runner source files currently contain hardcoded absolute Windows paths pointing to local directories. 
* **Action required:** You must manually update these paths in the source code to point to your local model and tokenizer files before attempting to compile and execute.
* For example, the runner entry points currently define paths such as:

    std::string gguf_path = "C:/Users/DrSulxX/CLionProjects/SovereignKernel/model_data/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
    std::string tokenizer_path = "C:/Users/DrSulxX/CLionProjects/SovereignKernel/model_data/tinyllama_tokenizer.bin";
    std::string config_path = "runtime_config.txt";

* At minimum, configure these paths for your own local environment to match the TinyLlama GGUF file, the tokenizer file, and the Karpathy `.bin` file.

## 2. Strict Model Support
The engine is built specifically for:
* **TinyLlama-1.1B-Chat** (GGUF / Q4_0 format)
* **Andrej Karpathy's `stories110M`** (original `.bin` checkpoint format)

Trying to run arbitrary external models or different architectures is **not recommended and not supported**. The custom kernels, tensor layouts, and dimensions are hardcoded for these exact models.

## 3. CPU Configuration & Hardware Constraints
* **AVX2 Requirement:** AVX2 is a hard compile-time requirement enforced by the build configuration.
* **Core Tuning:** The CPU runtime and OpenMP configuration are tuned around a specific hardware setup (8 physical cores, matching the Ryzen 7 7700 reference system). The runner explicitly sets the thread count to avoid 16-thread SMT synchronization overhead:

    #ifdef _OPENMP
        omp_set_num_threads(8); // Ryzen 7700 füüsiliste tuumade arv - hoiab ära 16-thread sync overhead'i
    #endif

  If testing locally on a CPU with a different core count, adjust this thread configuration in the code accordingly to avoid memory bandwidth contention.