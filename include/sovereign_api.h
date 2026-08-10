#pragma once

#ifdef _WIN32
  #ifdef SOVEREIGN_EXPORTS
    #define SOV_API __declspec(dllexport)
  #else
    #define SOV_API __declspec(dllimport)
  #endif
#else
  #define SOV_API __attribute__((visibility("default")))
#endif

extern "C" {

// Mootori elutsükkel ja diagnostika
SOV_API void sovereign_engine_init();
SOV_API void sovereign_engine_pulse();
SOV_API void sovereign_engine_shutdown();

// Tulevane liides miniLLM dekoodri / forward pass'i jaoks
// (Siia seome edaspidi kaalud ja sisendid)
SOV_API int sovereign_run_inference_step(const float* input_tokens, int num_tokens);

}