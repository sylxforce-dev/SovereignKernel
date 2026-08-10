#pragma once
#include "tensor.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <random>
#include <numeric>

// Phase 7 — Generation & Sampling (CPU only)

// Abifunktsioon: arvutab tensori elementide koguarvu shape'i põhjal
inline size_t get_tensor_size(const Tensor& t) {
    size_t s = 1;
    for (int dim : t.shape()) {
        s *= dim;
    }
    return s;
}

// 1. Greedy Search (Argmax) - Võtab alati kõige suurema tõenäosusega tokeni
inline int greedy_sample(const Tensor& logits) {
    if (logits.device() != Device::CPU) {
        throw std::runtime_error("greedy_sample: CPU only for Phase 7");
    }

    const float* data = logits.data();
    size_t size = get_tensor_size(logits);

    if (size == 0) throw std::runtime_error("greedy_sample: empty tensor");

    int best_idx = 0;
    float max_val = data[0];

    for (size_t i = 1; i < size; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            best_idx = static_cast<int>(i);
        }
    }
    return best_idx;
}

// 2. Temperatuuri skaalimine - Muudab jaotuse kas teravamaks (temp < 1) või lamedamaks (temp > 1)
//
// NB: kui kutsud seda ISE ENNE top_k_sample()-i sama logits tensori peal,
// rakendub temperature kaks korda (see funktsioon + top_k_sample'i enda
// sisemine skaleerimine), mis annab vaikimisi liiga terava/lameda jaotuse
// ilma ühegi veateateta. Kasuta KAS seda funktsiooni + greedy_sample OR
// top_k_sample'i enda temperature parameetrit - mitte mõlemat samal tensoril.
inline void apply_temperature(Tensor& logits, float temperature) {
    if (temperature <= 0.0f) throw std::runtime_error("Temperature must be > 0");
    if (temperature == 1.0f) return; // Temperatuur 1 ei muuda midagi

    float* data = logits.data();
    size_t size = get_tensor_size(logits);
    for (size_t i = 0; i < size; ++i) {
        data[i] /= temperature;
    }
}

// 3. Top-K Sampling - Võtab k parimat ja valib nende hulgast tõenäosuste alusel
//
// Rakendab temperature'i SISEMISELT (kopeeritud andmetel) - ära kutsu
// apply_temperature()-t eraldi enne seda samal tensoril, vt hoiatus üleval.
inline int top_k_sample(const Tensor& logits, int k, float temperature, std::mt19937& rng) {
    if (logits.device() != Device::CPU) {
        throw std::runtime_error("top_k_sample: CPU only for Phase 7");
    }
    if (temperature <= 0.0f) {
        // Ühtlustatud apply_temperature()'ga - varem see haru vaikimisi
        // EI skaleerinud kehtetu temperature'i korral, ilma veata (silent
        // no-op). Nüüd viskab sama moodi vea nagu apply_temperature().
        throw std::runtime_error("top_k_sample: temperature must be > 0");
    }

    size_t vocab_size = get_tensor_size(logits);
    if (vocab_size == 0) {
        throw std::runtime_error("top_k_sample: empty tensor");
    }
    if (k <= 0 || k > static_cast<int>(vocab_size)) {
        k = static_cast<int>(vocab_size); // Fallback: kui k on vigane, vaatab kõiki
    }

    // Kopeerime andmed, et me ei rikuks algset logits tensorit
    std::vector<float> scaled_logits(logits.data(), logits.data() + vocab_size);
    if (temperature != 1.0f) {
        for (float& val : scaled_logits) val /= temperature;
    }

    // Loome indeksite massiivi
    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);

    // Osaline sorteerimine (leiame ainult top K)
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
        [&scaled_logits](int i1, int i2) {
            return scaled_logits[i1] > scaled_logits[i2];
        });

    // Arvutame Softmaxi AINULT top K elementide peal, et hoida operatsioon kiire
    float max_logit = scaled_logits[indices[0]];
    std::vector<float> probs(k);
    float sum_probs = 0.0f;

    for (int i = 0; i < k; ++i) {
        probs[i] = std::exp(scaled_logits[indices[i]] - max_logit);
        sum_probs += probs[i];
    }

    // Normaliseerime tõenäosused (peavad kokku andma 1.0)
    for (int i = 0; i < k; ++i) {
        probs[i] /= sum_probs;
    }

    // Diskreetne valik juhuslikkuse baasil
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    int sampled_top_k_idx = dist(rng);

    return indices[sampled_top_k_idx];
}