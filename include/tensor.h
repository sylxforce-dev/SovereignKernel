#pragma once
#include <vector>
#include <cstdlib>
#include <cstring>      // memcpy
#include <cmath>        // std::fabs (NOT <cstdlib>'s int-only std::abs)
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

// 1. Device abstraction (Päevast üks raudne reegel)
enum class Device {
    CPU,
    CUDA
};

enum class DataType {
    FP32,
    FP16
};

class Tensor {
private:
    float* data_;
    std::vector<int> shape_;
    std::vector<int> stride_;
    size_t num_elements_;
    Device device_;
    DataType dtype_;
    bool owns_data_; // Et vältida topeltvabastusi

    void calculate_strides() {
        stride_.resize(shape_.size());
        int stride = 1;
        for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
            stride_[i] = stride;
            stride *= shape_[i];
        }
    }

    void free_data() {
        if (owns_data_ && data_) {
            if (device_ == Device::CPU) {
                std::free(data_);
            } else if (device_ == Device::CUDA) {
                cudaFree(data_);
            }
        }
        data_ = nullptr;
    }

public:
    // Konstruktor
    Tensor(const std::vector<int>& shape, Device device = Device::CPU, DataType dtype = DataType::FP32)
        : data_(nullptr), shape_(shape), device_(device), dtype_(dtype), owns_data_(true) {

        num_elements_ = 1;
        for (int dim : shape) {
            num_elements_ *= dim;
        }
        calculate_strides();

        size_t byte_size = num_elements_ * sizeof(float); // Eeldame FP32

        if (device_ == Device::CPU) {
            // CPU mälu
            data_ = static_cast<float*>(std::malloc(byte_size));
            if (!data_) throw std::runtime_error("CPU tensor allocation failed!");
        } else if (device_ == Device::CUDA) {
            // GPU VRAM eraldamine
            cudaError_t err = cudaMalloc(&data_, byte_size);
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA tensor allocation failed: ") + cudaGetErrorString(err));
            }
        }
    }

    // RAII Destruktor – halastamatu ja automaatne vabanemine
    ~Tensor() {
        free_data();
    }

    // Keelame koopia-konstruktori, et vältida samu pointereid eri objektides
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // --- Move-konstruktor ja move-assignment ---
    // Vajalik Phase 2 jaoks (matmul-tüüpi funktsioonid, mis loovad ja
    // tagastavad uue Tensor'i, või std::vector<Tensor> kasutus).
    // Ilma nendeta ei saa Tensor'it funktsioonist väärtusena tagastada.
    Tensor(Tensor&& other) noexcept
        : data_(other.data_),
          shape_(std::move(other.shape_)),
          stride_(std::move(other.stride_)),
          num_elements_(other.num_elements_),
          device_(other.device_),
          dtype_(other.dtype_),
          owns_data_(other.owns_data_) {
        // "Röövime" pointeri teiselt objektilt, et vältida topeltvabastust
        other.data_ = nullptr;
        other.owns_data_ = false;
        other.num_elements_ = 0;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            free_data(); // vabasta oma praegune mälu enne ülekirjutamist

            data_ = other.data_;
            shape_ = std::move(other.shape_);
            stride_ = std::move(other.stride_);
            num_elements_ = other.num_elements_;
            device_ = other.device_;
            dtype_ = other.dtype_;
            owns_data_ = other.owns_data_;

            other.data_ = nullptr;
            other.owns_data_ = false;
            other.num_elements_ = 0;
        }
        return *this;
    }

    // Getterid
    float* data() const { return data_; }
    Device device() const { return device_; }
    size_t num_elements() const { return num_elements_; }
    const std::vector<int>& shape() const { return shape_; }

    // CPU vs CUDA võrdluse ankurpunkt (Phase 3 jaoks)
    void compare(const Tensor& other, float threshold = 1e-3f) const {
        if (this->num_elements_ != other.num_elements_) {
            std::cout << "Status: FAIL (Size mismatch)\n";
            return;
        }

        // Kui teine on CUDA-s, peame ajutiselt CPU-sse tõmbama kontrolliks
        std::vector<float> host_data(this->num_elements_);
        if (this->device_ == Device::CUDA) {
            cudaMemcpy(host_data.data(), this->data_, this->num_elements_ * sizeof(float), cudaMemcpyDeviceToHost);
        } else {
            std::memcpy(host_data.data(), this->data_, this->num_elements_ * sizeof(float));
        }

        std::vector<float> other_host_data(other.num_elements_);
        if (other.device_ == Device::CUDA) {
            cudaMemcpy(other_host_data.data(), other.data_, other.num_elements_ * sizeof(float), cudaMemcpyDeviceToHost);
        } else {
            std::memcpy(other_host_data.data(), other.data_, other.num_elements_ * sizeof(float));
        }

        float max_error = 0.0f;
        for (size_t i = 0; i < this->num_elements_; ++i) {
            // std::fabs, MITTE std::abs — <cstdlib>'i std::abs(int) trunkeerib
            // float'i täisarvuks enne võrdlemist, mis annab vale (peaaegu
            // alati 0) tulemuse ja teeb compare()'i mõttetuks.
            float err = std::fabs(host_data[i] - other_host_data[i]);
            if (err > max_error) max_error = err;
        }

        std::cout << "Max error: " << max_error << "\n";
        if (max_error <= threshold) {
            std::cout << "Status: PASS\n";
        } else {
            std::cout << "Status: FAIL (Exceeds threshold)\n";
        }
    }
};
