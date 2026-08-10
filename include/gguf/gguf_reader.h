#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <stdexcept>

enum class GGUFValueType : uint32_t {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3,
    UINT32 = 4, INT32 = 5, FLOAT32 = 6, BOOL = 7,
    STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11, FLOAT64 = 12
};

struct GGUFTensorInfo {
    std::string name;
    uint32_t n_dimensions;
    std::vector<uint64_t> dimensions;
    uint32_t type;
    uint64_t offset;
};

struct GGUFHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t kv_count;
};

class GGUFReader {
public:
    GGUFHeader header;
    std::unordered_map<std::string, std::string> metadata_string;
    std::unordered_map<std::string, uint32_t> metadata_uint32;
    std::vector<GGUFTensorInfo> tensors;
    uint64_t data_offset = 0;

    void load(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("[GGUF Error] Ei suutnud avada GGUF faili: " + filepath);
        }

        file.read(reinterpret_cast<char*>(&header.magic), sizeof(header.magic));
        file.read(reinterpret_cast<char*>(&header.version), sizeof(header.version));

        if (header.magic != 0x46554747) { // "GGUF"
            throw std::runtime_error("[GGUF Error] Vigane maagiline number! See ei ole GGUF fail.");
        }

        file.read(reinterpret_cast<char*>(&header.tensor_count), sizeof(header.tensor_count));
        file.read(reinterpret_cast<char*>(&header.kv_count), sizeof(header.kv_count));

        std::cout << "[GGUF] Versioon: " << header.version
                  << " | Tensorid: " << header.tensor_count
                  << " | KV Paarid: " << header.kv_count << "\n";

        for (uint64_t i = 0; i < header.kv_count; ++i) {
            std::string key = read_string(file);
            uint32_t value_type_val;
            file.read(reinterpret_cast<char*>(&value_type_val), sizeof(value_type_val));
            auto val_type = static_cast<GGUFValueType>(value_type_val);

            if (val_type == GGUFValueType::STRING) {
                metadata_string[key] = read_string(file);
            } else if (val_type == GGUFValueType::UINT32 || val_type == GGUFValueType::INT32) {
                uint32_t val;
                file.read(reinterpret_cast<char*>(&val), sizeof(val));
                metadata_uint32[key] = val;
            } else {
                skip_value(file, val_type);
            }
        }

        for (uint64_t i = 0; i < header.tensor_count; ++i) {
            GGUFTensorInfo t;
            t.name = read_string(file);
            file.read(reinterpret_cast<char*>(&t.n_dimensions), sizeof(t.n_dimensions));

            t.dimensions.resize(t.n_dimensions);
            for (uint32_t d = 0; d < t.n_dimensions; ++d) {
                file.read(reinterpret_cast<char*>(&t.dimensions[d]), sizeof(t.dimensions[d]));
            }

            file.read(reinterpret_cast<char*>(&t.type), sizeof(t.type));
            file.read(reinterpret_cast<char*>(&t.offset), sizeof(t.offset));

            tensors.push_back(t);
        }

        // Määrame tegeliku andmete alguspunkti failis ja joondame selle
        data_offset = file.tellg();
        size_t alignment = 32;
        if (data_offset % alignment != 0) {
            data_offset = ((data_offset + alignment - 1) / alignment) * alignment;
        }

        std::cout << "[GGUF] Päis loetud edukalt! Andmete algusnihe (data_offset): " << data_offset << "\n";
    }

private:
    std::string read_string(std::ifstream& file) {
        uint64_t len;
        file.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string str(len, '\0');
        file.read(&str[0], len);
        return str;
    }

    void skip_value(std::ifstream& file, GGUFValueType type) {
        if (type == GGUFValueType::UINT8 || type == GGUFValueType::INT8 || type == GGUFValueType::BOOL) {
            file.seekg(1, std::ios::cur);
        } else if (type == GGUFValueType::UINT16 || type == GGUFValueType::INT16) {
            file.seekg(2, std::ios::cur);
        } else if (type == GGUFValueType::UINT32 || type == GGUFValueType::INT32 || type == GGUFValueType::FLOAT32) {
            file.seekg(4, std::ios::cur);
        } else if (type == GGUFValueType::UINT64 || type == GGUFValueType::INT64 || type == GGUFValueType::FLOAT64) {
            file.seekg(8, std::ios::cur);
        } else if (type == GGUFValueType::STRING) {
            uint64_t len;
            file.read(reinterpret_cast<char*>(&len), sizeof(len));
            file.seekg(len, std::ios::cur);
        } else if (type == GGUFValueType::ARRAY) {
            uint32_t arr_type_val;
            file.read(reinterpret_cast<char*>(&arr_type_val), sizeof(arr_type_val));
            uint64_t arr_len;
            file.read(reinterpret_cast<char*>(&arr_len), sizeof(arr_len));

            auto arr_type = static_cast<GGUFValueType>(arr_type_val);
            for (uint64_t i = 0; i < arr_len; ++i) {
                skip_value(file, arr_type);
            }
        } else {
            throw std::runtime_error("[GGUF Error] Tundmatu väärtuse tüüp parseris.");
        }
    }
};