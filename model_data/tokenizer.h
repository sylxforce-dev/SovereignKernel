#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <unordered_map>
#include <cstdio>

struct TokenIndex {
    std::string str;
    int id;
};

class Tokenizer {
private:
    std::vector<std::string> vocab;
    std::vector<float> vocab_scores;
    size_t max_token_length;
    std::unordered_map<std::string, int> sorted_vocab;
    size_t last_token_count = 0;

public:
    Tokenizer() : max_token_length(0) {}

    void load(const std::string& filepath, int vocab_size) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open tokenizer file: " + filepath);
        }

        int max_len = 0;
        file.read(reinterpret_cast<char*>(&max_len), sizeof(int));
        max_token_length = static_cast<size_t>(max_len);

        vocab.resize(vocab_size);
        vocab_scores.resize(vocab_size);
        sorted_vocab.clear();

        for (int i = 0; i < vocab_size; ++i) {
            float score = 0.0f;
            file.read(reinterpret_cast<char*>(&score), sizeof(float));
            vocab_scores[i] = score;

            int len = 0;
            file.read(reinterpret_cast<char*>(&len), sizeof(int));

            std::string s(len, '\0');
            file.read(&s[0], len);
            vocab[i] = s;
            sorted_vocab[s] = i;
        }
        file.close();
        std::cout << "[Tokenizer Debug] Loaded " << vocab_size << " tokens. Max len: " << max_token_length << "\n";
    }

    std::string decode(int token_id) const {
        if (token_id >= 0 && static_cast<size_t>(token_id) < vocab.size()) {
            if (token_id == 1 || token_id == 2 || token_id == 0) return "";

            std::string text = vocab[token_id];

            size_t pos = 0;
            while ((pos = text.find("\xe2\x96\x81", pos)) != std::string::npos) {
                text.replace(pos, 3, " ");
                pos += 1;
            }

            if (text.length() == 6 && text.substr(0, 3) == "<0x" && text.back() == '>') {
                int byte_val;
                if (std::sscanf(text.c_str(), "<0x%02X>", &byte_val) == 1) {
                    return std::string(1, static_cast<char>(byte_val));
                }
            }
            return text;
        }
        return "";
    }

    int encode_piece(const std::string& piece) const {
        auto it = sorted_vocab.find(piece);
        if (it != sorted_vocab.end()) return it->second;
        return -1;
    }

    std::vector<int> encode(const std::string& text, bool debug_trace = false) {
        std::vector<int> tokens;
        tokens.push_back(1);

        if (text.empty()) {
            last_token_count = tokens.size();
            return tokens;
        }

        std::string sp_text;
        for (char c : text) {
            if (c == ' ') sp_text += "\xe2\x96\x81";
            else sp_text += c;
        }

        size_t i = 0;
        while (i < sp_text.length()) {
            int best_id = -1;
            size_t best_len = 0;

            size_t max_search_len = std::min(max_token_length, sp_text.length() - i);

            for (size_t len = max_search_len; len > 0; --len) {
                std::string piece = sp_text.substr(i, len);
                int id = encode_piece(piece);
                if (id != -1) {
                    best_id = id;
                    best_len = len;
                    break;
                }
            }

            if (best_id != -1) {
                tokens.push_back(best_id);
                i += best_len;
            } else {
                unsigned char byte_val = static_cast<unsigned char>(sp_text[i]);
                char hex_buf[16];
                std::snprintf(hex_buf, sizeof(hex_buf), "<0x%02X>", byte_val);
                int bid = encode_piece(hex_buf);

                if (bid != -1) {
                    tokens.push_back(bid);
                } else {
                    int unk_id = encode_piece("<unk>");
                    if (unk_id != -1) tokens.push_back(unk_id);
                }
                i += 1;
            }
        }

        last_token_count = tokens.size();
        return tokens;
    }

    size_t get_last_token_count() const {
        return last_token_count;
    }
};