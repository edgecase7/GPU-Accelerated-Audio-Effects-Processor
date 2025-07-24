#ifndef WAV_HELPER_H
#define WAV_HELPER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// A minimal WAV header structure for 16-bit mono PCM files
#pragma pack(push, 1)
struct WavHeader {
    char riff_header[4];    // "RIFF"
    int chunk_size;
    char wave_header[4];    // "WAVE"
    char fmt_header[4];     // "fmt "
    int fmt_chunk_size;     // 16
    short audio_format;     // 1 for PCM
    short num_channels;     // 1 for mono
    int sample_rate;
    int byte_rate;
    short block_align;
    short bits_per_sample;
    char data_header[4];    // "data"
    int data_chunk_size;
};
#pragma pack(pop)

// Reads a 16-bit mono WAV file into a vector of floats normalized to [-1.0, 1.0]
bool read_wav(const std::string& filename, std::vector<float>& samples, int& sample_rate) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    WavHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WavHeader));
    sample_rate = header.sample_rate;

    if (std::string(header.riff_header, 4) != "RIFF" || header.bits_per_sample != 16) {
        std::cerr << "Error: Unsupported WAV format. Please use 16-bit mono." << std::endl;
        return false;
    }

    int num_samples = header.data_chunk_size / (header.bits_per_sample / 8);
    samples.resize(num_samples);
    std::vector<short> temp_samples(num_samples);
    file.read(reinterpret_cast<char*>(temp_samples.data()), header.data_chunk_size);

    for (int i = 0; i < num_samples; ++i) {
        samples[i] = static_cast<float>(temp_samples[i]) / 32768.0f;
    }

    return true;
}

// Writes a vector of floats to a 16-bit mono WAV file
bool write_wav(const std::string& filename, const std::vector<float>& samples, int sample_rate) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not create file " << filename << std::endl;
        return false;
    }

    WavHeader header;
    strncpy(header.riff_header, "RIFF", 4);
    strncpy(header.wave_header, "WAVE", 4);
    strncpy(header.fmt_header, "fmt ", 4);
    strncpy(header.data_header, "data", 4);

    header.fmt_chunk_size = 16;
    header.audio_format = 1;
    header.num_channels = 1;
    header.sample_rate = sample_rate;
    header.bits_per_sample = 16;
    header.byte_rate = sample_rate * header.num_channels * (header.bits_per_sample / 8);
    header.block_align = header.num_channels * (header.bits_per_sample / 8);
    header.data_chunk_size = samples.size() * header.num_channels * (header.bits_per_sample / 8);
    header.chunk_size = 36 + header.data_chunk_size;

    file.write(reinterpret_cast<const char*>(&header), sizeof(WavHeader));

    std::vector<short> int_samples(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        float sample = samples[i];
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        int_samples[i] = static_cast<short>(sample * 32767.0f);
    }

    file.write(reinterpret_cast<const char*>(int_samples.data()), header.data_chunk_size);
    return true;
}

#endif // WAV_HELPER_H