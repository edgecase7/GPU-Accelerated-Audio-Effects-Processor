#ifndef PPM_HELPER_H
#define PPM_HELPER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// Basic structure to hold image data and dimensions
struct Image {
    std::vector<unsigned char> data;
    int width;
    int height;
};

// Reads a simple P3 PPM file
bool read_ppm(const std::string& filename, Image& img) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }
    std::string magic_number;
    file >> magic_number;
    if (magic_number != "P3") {
        std::cerr << "Error: Not a valid P3 PPM file." << std::endl;
        return false;
    }
    file >> img.width >> img.height;
    int max_val;
    file >> max_val;

    img.data.resize(img.width * img.height * 3);
    for (size_t i = 0; i < img.data.size(); ++i) {
        int val;
        file >> val;
        img.data[i] = static_cast<unsigned char>(val);
    }
    return true;
}

// Writes an image to a P3 PPM file (Robust Version)
bool write_ppm(const std::string& filename, const Image& img) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not create file " << filename << std::endl;
        return false;
    }
    file << "P3\n";
    file << img.width << " " << img.height << "\n";
    file << "255\n";

    int line_char_count = 0;
    for (size_t i = 0; i < img.data.size(); ++i) {
        std::string s = std::to_string(static_cast<int>(img.data[i]));
        // Add a space before the number if it's not the start of a line
        if (line_char_count > 0) {
            file << " ";
            line_char_count++;
        }
        // If adding the next number would exceed 70 characters, start a new line
        if (line_char_count + s.length() > 70) {
            file << "\n";
            line_char_count = 0;
        }
        file << s;
        line_char_count += s.length();
    }
    file << "\n"; // Ensure the file ends with a newline
    return true;
}

#endif // PPM_HELPER_H
