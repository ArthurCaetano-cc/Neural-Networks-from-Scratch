// mnist_loader.cpp
// Step 1 of our project: read MNIST IDX files and expose grayscale values [0,1]
// Build: g++ -O2 -std=c++17 -Wall -Wextra -o mnist_loader mnist_loader.cpp
// Usage: ./mnist_loader <images-idx3-ubyte> <labels-idx1-ubyte> [--print N] [--csv out.csv]
// Notes: If your files are .gz, gunzip them first: `gunzip *.gz`

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

struct MnistDataset {
    vector<float> images;   // flattened, normalized to [0,1]
    vector<uint8_t> labels; // 0..9
    size_t num_images = 0;
    size_t rows = 0;
    size_t cols = 0;

    inline size_t image_size() const { return rows * cols; }
    inline const float* image_ptr(size_t i) const { return images.data() + i * image_size(); }
    inline float*       image_ptr(size_t i)       { return images.data() + i * image_size(); }
};

static uint32_t read_u32_be(ifstream& f) {
    uint8_t b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    if (!f) throw runtime_error("Unexpected EOF while reading u32");
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

static MnistDataset load_mnist(const string& images_path, const string& labels_path, bool normalize=true) {
    MnistDataset ds;

    // Load images (IDX3)
    {
        ifstream fi(images_path, ios::binary);
        if (!fi) throw runtime_error("Could not open images file: " + images_path);
        uint32_t magic = read_u32_be(fi);
        if (magic != 2051) throw runtime_error("Images file magic mismatch: expected 2051, got " + to_string(magic));
        uint32_t n = read_u32_be(fi);
        uint32_t rows = read_u32_be(fi);
        uint32_t cols = read_u32_be(fi);
        ds.num_images = n;
        ds.rows = rows;
        ds.cols = cols;
        ds.images.resize(size_t(n) * rows * cols);

        const size_t total = size_t(n) * rows * cols;
        for (size_t i = 0; i < total; ++i) {
            unsigned char pixel = 0;
            fi.read(reinterpret_cast<char*>(&pixel), 1);
            if (!fi) throw runtime_error("Unexpected EOF while reading image data");
            ds.images[i] = normalize ? (float(pixel) / 255.0f) : float(pixel);
        }
    }

    // Load labels (IDX1)
    {
        ifstream fl(labels_path, ios::binary);
        if (!fl) throw runtime_error("Could not open labels file: " + labels_path);
        uint32_t magic = read_u32_be(fl);
        if (magic != 2049) throw runtime_error("Labels file magic mismatch: expected 2049, got " + to_string(magic));
        uint32_t n = read_u32_be(fl);
        if (n != ds.num_images) {
            throw runtime_error("Images count (" + to_string(ds.num_images) + ") != labels count (" + to_string(n) + ")");
        }
        ds.labels.resize(n);
        for (uint32_t i = 0; i < n; ++i) {
            unsigned char lab = 0;
            fl.read(reinterpret_cast<char*>(&lab), 1);
            if (!fl) throw runtime_error("Unexpected EOF while reading label data");
            ds.labels[i] = lab; // 0..9
        }
    }

    return ds;
}

static void print_image_grid(const MnistDataset& ds, size_t idx, int precision=2) {
    if (idx >= ds.num_images) throw out_of_range("Image index out of range");
    const float* img = ds.image_ptr(idx);
    cout << "Image #" << idx << " (label=" << int(ds.labels[idx]) << ")\n";
    cout << fixed << setprecision(precision);
    for (size_t r = 0; r < ds.rows; ++r) {
        for (size_t c = 0; c < ds.cols; ++c) {
            cout << setw(5) << img[r * ds.cols + c];
        }
        cout << '\n';
    }
}

static void export_csv(const MnistDataset& ds, const string& out_path, size_t max_rows=0) {
    ofstream fo(out_path);
    if (!fo) throw runtime_error("Could not open output CSV: " + out_path);
    // header: label,p0,p1,...,p783
    fo << "label";
    for (size_t i = 0; i < ds.image_size(); ++i) fo << ",p" << i;
    fo << "\n";

    size_t N = max_rows ? min(ds.num_images, max_rows) : ds.num_images;
    fo << fixed << setprecision(6);
    for (size_t idx = 0; idx < N; ++idx) {
        fo << int(ds.labels[idx]);
        const float* img = ds.image_ptr(idx);
        for (size_t i = 0; i < ds.image_size(); ++i) fo << ',' << img[i];
        fo << '\n';
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <images-idx3-ubyte> <labels-idx1-ubyte> [--print N] [--csv out.csv] [--csv-max M]\n";
        return 1;
    }
    string images_path = argv[1];
    string labels_path = argv[2];

    size_t printN = 0; // if >0, print first N images as grids
    string csvOut;
    size_t csvMax = 0;

    for (int i = 3; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--print" && i + 1 < argc) {
            printN = stoul(argv[++i]);
        } else if (arg == "--csv" && i + 1 < argc) {
            csvOut = argv[++i];
        } else if (arg == "--csv-max" && i + 1 < argc) {
            csvMax = stoul(argv[++i]);
        } else {
            cerr << "Unknown arg: " << arg << "\n";
            return 1;
        }
    }

    try {
        auto ds = load_mnist(images_path, labels_path);
        cout << "Loaded MNIST: " << ds.num_images << " images of size " << ds.rows << "x" << ds.cols << "\n";

        if (!csvOut.empty()) {
            export_csv(ds, csvOut, csvMax);
            cout << "Wrote CSV: " << csvOut << "\n";
        }

        for (size_t i = 0; i < printN && i < ds.num_images; ++i) {
            print_image_grid(ds, i, 2);
            cout << "\n";
        }

        // Quick sanity: print first 10 pixel values of image 0
        if (printN == 0 && ds.num_images > 0) {
            const float* img0 = ds.image_ptr(0);
            cout << "First 10 pixels of image 0 (normalized): ";
            for (int i = 0; i < 10; ++i) cout << img0[i] << (i+1<10?", ":"\n");
            cout << "Label: " << int(ds.labels[0]) << "\n";
        }
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 2;
    }

    return 0;
}
