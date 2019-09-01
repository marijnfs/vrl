#ifndef __RAW_H__
#define __RAW_H__

#include "volume.h"
#include "util.h"
#include <string>
#include <iostream>

inline Volume open_raw(std::string name1, std::string name2, std::string name3, int W, int H, int Z)
{
    std::ifstream file1(name1.c_str(), std::ios::binary);
    std::ifstream file2(name2.c_str(), std::ios::binary);
    std::ifstream file3(name3.c_str(), std::ios::binary);

    if (!file1 || !file2 || !file3)
        throw StringException(name1.c_str());

    Volume volume(VolumeShape{Z, 3, W, H});
    std::vector<float> data(Z*3*W*H);
    std::vector<float>::iterator it(data.begin());



    for (int z(0); z < Z; z++){
        for (int i(0); i < W*H; i++, it++){
            *it = (float)byte_read<short>(file1);

        }
        for (int i(0); i < W*H; i++, it++){
            *it = (float)byte_read<short>(file2);
        }
        for (int i(0); i < W*H; i++, it++){
            *it = (float)byte_read<short>(file3);

        }

    }

    // std::vector<float>::iterator it1(data.begin());
    // for (int n(0); n < Z*3; n++, it1 += (W * H)){
    //     normalize(it1, it1 + (W * H));
    // }

    // normalization
    std::vector<float> means(3);
    std::vector<float> vars(3);

    std::vector<float>::iterator it1(data.begin());
    for (int n(0); n < Z; n++)
        for (int c(0); c < 3; c++)
            for (int i(0); i < W * H; i++, it1++)
                means[c] += *it1;

    for (int c(0); c < 3; c++)
        means[c] /= Z * W * H;

    it1 = data.begin();
    for (int n(0); n < Z; n++)
        for (int c(0); c < 3; c++)
            for (int i(0); i < W * H; i++, it1++)
                vars[c] += (*it1-means[c])*(*it1-means[c]);


    for (int c(0); c < 3; c++)
        vars[c] = sqrt(vars[c] / (Z * W * H - 1));

    it1 = data.begin();
    for (int n(0); n < Z; n++)
        for (int c(0); c < 3; c++)
            for (int i(0); i < W * H; i++, it1++)
                *it1 = (*it1-means[c])/vars[c];



    volume.from_vector(data);

    std::cout << "read " << volume.shape << " "  << volume.buf->n << std::endl;
    return volume;
}

inline Volume open_raw5(std::string name1, std::string name2, std::string name3, std::string name4, std::string name5,
                            int W, int H, int Z)
{
    std::ifstream file1(name1.c_str(), std::ios::binary);
    std::ifstream file2(name2.c_str(), std::ios::binary);
    std::ifstream file3(name3.c_str(), std::ios::binary);
    std::ifstream file4(name4.c_str(), std::ios::binary);
    std::ifstream file5(name5.c_str(), std::ios::binary);

    if (!file1 || !file2 || !file3 || !file4 || !file5)
        throw StringException(name1.c_str());

    Volume volume(VolumeShape{Z, 5, W, H});
    std::vector<float> data(Z*5*W*H);
    std::vector<float>::iterator it(data.begin());



    for (int z(0); z < Z; z++){
        for (int i(0); i < W*H; i++, it++){
            *it = (float)byte_read<short>(file1);

        }
        for (int i(0); i < W*H; i++, it++){
            *it = (float)byte_read<short>(file2);
        }
        for (int i(0); i < W*H; i++, it++){
            *it = (float)byte_read<short>(file3);
        }
        for (int i(0); i < W*H; i++, it++){
            *it = (float)byte_read<short>(file4);
        }
        for (int i(0); i < W*H; i++, it++){
            *it = (float)byte_read<short>(file5);
        }

    }

    // std::vector<float>::iterator it1(data.begin());
    // for (int n(0); n < Z*3; n++, it1 += (W * H)){
    //     normalize(it1, it1 + (W * H));
    // }

    // normalization
    std::vector<float> means(5);
    std::vector<float> vars(5);

    std::vector<float>::iterator it1(data.begin());
    for (int n(0); n < Z; n++)
        for (int c(0); c < 5; c++)
            for (int i(0); i < W * H; i++, it1++)
                means[c] += *it1;

    for (int c(0); c < 5; c++)
        means[c] /= Z * W * H;

    it1 = data.begin();
    for (int n(0); n < Z; n++)
        for (int c(0); c < 5; c++)
            for (int i(0); i < W * H; i++, it1++)
                vars[c] += (*it1-means[c])*(*it1-means[c]);


    for (int c(0); c < 5; c++)
        vars[c] = sqrt(vars[c] / (Z * W * H - 1));

    it1 = data.begin();
    for (int n(0); n < Z; n++)
        for (int c(0); c < 5; c++)
            for (int i(0); i < W * H; i++, it1++)
                *it1 = (*it1-means[c])/vars[c];



    volume.from_vector(data);

    std::cout << "read " << volume.shape << " "  << volume.buf->n << std::endl;
    return volume;
}

inline Volume open_raw(std::string name, int W, int H, int Z)
{
    int C = 5;
    std::ifstream file(name.c_str(), std::ios::binary);

    if (!file)
        throw StringException(name.c_str());

    Volume volume(VolumeShape{Z, C, W, H});
    std::vector<float> data(Z * C * W * H);

    for (int z(0); z < Z; z++){
        for (int i(0); i < W*H; i++){
            float value = byte_read<float>(file);
            int label(0);
            if (value == 5 || value == 6) label = 1;
            if (value == 1 || value == 2) label = 2;
            if (value == 3 || value == 4) label = 3;
            if (value == 7 || value == 8) label = 4;

            data[z*C*W*H + label * W*H + i] = 1;

        }
    }

    volume.from_vector(data);

    return volume;

}

inline void save_raw_classification(std::string name, std::vector<float> image, VolumeShape s)
{
    std::vector<short> vec(s.z * s.w * s.h);

    for (int z(0); z < s.z; z++)
        for (int y(0); y < s.h; y++)
            for (int x(0); x < s.w; x++){
                int best = 0;
                float value = 0;
                for (int c(0); c < s.c; c++){
                    float v = image[z * s.h * s.w * s.c + c * s.h * s.w + x * s.h + y];
                    if (v > value){
                        best = c;
                        value = v;
                    }
                }
                vec[z * s.w * s.h + x * s.h + y] = best;
            }


    std::ofstream file(name.c_str(), std::ios::binary);

    for (auto x: vec){
        byte_write(file, x);
    }
}


#endif
