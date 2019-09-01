#ifndef __TIFF_H__
#define __TIFF_H__

#include "volume.h"
#include <tiffio.h>
#include <string>
#include <iostream>

inline Volume open_tiff(std::string name, bool do_normalize = false, bool binary = false, bool rgb = false)
{
    TIFF* tif = TIFFOpen(name.c_str(), "r");
    int z = 0;
   	uint32 w(0), h(0);

   	std::vector<std::vector<float> > v_data;
    do {
        size_t npixels;

        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
        npixels = w * h;
        std::cout << w << " " << h << std::endl;
		uint32 *raster = (uint32*) _TIFFmalloc(npixels * sizeof (uint32));

        std::vector<float> fdata;
        fdata.reserve(npixels);
        z++;
        if (TIFFReadRGBAImage(tif, w, h, raster, 1)) {
			//for (auto d : data)
			//	std::cout << int(d) << " ";
        	for (size_t i(0); i < npixels; ++i) {
				uint32 val(raster[i]);

                fdata.push_back(static_cast<float>(val % (1 << 8)) / 255.);
                if (rgb) {
                    val /= (1 << 8);
                    fdata.push_back(static_cast<float>(val % (1 << 8)) / 255.);

                    val /= (1 << 8);
                    fdata.push_back(static_cast<float>(val % (1 << 8)) / 255.);
                }
			}
			_TIFFfree(raster);
        } else {
        	throw "fail";
        }
        v_data.push_back(fdata);


    } while (TIFFReadDirectory(tif));

	if (do_normalize) {
		for (auto &v : v_data)
			normalize(&v);
	}

    if (binary) {
        Volume v(VolumeShape{z, 2, w, h});
        for (size_t i(0); i < v_data.size(); ++i) {
            handle_error( cudaMemcpy(v.slice(i) + v_data[i].size(), &v_data[i][0], v_data[i].size() * sizeof(F), cudaMemcpyHostToDevice));
            for (auto &v : v_data[i])
                v = 1.0 - v;
            handle_error( cudaMemcpy(v.slice(i), &v_data[i][0], v_data[i].size() * sizeof(F), cudaMemcpyHostToDevice));
        }
        TIFFClose(tif);
        return v;
    } else if(rgb) {
        Volume v(VolumeShape{z, 3, w, h});
        std::vector<std::vector<float> > v_data_rgb(v_data);
        for (size_t i(0); i < v_data.size(); ++i)
                for (size_t c(0); c < 3; ++c)
                    for (size_t n(0); n < w * h; ++n){
                        v_data_rgb[i][c*w*h+n] = v_data[i][3*n+c];
                    }

        for (size_t i(0); i < v_data.size(); ++i)
           handle_error( cudaMemcpy(v.slice(i), &v_data[i][0], v_data[i].size() * sizeof(F), cudaMemcpyHostToDevice));
        std::cout << z << std::endl;
        TIFFClose(tif);
        return v;
    } else {
        Volume v(VolumeShape{z, 1, w, h});
        for (size_t i(0); i < v_data.size(); ++i)
    	   handle_error( cudaMemcpy(v.slice(i), &v_data[i][0], v_data[i].size() * sizeof(F), cudaMemcpyHostToDevice));
        std::cout << z << std::endl;
        TIFFClose(tif);
        return v;
    }
}

inline void save_tiff(std::string name, std::vector<float> image, VolumeShape imgshape, int c = 0)
{

    TIFF *out = 0;
    // int c = 1;


    out = TIFFOpen(name.c_str(), "w");
    if (!out)
    {
            fprintf (stderr, "Can't open %s for writing\n", name.c_str());
    }

    // imgshape.z = 1;
    for (int page = 0; page <  imgshape.z; page++)
    {
        TIFFSetField(out, TIFFTAG_IMAGEWIDTH, (uint32) imgshape.w);
        TIFFSetField(out, TIFFTAG_IMAGELENGTH, (uint32) imgshape.h);
        TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, (uint16) sizeof(float) * 8);
        TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, (uint16) 1);
        TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(out, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
        // uint32 rowsperstrip = TIFFDefaultStripSize(out, (uint32) imgshape.h);
        // TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, 1);
        // TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
        //TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_BOTLEFT);
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(out, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
        TIFFSetField(out, TIFFTAG_PAGENUMBER, page, imgshape.z);

        for (int w = 0; w < imgshape.w; w++){
            float *line = (float *) &image[imgshape.offset(page, c, imgshape.w - w - 1, 0)];

            int done = TIFFWriteScanline(out, line, (uint32) w, (tsample_t) 0);

            if (done < 0) {
                fprintf(stderr, "writeTIFF: error writing row %i\n", (int) w);
            }
        }
        TIFFWriteDirectory(out);
    }

    TIFFClose(out);
}

#endif
