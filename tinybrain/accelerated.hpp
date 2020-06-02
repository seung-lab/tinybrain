/*
Copyright (C) 2019, William Silversmith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/


#ifndef ACCELERATED_HPP
#define ACCELERATED_HPP

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include "immintrin.h"

namespace accelerated {

// 2x2 below

template <typename T, typename U>
inline void accumulate_x_pass(
  T* channel, U* accum,
  const size_t sx, const size_t osx, 
  const size_t offset, const size_t o_offset
) {

  const bool odd_x = (sx & 0x01);

  size_t i_idx, o_idx;

  for (size_t x = 0, ox = 0; x < sx - (size_t)odd_x; x += 2, ox++) {
    i_idx = x + offset;
    o_idx = ox + o_offset;
    accum[o_idx] += channel[i_idx];
    accum[o_idx] += channel[i_idx + 1];
  }

  if (odd_x) {
    // << 1 bc we need to multiply by two on the edge 
    // to avoid darkening during render
    accum[(osx - 1) + o_offset] += static_cast<U>(channel[(sx - 1) + offset]) * 2;
  }
}

template <typename T>
void accumulate_2x2_dual_pass(
  T* channel, T* accum,
  const size_t sx, const size_t sy, 
  const size_t osx, const size_t osy,
  const size_t yoff, const size_t oyoff
);

template <>
inline void accumulate_2x2_dual_pass(
  float* channel, float* accum,
  const size_t sx, const size_t sy, 
  const size_t osx, const size_t osy,
  const size_t yoff, const size_t oyoff
) {

  const bool odd_x = (sx & 0x01);

  size_t x, ox;
  const size_t sxv = (sx >> 3) << 3; // minimum 8 elements

  __m128 row1, row2, res1, res2;
  for (x = 0, ox = 0; x < sxv; x += 8, ox += 4) {
    row1 = _mm_loadu_ps(channel + yoff + x);
    row2 = _mm_loadu_ps(channel + yoff + x + sx);
    res1 = _mm_add_ps(row1, row2);

    row1 = _mm_loadu_ps(channel + yoff + (x+4));
    row2 = _mm_loadu_ps(channel + yoff + (x+4) + sx);
    res2 = _mm_add_ps(row1, row2);

    _mm_storeu_ps(accum + oyoff + ox, _mm_hadd_ps(res1, res2));
  }

  size_t i_idx, o_idx;
  for (; x < sx - (size_t)odd_x; x += 2, ox++) {
    i_idx = x + yoff;
    o_idx = ox + oyoff;
    accum[o_idx] += channel[i_idx];
    accum[o_idx] += channel[i_idx + 1];
    accum[o_idx] += channel[i_idx + sx];
    accum[o_idx] += channel[i_idx + sx + 1];
  }

  if (odd_x) {
    // << 1 bc we need to multiply by two on the edge 
    // to avoid darkening during render
    accum[(osx - 1) + oyoff] += (channel[(sx - 1) + yoff]) * 2;
    accum[(osx - 1) + oyoff] += (channel[(sx - 1) + yoff + sx]) * 2;
  }
}

template <>
inline void accumulate_2x2_dual_pass(
  double* channel, double* accum,
  const size_t sx, const size_t sy, 
  const size_t osx, const size_t osy,
  const size_t yoff, const size_t oyoff
) {

  const bool odd_x = (sx & 0x01);

  size_t x, ox;
  const size_t sxv = (sx >> 2) << 2; // minimum 4 elements

  __m128d row1, row2, res1, res2;
  for (x = 0, ox = 0; x < sxv; x += 4, ox += 2) {
    row1 = _mm_loadu_pd(channel + yoff + x);
    row2 = _mm_loadu_pd(channel + yoff + x + sx);
    res1 = _mm_add_pd(row1, row2);

    row1 = _mm_loadu_pd(channel + yoff + (x+2));
    row2 = _mm_loadu_pd(channel + yoff + (x+2) + sx);
    res2 = _mm_add_pd(row1, row2);

    _mm_storeu_pd(accum + oyoff + ox, _mm_hadd_pd(res1, res2));
  }

  size_t i_idx, o_idx;
  for (; x < sx - (size_t)odd_x; x += 2, ox++) {
    i_idx = x + yoff;
    o_idx = ox + oyoff;
    accum[o_idx] += channel[i_idx];
    accum[o_idx] += channel[i_idx + 1];
    accum[o_idx] += channel[i_idx + sx];
    accum[o_idx] += channel[i_idx + sx + 1];
  }

  if (odd_x) {
    // << 1 bc we need to multiply by two on the edge 
    // to avoid darkening during render
    accum[(osx - 1) + oyoff] += (channel[(sx - 1) + yoff]) * 2;
    accum[(osx - 1) + oyoff] += (channel[(sx - 1) + yoff + sx]) * 2;
  }
}


template <typename T, typename U>
U* accumulate_2x2(
    T* channel, 
    const size_t sx, const size_t sy, 
    const size_t sz = 1, const size_t sw = 1
  ) {

  const size_t sxy = sx * sy;

  const size_t osx = (sx + 1) >> 1;
  const size_t osy = (sy + 1) >> 1;
  const size_t osxy = osx * osy;
  const size_t ovoxels = osxy * sz * sw;

  const bool odd_y = (sy & 0x01);

  U* accum = new U[ovoxels]();

  size_t y, oy;
  size_t zoff, ozoff, oyoff;

  for (size_t w = 0; w < sw; w++) {
    for (size_t z = 0; z < sz; z++) {
      zoff = sxy * (z + sz * w);
      ozoff = osxy * (z + sz * w);

      for (y = 0, oy = 0; y < sy - (size_t)odd_y; y++, oy++) {
        accumulate_x_pass<T, U>(
          channel, accum, 
          sx, osx,
          (sx * y + zoff), (osx * oy + ozoff)
        );

        y++;

        accumulate_x_pass<T, U>(
          channel, accum, 
          sx, osx,
          (sx * y + zoff), (osx * oy + ozoff)
        );
      }

      if (odd_y) {
        y = sy - 1;
        oy = osy - 1;
        oyoff = (osx * oy + ozoff);
        accumulate_x_pass<T, U>(
          channel, accum, 
          sx, osx,
          (sx * y + zoff), oyoff
        );
        
        // double values to prevent darkening 
        for (size_t x = 0; x < osx; x++) {
          accum[x + oyoff] *= 2;
        }
      }
    }
  }

  return accum;
}

template <typename T>
T* accumulate_2x2f(
    T* channel, 
    const size_t sx, const size_t sy, 
    const size_t sz = 1, const size_t sw = 1
  ) {

  const size_t sxy = sx * sy;

  const size_t osx = (sx + 1) >> 1;
  const size_t osy = (sy + 1) >> 1;
  const size_t osxy = osx * osy;
  const size_t ovoxels = osxy * sz * sw;

  const bool odd_y = (sy & 0x01);

  alignas(16) T* accum = new T[ovoxels]();

  size_t y, oy;
  size_t zoff, ozoff, oyoff;

  for (size_t w = 0; w < sw; w++) {
    for (size_t z = 0; z < sz; z++) {
      zoff = sxy * (z + sz * w);
      ozoff = osxy * (z + sz * w);

      for (y = 0, oy = 0; y < sy - (size_t)odd_y; y += 2, oy++) {
        accumulate_2x2_dual_pass<T>(
          channel, accum, 
          sx, sy, 
          osx, osy,
          (sx * y + zoff), (osx * oy + ozoff)
        );
      }

      if (odd_y) {
        y = sy - 1;
        oy = osy - 1;
        oyoff = (osx * oy + ozoff);
        accumulate_x_pass<T, T>(
          channel, accum, 
          sx, osx,
          (sx * y + zoff), oyoff
        );
        
        // double values to prevent darkening 
        for (size_t x = 0; x < osx; x++) {
          accum[x + oyoff] *= 2;
        }
      }
    }
  }

  return accum;
}

// 2x2x2 below

template <typename T, typename U>
U* accumulate_2x2x2(
    T* channel, 
    const size_t sx, const size_t sy, 
    const size_t sz, const size_t sw = 1
  ) {

  const size_t sxy = sx * sy;

  const size_t osx = (sx + 1) >> 1;
  const size_t osy = (sy + 1) >> 1;
  const size_t osz = (sz + 1) >> 1;
  const size_t osxy = osx * osy;
  const size_t ovoxels = osxy * osz * sw;

  const bool odd_y = (sy & 0x01);
  const bool odd_z = (sz & 0x01);

  U* accum = new U[ovoxels]();

  size_t y, oy;
  size_t z, oz;
  size_t zoff, ozoff, oyoff;

  for (size_t w = 0; w < sw; w++) {
    for (z = 0, oz = 0; z < sz - (size_t)odd_z; oz++) {
      ozoff = osxy * (oz + osz * w);
      
      for (size_t zz = 0; zz < 2 && z < sz - (size_t)odd_z; zz++, z++) {
        zoff = sxy * (z + sz * w);

        for (y = 0, oy = 0; y < sy - (size_t)odd_y; y++, oy++) {
          accumulate_x_pass<T, U>(
            channel, accum, 
            sx, osx,
            (sx * y + zoff), (osx * oy + ozoff)
          );

          y++;

          accumulate_x_pass<T, U>(
            channel, accum, 
            sx, osx,
            (sx * y + zoff), (osx * oy + ozoff)
          );
        }

        if (odd_y) {
          y = sy - 1;
          oy = osy - 1;
          oyoff = (osx * oy + ozoff);
          accumulate_x_pass<T, U>(
            channel, accum, 
            sx, osx,
            (sx * y + zoff), oyoff
          );
        }
      }

      if (odd_y) {
        y = sy - 1;
        oy = osy - 1;
        oyoff = (osx * oy + ozoff);
        // double values to prevent darkening 
        for (size_t x = 0; x < osx; x++) {
          accum[x + oyoff] *= 2;
        }
      }
    }

    if (odd_z) {
      z = sz - 1;
      oz = osz - 1;
      ozoff = osxy * (oz + osz * w);
      zoff = sxy * (z + sz * w);

      for (y = 0, oy = 0; y < sy - (size_t)odd_y; y++, oy++) {
        accumulate_x_pass<T, U>(
          channel, accum, 
          sx, osx,
          (sx * y + zoff), (osx * oy + ozoff)
        );

        y++;

        accumulate_x_pass<T, U>(
          channel, accum, 
          sx, osx,
          (sx * y + zoff), (osx * oy + ozoff)
        );
      }

      if (odd_y) {
        y = sy - 1;
        oy = osy - 1;
        oyoff = (osx * oy + ozoff);
        accumulate_x_pass<T, U>(
          channel, accum, 
          sx, osx,
          (sx * y + zoff), oyoff
        );
        
        // double values to prevent darkening 
        for (size_t x = 0; x < osx; x++) {
          accum[x + oyoff] *= 2;
        }
      }

      // double values to prevent darkening 
      for (size_t i = 0; i < osxy; i++) {
        accum[i + ozoff] *= 2;
      }
    }
  }

  return accum;
}

template <typename T, typename U>
inline void render_image(T* accum, U* oimg, const uint32_t bitshift, const size_t ovoxels) {
  for (size_t i = 0; i < ovoxels; i++) {
    oimg[i] = static_cast<U>(accum[i] >> bitshift);
  }
}

template <typename T>
inline void render_image_floating(T* accum, T* oimg, const T divisor, const size_t ovoxels) {
  for (size_t i = 0; i < ovoxels; i++) {
    oimg[i] = accum[i] / divisor;
  }
}

template <typename T>
inline void shift_right(T* accum, const size_t ovoxels, const size_t bits) {
  for (size_t i = 0; i < ovoxels; i++) {
    accum[i] >>= bits;
  }
}

// MODE POOLING 2x2x2
// based on code by Chris Jordan, 2017

template <typename T>
inline void _mode_pooling_2x2x2(
  T* img, T* oimg,
  const size_t sx, const size_t sy, 
  const size_t sz, const size_t sw = 1
) {

  const size_t sxy = sx * sy;

  const size_t osx = (sx + 1) >> 1;
  const size_t osy = (sy + 1) >> 1;
  const size_t osxy = osx * osy;

  T vals[8];
  T cur_val, max_val;
  size_t max_ct, cur_ct;

  for (size_t z = 0; z < sz; z += 2) {
    for (size_t y = 0; y < sy; y += 2) {
      for (size_t x = 0; x < sx; x += 2) {
        size_t offset = x + sx * y + sxy * z;
        
        size_t plus_x = (x < sx - 1);
        size_t plus_y = (y < sy - 1) * sx;
        size_t plus_z = (z < sz - 1) * sxy;

        vals[0] = img[ offset ];
        vals[1] = img[ offset + plus_x ];
        vals[2] = img[ offset + plus_y ];
        vals[3] = img[ offset + plus_x + plus_y ];
        vals[4] = img[ offset + plus_z ];
        vals[5] = img[ offset + plus_x + plus_z ];
        vals[6] = img[ offset + plus_y + plus_z ];
        vals[7] = img[ offset + plus_x + plus_y + plus_z ];
        
        size_t o_loc = (x >> 1) + osx * (y >> 1) + osxy * (z >> 1);
        // These two if statements could be removed, but they add a very small
        // cost on random data (< 10%) and can speed up connectomics data by ~4x
        if (vals[0] == vals[1] && vals[0] == vals[2] && vals[0] == vals[3]) {
          oimg[o_loc] = vals[0];
          continue;
        }
        else if (vals[4] == vals[5] && vals[4] == vals[6] && vals[4] == vals[7]) {
          oimg[o_loc] = vals[4];
          continue;
        }

        max_ct = 0;
        for (short int t = 0; t < 8; t++) {
          cur_val = vals[t];
          cur_ct = 0;
          for (short int p = 0; p < 8; p++) {
            cur_ct += (cur_val == vals[p]);
          }

          if (cur_ct > max_ct) {
              max_ct = cur_ct;
              max_val = cur_val;
          }
        }

        oimg[o_loc] = max_val;
      }
    }
  }
}


};

#endif
