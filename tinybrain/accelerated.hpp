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

template <typename T, typename U>
inline void accumulate_2x2_x_pass(
  T* channel, U* accum,
  const size_t sx, const size_t sy, 
  const size_t osx, const size_t osy,
  const size_t yoff, const size_t oyoff
) {

  const bool odd_x = (sx & 0x01);

  size_t i_idx, o_idx;

  for (size_t x = 0, ox = 0; x < sx - (size_t)odd_x; x += 2, ox++) {
    i_idx = x + yoff;
    o_idx = ox + oyoff;
    accum[o_idx] += channel[i_idx];
    accum[o_idx] += channel[i_idx + 1];
  }

  if (odd_x) {
    // << 1 bc we need to multiply by two on the edge 
    // to avoid darkening during render
    accum[(osx - 1) + oyoff] += (U)(channel[(sx - 1) + yoff]) * 2;
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
        accumulate_2x2_x_pass<T, U>(
          channel, accum, 
          sx, sy, 
          osx, osy,
          (sx * y + zoff), (osx * oy + ozoff)
        );

        y++;

        accumulate_2x2_x_pass<T, U>(
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
        accumulate_2x2_x_pass<T, U>(
          channel, accum, 
          sx, sy, osx, osy,
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
        accumulate_2x2_x_pass<T, T>(
          channel, accum, 
          sx, sy, osx, osy,
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

template <typename T, typename U>
inline void render_image(T* accum, U* oimg, const uint32_t bitshift, const size_t ovoxels) {
  for (size_t i = 0; i < ovoxels; i++) {
    oimg[i] = (U)(accum[i] >> bitshift);
  }
}

template <typename T>
inline void render_image_floating(T* accum, T* oimg, const T divisor, const size_t ovoxels) {
  for (size_t i = 0; i < ovoxels; i++) {
    oimg[i] = accum[i] / divisor;
  }
}

template <typename T>
inline void shift_eight(T* accum, const size_t ovoxels) {
  for (size_t i = 0; i < ovoxels; i++) {
    accum[i] >>= 8;
  }
}



};

#endif
