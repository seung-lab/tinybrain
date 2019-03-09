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


#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>

#ifndef ACCELERATED_HPP
#define ACCELERATED_HPP

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

template <typename T, typename U>
U* accumulate_2x2(
    T* channel, 
    const size_t sx, const size_t sy, const size_t sz = 1, const size_t sw = 1
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

template <typename T, typename U>
inline void render_image(T* accum, U* oimg, const uint32_t bitshift, const size_t ovoxels) {
  for (size_t i = 0; i < ovoxels; i++) {
    oimg[i] = (T)(accum[i] >> bitshift);
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
