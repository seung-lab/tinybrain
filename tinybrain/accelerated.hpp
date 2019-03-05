
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>

#ifndef ACCELERATED_HPP
#define ACCELERATED_HPP

namespace accelerated {


template <typename T>
inline void accumulate_2x2_x_pass(
  T* channel, uint16_t* accum,
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
    accum[(osx - 1) + oyoff] += (uint16_t)(channel[(sx - 1) + yoff]) << 1;
  }
}

template <typename T>
uint16_t* accumulate_2x2(
    T* channel, 
    const size_t sx, const size_t sy, const size_t sz
  ) {

  const size_t sxy = sx * sy;
  const size_t voxels = sx * sy * sz;

  const size_t osx = (sx + 1) / 2;
  const size_t osy = (sy + 1) / 2;
  const size_t osxy = osx * osy;
  const size_t ovoxels = osxy * sz;

  const bool odd_x = (sx & 0x01);
  const bool odd_y = (sy & 0x01);

  uint16_t* accum = new uint16_t[ovoxels]();

  size_t y, oy;
  size_t zoff, ozoff, oyoff;

  for (size_t z = 0; z < sz; z++) {
    zoff = sxy * z;
    ozoff = osxy * z;

    for (y = 0, oy = 0; y < sy - (size_t)odd_y; y++, oy++) {
      accumulate_2x2_x_pass<T>(
        channel, accum, 
        sx, sy, 
        osx, osy,
        (sx * y + zoff), (osx * oy + ozoff)
      );

      y++;

      accumulate_2x2_x_pass<T>(
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
      accumulate_2x2_x_pass<T>(
        channel, accum, 
        sx, sy, osx, osy,
        (sx * y + zoff), oyoff
      );
      
      // double values to prevent darkening 
      for (size_t x = 0; x < osx; x++) {
        accum[x + oyoff] <<= 1;
      }
    }
  }

  return accum;
}

// float* average_pool_2x2x1x1(
//     float* channel, 
//     const size_t sx, const size_t sy, const size_t sz, const size_t sw
//   ) {

//   const size_t sxy = sx * sy;
//   const size_t voxels = sx * sy * sz;

//   const size_t osx = (sx + 1) / 2;
//   const size_t osy = (sy + 1) / 2;
//   const size_t osxy = (sxy + 3) / 4;
//   const size_t ovoxels = osxy * sz * sw;

//   float* oimg = new float[ovoxels]();

//   size_t x, ox, y, oy;
//   size_t i_idx, o_idx;

//   size_t yoff, oyoff, zoff, ozoff;

//   for (size_t w = 0; w < sw; w++) {
//     for (size_t z = 0; z < sz; z++) {
//       zoff = sxy * (z + sz * w);
//       ozoff = osxy * (z + sz * w);

//       for (y = 0, oy = 0; y < sy; y += 2, oy++) {
//         yoff = sx * y + zoff;
//         oyoff = osx * oy + ozoff;
        
//         for (x = 0, ox = 0; x < sx; x += 2, ox++) {
//           i_idx = x + yoff;
//           o_idx = ox + oyoff;
//           oimg[o_idx] += channel[i_idx];
//           oimg[o_idx] += channel[i_idx + 1];
//         }

//         y++;
//         yoff = sx * y + zoff;
//         oyoff = osx * oy + ozoff;
//         for (x = 0, ox = 0; x < sx; x += 2, ox++) {
//           i_idx = x + yoff;
//           o_idx = ox + oyoff;
//           oimg[o_idx] += channel[i_idx];
//           oimg[o_idx] += channel[i_idx + 1];
//           oimg[o_idx] /= 4.0;
//         }
//       }
//     }
//   }

//   return oimg;
// }

};

#endif
