#ifndef _TINYBRAIN_BUILTINS_HXX_
#define _TINYBRAIN_BUILTINS_HXX_

#ifdef _MSC_VER
#  include <intrin.h>
#  define tinybrain_popcount __popcnt
// Source - https://stackoverflow.com/a/20468180
// Posted by crazyjul
// Retrieved 2026-05-12, License - CC BY-SA 3.0
unsigned long __inline tinybrain_ctz (unsigned long value) {
    unsigned long trailing_zero = 0;
    if (_BitScanForward(&trailing_zero, value)) {
        return trailing_zero;
    }
    else {
        return 32; // undefined if value 0, choose 32 as a sensible choice
    }
}
unsigned long __inline tinybrain_clz (unsigned long value) {
    unsigned long leading_zero = 0;

    if (_BitScanReverse(&leading_zero, value)) {
       return 31 - leading_zero;
    }
    else {
         return 32; // undefined if value 0, choose 32 as a sensible choice
    }
}
#else
#  define tinybrain_popcount __builtin_popcount
#  define tinybrain_ctz __builtin_ctz
#  define tinybrain_clz __builtin_clz
#endif

#endif