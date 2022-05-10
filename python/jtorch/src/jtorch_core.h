#pragma once
#include "common.h"

namespace jittor {

// @pyjt(device)
struct Device {
    string name;
    

    // @pyjt(__init__)
    Device(const string& name);
};

}