#pragma once
#include "common.h"
#include "var_holder.h"
#include "misc/fast_shared_ptr.h"

namespace jittor {

// @pyjt(device)
struct Device {
    string name;
    
    // @pyjt(__init__)
    Device(const string& name);
};

// @pyjt(backward)
void backward(VarHolder* x);

// @pyjt(grad_set)
void grad_set(VarHolder* x, Maybe<VarHolder> v);
// @pyjt(grad_get)
Maybe<VarHolder> grad_get(VarHolder* x);
// @pyjt(grad_del)
void grad_del(VarHolder* x);

// @pyjt(retain_grad_set)
inline void retain_grad_set(VarHolder* x, bool v) {
    x->var->flags.set(NodeFlags::_th_require_grad, v);
}
// @pyjt(retain_grad_get)
inline bool retain_grad_get(VarHolder* x) {
    return x->var->flags.get(NodeFlags::_th_require_grad);
}

}