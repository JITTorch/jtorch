
#include "pyjt/py_obj_holder.h"
#include "utils/str_utils.h"
#include "jtorch_core.h"

namespace jittor {

void pyjt_def_all(PyObject* m);

EXTERN_LIB void setter_use_cuda(int value);

Device::Device(const string& name) : name(name) {
    if (startswith(name, "cpu"))
        setter_use_cuda(0);
    else
        setter_use_cuda(1);
}

}

static void init_module(PyModuleDef* mdef, PyObject* m) {
    mdef->m_doc = "Inner c++ core of jtorch";
    jittor::pyjt_def_all(m);
}
PYJT_MODULE_INIT(jtorch_core);
