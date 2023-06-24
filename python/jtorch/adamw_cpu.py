import jittor as jt
new_cc_path = "/home/cjld/test2/aocc-compiler-4.0.0/bin/clang++"
new_cc_flags = jt.flags.cc_flags + " -fnt-store -ffp-contract=fast -g "
new_cc_flags = new_cc_flags.replace("-march=native", "-mavx2 -mf16c")

import time
st1 = st2 = st3 = 0
stcount = 0

import jittor_utils
from jittor.misc import _simple_for
def isnfinite(x): return _simple_for(x, "isnan(float(x)) || isinf(float(x))")


def get_header(NT, TOTAL, N):
    TOFF = max(0, 16*jt.flags.device_id)
    return f"""
#define NT {NT}
#define TOTAL {TOTAL}
#define N {N}
#define TOFF {TOFF}
#include <thread>
#include <pthread.h>
#include <immintrin.h>

inline static void set_aff(int i) {{
    // return;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(i, &cpuset);
    int rc = pthread_setaffinity_np(pthread_self(),
                                    sizeof(cpu_set_t), &cpuset);
    if (sched_getcpu() != i)
        std::cout << "error set aff " << i << "!=" << sched_getcpu() << std::endl;
}}


#if NT == 1

#define T_EXEC_BEGIN int tid=0; set_aff(TOFF+1);
#define T_EXEC_END

#else

#define T_EXEC_BEGIN \\
    std::vector<std::thread> ts; \\
    ts.reserve(NT); \\
    set_aff(TOFF+1); \\
    for (int tid=0; tid<NT; tid++) {{ \\
        auto func = [=]() {{ \\
        set_aff(TOFF+TOTAL/NT*tid);
#define T_EXEC_END \\
        }}; \\
        ts.emplace_back(std::move(func)); \\
    }} \\
    for (int tid=0; tid<NT; tid++) ts[tid].join();

#endif


"""

def to_32(v):
    if v.dtype == "float32":
        return v
    header = get_header(2, 16, v.numel())
    v32 = jt.code(v.shape, "float32", [v], '''
    T_EXEC_BEGIN;
    ssize_t d = N/NT;
    // using __m128i  _mm_cvtps_ph(__m256 x, int imm);
    __m128i* __restrict__ a = (__m128i* __restrict__)(in0_p+tid*d);
    __m256* __restrict__ b = (__m256* __restrict__)(out0_p+tid*d);
    #define UNROLL 8
	for (ssize_t j=0; j<((N/NT-1)/8+1); j+=UNROLL) {
        @for(i,0,8,1,auto aa@i = _mm_stream_load_si128(a+j+@i);)
        @for(i,0,8,1,auto bb@i = _mm256_cvtph_ps(aa@i););
        @for(i,0,8,1,_mm256_stream_ps((float* __restrict__)(b+j+@i), bb@i););

    }
    T_EXEC_END;
        ''', cpu_header=header).sync()
    v.swap(v32)
    return v32


def to_16(v):
    if v.dtype == "float16":
        return v
    header = get_header(2, 16, v.numel())
    v16 = jt.code(v.shape, "float16", [v], '''
    T_EXEC_BEGIN;
    ssize_t d = N/NT;
    // using __m128i  _mm_cvtps_ph(__m256 x, int imm);
    __m256* __restrict__ a = (__m256* __restrict__)(in0_p+tid*d);
    __m128i* __restrict__ b = (__m128i* __restrict__)(out0_p+tid*d);
    #define UNROLL 8
	for (ssize_t j=0; j<((N/NT-1)/8+1); j+=UNROLL) {
        @for(i,0,8,1,auto aa@i = _mm256_load_ps((float* __restrict__)(a+j+@i));)
        @for(i,0,8,1,auto bb@i = _mm256_cvtps_ph(aa@i, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC););
        @for(i,0,8,1,_mm_stream_si128(b+j+@i, bb@i););
    }
    T_EXEC_END;
        ''', cpu_header=header).sync()
    v.swap(v16)
    return v16




def adamw_step(p32,g32,v32,m32,mp32,data):
    t1 = time.time()
    if p32.dtype == "float16":
        to_32(p32)
    if g32.dtype == "float16":
        to_32(g32)
    if v32.dtype == "float16":
        to_32(v32)
    if m32.dtype == "float16":
        to_32(m32)
    if mp32.dtype == "float16":
        to_32(mp32)

    t2 = time.time()
    header = get_header(2, 16, p32.numel())
    with jt.flag_scope(cc_path=new_cc_path, cc_flags=new_cc_flags):
        jt.code([1], "float32", [p32,g32,v32,m32,mp32], '''
    T_EXEC_BEGIN;
    float lr = data["lr"];
    float eps = data["eps"];
    float weight_decay = data["weight_decay"];
    float b0 = data["b0"];
    float b1 = data["b1"];
    float n = data["n"];
    float pm = 1 - lr * weight_decay;
    float bias_correction1_inv = 1.0f / (1.0f - powf(b0,n));
    float bias_correction2_inv = 1.0f / sqrtf(1.0f - powf(b1,n));
    float step_size = lr * bias_correction1_inv;

    ssize_t d = (N-1) / NT + 1;
    float* __restrict__ pp = in0_p+tid*d;
    float* __restrict__ gg = in1_p+tid*d;
    float* __restrict__ vv = in2_p+tid*d;
    float* __restrict__ mm = in3_p+tid*d;
    float* __restrict__ mmp = in4_p+tid*d;
    // LOGir << N/NT << pp << gg << vv << mm;
	for (ssize_t j=0; j<N/NT; j++) {
        if (tid*d+j>=N) 
            break;
        float p = mmp[j];
        float g = gg[j];
        float v = vv[j];
        float m = mm[j];
        p = p * pm;
        m = b0 * m + (1-b0) * g;
        v = b1 * v + (1-b1) * g * g;
        float denom = sqrtf(v) * bias_correction2_inv + eps;
        p = p - step_size * m / denom;
        //float denom = sqrtf((v*(bias_correction2_inv*bias_correction2_inv)+(eps*eps)));
        // p = p - step_size * m / denom;
        mmp[j] = p;
        vv[j] = v;
        mm[j] = m;
        pp[j] = p;
    }
    T_EXEC_END;
        ''', cpu_header=header,
    data=data).sync()
    t3 = time.time()
    # p32.isnan().sync()
    # has_inf_or_nan = isnfinite(p32).sum().item()
    # print("has_inf_or_nan", has_inf_or_nan)
    to_16(p32).sync()
    t4 = time.time()
    global stcount, st1, st2, st3
    st1 += t2 - t1
    st2 += t3 - t2
    st3 += t4 - t3
    if stcount == 0:
        print("adam time:", st1, st2, st3)
        st1 = st2 = st3 = 0
    stcount += 1

def cal_inf_sqr(g16,grad_scale):
    assert g16.dtype == "float16"
    # g32 = g16.float32()
    # find_inf = isnfinite(g32).sum().item()
    # if find_inf:
    #     all_sqr = 1.0
    # else:
    #     all_sqr = jt.sqr(g32/grad_scale).sum()
    # return all_sqr, find_inf
    with jt.flag_scope(compile_options={"FLAGS: -O2 ":1}):
        header = get_header(4, 16, "in0->num")
        s_sqr,find_inf = jt.code([[1],[1]], ["float32","int32"], [g16], '''
        int64 N2 = ((N-1)/(NT*8)+1)*(NT*8);
        for (int64 i = N;i<N2;i++){
            in0_p[i] = 0;
        }
        T_EXEC_BEGIN;
        float grad_scale = data["grad_scale"];
        float grad_scale_inv = 1.0/(grad_scale*grad_scale);
        ssize_t d = (N2-1) / NT + 1;

        // float32* __restrict__ gg = in0_p+tid*d;
        // float32 norm = 0;
        // int sum = 0;
        // for (ssize_t j=0; j<d; j++) {
        //     float g = gg[j];
        //     sum += std::isinf(g);
        //     norm += g*g*grad_scale_inv;
        // }
        float16* __restrict__ gg = in0_p+tid*d;
        __m256 norm_v = _mm256_setzero_ps(); // 初始化为 0
        __m256 sum_v = _mm256_setzero_ps(); // 初始化为 0
        __m256 grad_scale_inv_v = _mm256_set1_ps(grad_scale_inv); // 复制 grad_scale_inv 到向量的所有元素
        __m256 one_v = _mm256_set1_ps(1.0f); // 复制 1.0f 到向量的所有元素
        __m256 g_v, g_squared_v;
        __m256 inf_v = _mm256_set1_ps(INFINITY);

        for (ssize_t j=0; j<d; j+=8) { // 按照向量大小来组织循环
            g_v = _mm256_cvtph_ps(_mm_stream_load_si128((__m128i*)(gg + j))); // 加载 8 个元素
            g_squared_v = _mm256_mul_ps(g_v, g_v); // 平方
            sum_v = _mm256_add_ps(sum_v, _mm256_and_ps(one_v, _mm256_cmp_ps(g_v, inf_v, _CMP_EQ_OQ))); // 累加求和
            norm_v = _mm256_add_ps(norm_v, _mm256_mul_ps(g_squared_v, grad_scale_inv_v)); // 平方后乘以 grad_scale_inv 累加求和
        }

        float32 norm_array[8] __attribute__((aligned(32)));
        _mm256_store_ps(norm_array, norm_v); // 存储向量的结果

        float32 norm = 0;
        for (int i = 0; i < 8; i++)
            norm += norm_array[i]; // 累加求和

        float32 sum_array[8] __attribute__((aligned(32)));
        _mm256_store_ps(sum_array, sum_v); // 存储向量的结果

        int sum = 0;
        for (int i = 0; i < 8; i++)
            sum += static_cast<int>(sum_array[i]); // 累加求和
        out0_p[tid] = norm;
        out1_p[tid] = sum;

        T_EXEC_END;

        for (int i=1;i<NT;i++){
           out0_p[0] += out0_p[i];
           out1_p[0] += out1_p[i];
        }

            ''', cpu_header=header,data={"grad_scale":grad_scale})
    s_sqr = s_sqr.item()
    find_inf = find_inf.item()
    return s_sqr,find_inf

def adamw_step_v2(p16,g16,v32,m32,p32,data):
    t1 = time.time()
    assert p16.dtype == "float16"
    assert g16.dtype == "float16"
    assert p32.dtype == "float32"
    assert m32.dtype == "float32"
    assert v32.dtype == "float32"
    header = get_header(4, 16, "in0->num")
    jt.code([1], "float32", [p32,g16,v32,m32,p16], '''
    T_EXEC_BEGIN;
    float lr = data["lr"];
    float eps = data["eps"];
    float weight_decay = data["weight_decay"];
    float b0 = data["b0"];
    float b1 = data["b1"];
    float n = data["n"];
    float grad_scale_inv = 1.0 / data["grad_scale"];
    float pm = 1 - lr * weight_decay;
    float bias_correction1_inv = 1.0f / (1.0f - powf(b0,n));
    float bias_correction2_inv = 1.0f / sqrtf(1.0f - powf(b1,n));
    float step_size = lr * bias_correction1_inv;

    ssize_t d = (N-1)/NT+1;
    float* __restrict__ pp = in0_p+tid*d;
    float16* __restrict__ gg = in1_p+tid*d;
    float* __restrict__ vv = in2_p+tid*d;
    float* __restrict__ mm = in3_p+tid*d;
    float16* __restrict__ pp16 = in4_p+tid*d;

    __m256 pm_v = _mm256_set1_ps(pm);
    __m256 b0_v = _mm256_set1_ps(b0);
    __m256 b1_v = _mm256_set1_ps(b1);
    __m256 x_b0_v = _mm256_set1_ps((1-b0)*grad_scale_inv);
    __m256 x_b1_v = _mm256_set1_ps((1-b1)*grad_scale_inv*grad_scale_inv);
    __m256 step_size_v = _mm256_set1_ps(step_size);
    __m256 eps_v = _mm256_set1_ps(eps);
    __m256 eps2_v = _mm256_set1_ps(1e-30f);
    __m256 bias_correction2_inv_v = _mm256_set1_ps(bias_correction2_inv);

    for (ssize_t j=0; j<d; j+=8) {
        auto g16_v = _mm_stream_load_si128((__m128i* __restrict__)(gg + j));
        __m256 p_v = _mm256_loadu_ps(pp + j);
        __m256 v_v = _mm256_loadu_ps(vv + j);
        __m256 m_v = _mm256_loadu_ps(mm + j);

        auto g_v = _mm256_cvtph_ps(g16_v);

        p_v = _mm256_mul_ps(p_v, pm_v);
        m_v = _mm256_fmadd_ps(b0_v, m_v, _mm256_mul_ps(x_b0_v, g_v));
        v_v = _mm256_fmadd_ps(b1_v, v_v, _mm256_mul_ps(x_b1_v, _mm256_mul_ps(g_v, g_v)));
        __m256 sqrt_v_v = _mm256_mul_ps(v_v, _mm256_rsqrt_ps(_mm256_max_ps(v_v, eps2_v)));
        __m256 denom_v = _mm256_add_ps(_mm256_mul_ps(sqrt_v_v, bias_correction2_inv_v), eps_v);
        p_v = _mm256_sub_ps(p_v, _mm256_div_ps(_mm256_mul_ps(step_size_v, m_v), denom_v));

        auto p16_v = _mm256_cvtps_ph(p_v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        _mm256_stream_ps(pp + j, p_v);
        _mm256_stream_ps(vv + j, v_v);
        _mm256_stream_ps(mm + j, m_v);
        _mm_stream_si128((__m128i* __restrict__)(pp16 + j), p16_v);
    }
    T_EXEC_END;
        ''', cpu_header=header,
    data=data).sync()
    t2 = time.time()
    global stcount, st1, st2, st3
    st1 += t2 - t1
    if stcount == 0:
        print("adam time:", st1)
        st1 = st2 = st3 = 0
    stcount += 1



cpu_adam_module = jittor_utils.compile_module(get_header(4, 16, "in0->num") + '''
#include "common.h"
#include "var_holder.h"
#include "type/fp16_compute.h"
#include "var.h"

namespace jittor {
    
extern void free_var_mem(Var* v);

// @pyjt(adamw_step_v3)
void adamw_step_v3(VarHolder* p, VarHolder* g, VarHolder* v, VarHolder* m, VarHolder* p32, unordered_map<string, float64>& data, int64 _) {

    // only update params once
    static unordered_map<int64, int64> last;
    static int64 last_step = 0;
    static int64 total_bytes = 0, total_ns = 0;
    int64 step = data["n"];
    if (last_step != step) {
        last_step = step;
        LOGir << "update param count:" << last.size() << "total_bytes(GB):" << total_bytes/1e9 << "total_time:" << total_ns/1e9 << "BW(GB/s):" << total_bytes*1.0/total_ns;
        total_ns = total_bytes = 0;
    }
    // LOGir << "check update param " << p32->var << step;
    if (last[p32->var->id] != step) {
        last[p32->var->id] = step;
    } else {
        //LOGir << "return";
        return;
    }

    //LOGir<<"adam step";


    ASSERT(!g->var->allocator->is_cuda());
    ASSERT(!v->var->allocator->is_cuda());
    ASSERT(!m->var->allocator->is_cuda());
    ASSERT(!p32->var->allocator->is_cuda());
    if (p->var->allocator->is_cuda()) {
        free_var_mem(p->var);
        p->var->alloc(cpu_allocator);
    }
    
    auto in0 = p32->var;
    auto in0_p = in0->ptr<float32>();
    auto in1_p = g->var->ptr<float16>();
    auto in2_p = v->var->ptr<float32>();
    auto in3_p = m->var->ptr<float32>();
    auto in4_p = p->var->ptr<float16>();

    float lr = data["lr"];
    float eps = data["eps"];
    float weight_decay = data["weight_decay"];
    float b0 = data["b0"];
    float b1 = data["b1"];
    float n = data["n"];
    float grad_scale_inv = 1.0 / data["grad_scale"];

    //LOGir<<"Optim hyp"<<lr<<eps<<weight_decay<<b0<<b1<<n<<grad_scale_inv;

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    T_EXEC_BEGIN;
    float pm = 1 - lr * weight_decay;
    float bias_correction1_inv = 1.0f / (1.0f - powf(b0,n));
    float bias_correction2_inv = 1.0f / sqrtf(1.0f - powf(b1,n));
    float step_size = lr * bias_correction1_inv;

    ssize_t d = (N-1)/NT+1;
    float* __restrict__ pp = (float* __restrict__)(in0_p+tid*d);
    float16* __restrict__ gg = (float16* __restrict__)(in1_p+tid*d);
    float* __restrict__ vv = (float* __restrict__)(in2_p+tid*d);
    float* __restrict__ mm = (float* __restrict__)(in3_p+tid*d);
    float16* __restrict__ pp16 = (float16* __restrict__)(in4_p+tid*d);

    __m256 pm_v = _mm256_set1_ps(pm);
    __m256 b0_v = _mm256_set1_ps(b0);
    __m256 b1_v = _mm256_set1_ps(b1);
    __m256 x_b0_v = _mm256_set1_ps((1-b0)*grad_scale_inv);
    __m256 x_b1_v = _mm256_set1_ps((1-b1)*grad_scale_inv*grad_scale_inv);
    __m256 step_size_v = _mm256_set1_ps(step_size);
    __m256 eps_v = _mm256_set1_ps(eps);
    __m256 eps2_v = _mm256_set1_ps(1e-30f);
    __m256 bias_correction2_inv_v = _mm256_set1_ps(bias_correction2_inv);

    for (ssize_t j=0; j<d; j+=8) {
        auto g16_v = _mm_stream_load_si128((__m128i* __restrict__)(gg + j));
        __m256 p_v = _mm256_loadu_ps(pp + j);
        __m256 v_v = _mm256_loadu_ps(vv + j);
        __m256 m_v = _mm256_loadu_ps(mm + j);

        auto g_v = _mm256_cvtph_ps(g16_v);

        p_v = _mm256_mul_ps(p_v, pm_v);
        m_v = _mm256_fmadd_ps(b0_v, m_v, _mm256_mul_ps(x_b0_v, g_v));
        v_v = _mm256_fmadd_ps(b1_v, v_v, _mm256_mul_ps(x_b1_v, _mm256_mul_ps(g_v, g_v)));
        __m256 sqrt_v_v = _mm256_mul_ps(v_v, _mm256_rsqrt_ps(_mm256_max_ps(v_v, eps2_v)));
        __m256 denom_v = _mm256_add_ps(_mm256_mul_ps(sqrt_v_v, bias_correction2_inv_v), eps_v);
        p_v = _mm256_sub_ps(p_v, _mm256_div_ps(_mm256_mul_ps(step_size_v, m_v), denom_v));

        auto p16_v = _mm256_cvtps_ph(p_v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        _mm256_stream_ps(pp + j, p_v);
        _mm256_stream_ps(vv + j, v_v);
        _mm256_stream_ps(mm + j, m_v);
        _mm_stream_si128((__m128i* __restrict__)(pp16 + j), p16_v);
    }
    T_EXEC_END;
    
    auto now = std::chrono::high_resolution_clock::now();
    total_ns +=  (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(now-start).count();
    total_bytes += N*1ll*(4*3*2+4);
}


// @pyjt(adamw_step_v4)
void adamw_step_v4(VarHolder* p, int64 ga, VarHolder* v, VarHolder* m, VarHolder* p32, unordered_map<string, float64>& data){
    auto gh = VarHolder((Var*)ga);
    auto g  = &gh;
    adamw_step_v3(p,g,v,m,p32,data,0);
}

// @pyjt(grad_norm)
float64 grad_norm(int64 _) {
    auto in0 = (Var*)_;
    auto in0_p = in0->ptr<float16>();

    #define UNROLL 4
    float _tsum[NT];
    float* tsum = &_tsum[0];
    int A = NT*8*UNROLL;
    int64 NN = ((N-1)/A+1) * A;
    for (int i=N; i<NN; i++) in0_p[i] = 0;

    T_EXEC_BEGIN;
    ssize_t d = (NN-1)/NT+1;
    __m128i* __restrict__ a = (__m128i* __restrict__)(in0_p+tid*d);
    __m256 sum = _mm256_set1_ps(0);
    const __m256i mask = _mm256_set1_epi32(255);
    int has_inf_or_nan = 0;
	for (ssize_t j=0; j<((d-1)/8+1); j+=UNROLL) {
              auto aa0 = _mm_stream_load_si128(a+j+0);      
              auto aa1 = _mm_stream_load_si128(a+j+1);      
              auto aa2 = _mm_stream_load_si128(a+j+2);      
              auto aa3 = _mm_stream_load_si128(a+j+3);
              auto bb0 = _mm256_cvtph_ps(aa0);      
              auto bb1 = _mm256_cvtph_ps(aa1);      
              auto bb2 = _mm256_cvtph_ps(aa2);      
              auto bb3 = _mm256_cvtph_ps(aa3);
              auto bb20 = _mm256_mul_ps(bb0, bb0);      
              auto bb21 = _mm256_mul_ps(bb1, bb1);      
              auto bb22 = _mm256_mul_ps(bb2, bb2);      
              auto bb23 = _mm256_mul_ps(bb3, bb3);
              sum = _mm256_add_ps(sum, bb20);      
              sum = _mm256_add_ps(sum, bb21);      
              sum = _mm256_add_ps(sum, bb22);      
              sum = _mm256_add_ps(sum, bb23);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, sum);
    for (int i=1; i<8; i++) tmp[0] += tmp[i];
    tsum[tid] = tmp[0];
    T_EXEC_END;

    for (int i=1; i<NT; i++) tsum[0] += tsum[i];
    return tsum[0];
}

// @pyjt(grad_name)
string grad_name(int64 _) {
    auto in0 = (Var*)_;
    return string(in0->name.c_str());
}
}

''', jt.compiler.cc_flags + " -O2 ")
