#pragma once
#include <array>
#include <algorithm>

namespace arm_preparation {
using A6 = std::array<double, 6>;
struct Sample { A6 q{}, dq{}, ddq{}; };

// Same endpoint conditions as a2w_deploy.arm_preparation.quintic.
inline Sample sample(const A6& from, const A6& velocity, const A6& to,
                     double elapsed, double duration) {
    Sample out;
    if (elapsed >= duration) { out.q=to; return out; }
    const double u=std::clamp(elapsed/duration,0.,1.);
    for(int i=0;i<6;++i) {
        const double d=to[i]-from[i], v=duration*velocity[i];
        const double a3=10*d-6*v, a4=-15*d+8*v, a5=6*d-3*v;
        out.q[i]=from[i]+v*u+a3*u*u*u+a4*u*u*u*u+a5*u*u*u*u*u;
        out.dq[i]=(v+3*a3*u*u+4*a4*u*u*u+5*a5*u*u*u*u)/duration;
        out.ddq[i]=(6*a3*u+12*a4*u*u+20*a5*u*u*u)/(duration*duration);
    }
    return out;
}
}
