#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <chrono>

// Fixed little-endian ABI shared by the robot's ARM64 NX and x86_64 PC2.
namespace body_link {
constexpr uint32_t magic = 0x57425031;
enum Kind : uint16_t { STATE=1, COMMAND=2, RELEASE=3, RELEASE_REPLY=4,
                       STATUS=5, STATUS_REPLY=6 };
struct Header { uint32_t magic_value=magic; uint16_t version=1, kind=0; uint64_t sequence=0; };
struct State {
    Header header;
    uint64_t source_monotonic_ns=0;
    uint32_t tick=0;
    uint8_t machine=0, reserved[3]{};
    std::array<double,16> q{}, dq{};
    std::array<double,4> quaternion{1.,0.,0.,0.};
    std::array<double,3> gyro{};
};
struct Command { Header header; std::array<double,16> q{}, dq{}, kp{}, kd{}; };
struct Reply { Header header; int32_t result=0; uint32_t reserved=0; };
struct Status { Header header; uint64_t states=0, commands=0, releases=0; };
static_assert(sizeof(Header)==16 && sizeof(State)==344 && sizeof(Command)==528);
static_assert(sizeof(Reply)==24 && sizeof(Status)==40);
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "Protocol uses little endian");
inline uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline bool valid(const Header& h) { return h.magic_value==magic && h.version==1; }
template<class T> uint32_t crc(const T& message) {
    uint32_t value=0xffffffffu;
    const auto* bytes=reinterpret_cast<const unsigned char*>(&message);
    for(size_t i=0;i<sizeof(message)-4;i+=4) {
        uint32_t word; std::memcpy(&word,bytes+i,4);
        for(uint32_t bit=0x80000000u;bit;bit>>=1) {
            const bool top=value&0x80000000u;
            value<<=1;
            if(top)value^=0x04c11db7u;
            if(word&bit)value^=0x04c11db7u;
        }
    }
    return value;
}
}
