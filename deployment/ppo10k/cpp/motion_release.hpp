#pragma once
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <chrono>
#include <thread>

inline int release_native_motion() {
    unitree::robot::b2::MotionSwitcherClient client;
    client.SetTimeout(2.f);
    client.Init();
    std::string form, mode;
    int result=client.CheckMode(form,mode);
    if(result || mode.empty()) return result;
    result=client.ReleaseMode();
    if(result) return result;
    auto until=std::chrono::steady_clock::now()+std::chrono::seconds(3);
    do {
        result=client.CheckMode(form,mode);
        if(result || mode.empty())return result;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    } while(std::chrono::steady_clock::now()<until);
    return -1;
}
