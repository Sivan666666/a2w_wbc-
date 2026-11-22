// Positive-path cross-machine protocol test. DDS side is fixed to lo/domain79.
#include "remote_body.hpp"
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <iostream>
#include <cmath>
using namespace unitree::robot;
using LowState=unitree_hg::msg::dds_::LowState_;
using LowCmd=unitree_hg::msg::dds_::LowCmd_;

int main(int argc,char**argv) {
    if(argc==3 && std::string(argv[1])=="sender") {
        RemoteBody body(argv[2],16940,16941);
        body.wait_state();
        auto state=body.snapshot().first;
        for(int i=0;i<16;++i)
            if(std::abs(state.q[i]-.01*(i+1))>1e-6) return 2;
        body_link::Command command;
        for(int i=0;i<16;++i) {
            const int policy_index=i<12?i:i+6;
            command.q[i]=.01*(policy_index+1);
            command.dq[i]=-.02*(policy_index+1);
            command.kp[i]=i<12?120.:0.; command.kd[i]=i<12?3.:.6;
        }
        for(int i=0;i<100;++i) {
            body.command(command);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        auto status=body.status();
        std::cout<<"{\"isolated_domain\":79,\"commands\":"<<status.commands
                 <<",\"releases\":"<<status.releases<<"}"<<std::endl;
        return status.commands==100 && status.releases==0?0:3;
    }
    if(argc!=2 || std::string(argv[1])!="dds-peer")return 2;
    ChannelFactory::Instance()->Init(79,"lo");
    std::atomic<int> received{0},bad{0};
    ChannelSubscriber<LowCmd> input("rt/lowcmd");
    input.InitChannel([&](const void* raw) {
        const auto& command=*static_cast<const LowCmd*>(raw);
        bool ok=command.crc()==body_link::crc(command) && command.mode_machine()==42;
        for(int i=0;i<16;++i) {
            const int p=i<12?i:i+6;
            const auto& motor=command.motor_cmd()[i];
            ok=ok && motor.mode()==1 && std::abs(motor.q()-.01*(p+1))<1e-6
                  && std::abs(motor.dq()+.02*(p+1))<1e-6
                  && std::abs(motor.kp()-(i<12?120.:0.))<1e-6
                  && std::abs(motor.kd()-(i<12?3.:.6))<1e-6 && motor.tau()==0.;
        }
        if(!ok)++bad;
        ++received;
    },100);
    ChannelPublisher<LowState> output("rt/lowstate"); output.InitChannel();
    LowState state{}; state.mode_machine()=42; state.imu_state().quaternion()[0]=1.;
    for(int i=0;i<16;++i)state.motor_state()[i].q()=.01*(i+1);
    auto until=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    while(std::chrono::steady_clock::now()<until && received<100) {
        ++state.tick(); output.Write(state);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    input.CloseChannel();
    std::cout<<"{\"interface\":\"lo\",\"domain\":79,\"received\":"<<received.load()
             <<",\"bad_crc_or_fields\":"<<bad.load()<<"}"<<std::endl;
    return received==100 && bad==0?0:4;
}
