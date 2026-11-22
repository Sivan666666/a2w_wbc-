// PC2 owns A2W DDS. Starting this process only subscribes and forwards state.
// LowCmd and mode changes occur only on explicit NX requests.
#include "body_protocol.hpp"
#include "motion_release.hpp"
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <poll.h>
#include <unistd.h>
#include <atomic>
#include <csignal>
#include <iostream>

using namespace unitree::robot;
using LowState=unitree_hg::msg::dds_::LowState_;
using LowCmd=unitree_hg::msg::dds_::LowCmd_;
volatile std::sig_atomic_t running=1;
void stop(int){running=0;}

int main(int argc,char** argv) {
    if(argc<5) {
        std::cerr<<"Usage: pc2_body_bridge DDS_INTERFACE NX_IP PC2_PORT NX_PORT [DDS_DOMAIN]\n";
        return 2;
    }
    const int domain=argc>5?std::stoi(argv[5]):0;
    int fd=::socket(AF_INET,SOCK_DGRAM,0);
    sockaddr_in own{}; own.sin_family=AF_INET; own.sin_port=htons(std::stoi(argv[3]));
    sockaddr_in peer{}; peer.sin_family=AF_INET; peer.sin_port=htons(std::stoi(argv[4]));
    if(fd<0 || inet_pton(AF_INET,argv[2],&peer.sin_addr)!=1 ||
       ::bind(fd,reinterpret_cast<sockaddr*>(&own),sizeof(own))<0 ||
       ::connect(fd,reinterpret_cast<sockaddr*>(&peer),sizeof(peer))<0) {
        perror("body bridge UDP"); return 1;
    }
    std::atomic<uint64_t> states{0};
    std::atomic<uint8_t> machine{0};
    uint64_t commands=0,releases=0;
    ChannelFactory::Instance()->Init(domain,argv[1]);
    ChannelPublisher<LowCmd> output("rt/lowcmd"); output.InitChannel();
    ChannelSubscriber<LowState> input("rt/lowstate");
    input.InitChannel([&](const void* raw) {
        const auto& s=*static_cast<const LowState*>(raw);
        body_link::State packet; packet.header.kind=body_link::STATE;
        packet.header.sequence=++states;
        packet.source_monotonic_ns=body_link::now_ns(); packet.tick=s.tick();
        packet.machine=s.mode_machine(); machine=packet.machine;
        for(int i=0;i<16;++i){packet.q[i]=s.motor_state()[i].q();packet.dq[i]=s.motor_state()[i].dq();}
        for(int i=0;i<4;++i)packet.quaternion[i]=s.imu_state().quaternion()[i];
        for(int i=0;i<3;++i)packet.gyro[i]=s.imu_state().gyroscope()[i];
        ::send(fd,&packet,sizeof(packet),MSG_DONTWAIT);
    },1);
    std::signal(SIGINT,stop); std::signal(SIGTERM,stop);
    std::cout<<"PC2 body bridge: "<<argv[1]<<" domain="<<domain<<" -> "<<argv[2]
             <<":"<<argv[4]<<"; awaiting NX commands; no automatic ReleaseMode"<<std::endl;
    while(running) {
        pollfd p{fd,POLLIN,0};
        if(::poll(&p,1,100)<=0)continue;
        std::array<char,2048> bytes;
        auto n=::recv(fd,bytes.data(),bytes.size(),MSG_DONTWAIT);
        if(n<static_cast<ssize_t>(sizeof(body_link::Header)))continue;
        body_link::Header h; std::memcpy(&h,bytes.data(),sizeof(h));
        if(!body_link::valid(h))continue;
        if(h.kind==body_link::COMMAND && n==sizeof(body_link::Command)) {
            body_link::Command command; std::memcpy(&command,bytes.data(),sizeof(command));
            LowCmd msg{}; msg.mode_pr()=0; msg.mode_machine()=machine;
            for(int i=0;i<16;++i) {
                auto& motor=msg.motor_cmd()[i]; motor.mode()=1;
                motor.q()=command.q[i]; motor.dq()=command.dq[i];
                motor.kp()=command.kp[i]; motor.kd()=command.kd[i]; motor.tau()=0.;
            }
            msg.crc()=body_link::crc(msg); output.Write(msg); ++commands;
        } else if(h.kind==body_link::RELEASE && n==sizeof(h)) {
            ++releases;
            body_link::Reply reply; reply.header=h; reply.header.kind=body_link::RELEASE_REPLY;
            try{reply.result=release_native_motion();}catch(...){reply.result=-2;}
            ::send(fd,&reply,sizeof(reply),0);
            std::cout<<"NX release request result="<<reply.result<<std::endl;
        } else if(h.kind==body_link::STATUS && n==sizeof(h)) {
            body_link::Status reply; reply.header=h; reply.header.kind=body_link::STATUS_REPLY;
            reply.states=states; reply.commands=commands; reply.releases=releases;
            ::send(fd,&reply,sizeof(reply),0);
        }
    }
    input.CloseChannel(); ::close(fd);
    std::cout<<"{\"states\":"<<states.load()<<",\"commands\":"<<commands
             <<",\"releases\":"<<releases<<"}"<<std::endl;
}
