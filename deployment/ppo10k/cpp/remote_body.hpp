#pragma once
#include "body_protocol.hpp"
#include <arpa/inet.h>
#include <sys/socket.h>
#include <poll.h>
#include <unistd.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

class RemoteBody {
    int socket_=-1;
    std::atomic<bool> running_{true};
    std::atomic<uint64_t> sequence_{0};
    std::thread receiver_;
    std::mutex mutex_;
    std::condition_variable changed_;
    body_link::State state_{};
    body_link::Reply reply_{};
    body_link::Status status_{};
    uint64_t received_=0;

    template<class T> void send(const T& packet) {
        if(::send(socket_,&packet,sizeof(packet),0)!=sizeof(packet))
            throw std::runtime_error("PC2 body UDP send failed");
    }
    void receive() {
        while(running_) {
            pollfd p{socket_,POLLIN,0};
            if(::poll(&p,1,100)<=0)continue;
            std::array<char,2048> bytes;
            auto n=::recv(socket_,bytes.data(),bytes.size(),MSG_DONTWAIT);
            if(n<static_cast<ssize_t>(sizeof(body_link::Header)))continue;
            body_link::Header h;
            std::memcpy(&h,bytes.data(),sizeof(h));
            if(!body_link::valid(h))continue;
            std::lock_guard<std::mutex> lock(mutex_);
            if(h.kind==body_link::STATE && n==sizeof(state_)) {
                std::memcpy(&state_,bytes.data(),sizeof(state_)); ++received_;
            } else if(h.kind==body_link::RELEASE_REPLY && n==sizeof(reply_)) {
                std::memcpy(&reply_,bytes.data(),sizeof(reply_));
            } else if(h.kind==body_link::STATUS_REPLY && n==sizeof(status_)) {
                std::memcpy(&status_,bytes.data(),sizeof(status_));
            }
            changed_.notify_all();
        }
    }
public:
    RemoteBody(const std::string& host,int peer,int local) {
        socket_=::socket(AF_INET,SOCK_DGRAM,0);
        sockaddr_in own{}; own.sin_family=AF_INET; own.sin_port=htons(local);
        sockaddr_in remote{}; remote.sin_family=AF_INET; remote.sin_port=htons(peer);
        if(socket_<0 || inet_pton(AF_INET,host.c_str(),&remote.sin_addr)!=1 ||
           ::bind(socket_,reinterpret_cast<sockaddr*>(&own),sizeof(own))<0 ||
           ::connect(socket_,reinterpret_cast<sockaddr*>(&remote),sizeof(remote))<0) {
            if(socket_>=0)::close(socket_);
            throw std::runtime_error("PC2 body UDP bind/connect failed");
        }
        receiver_=std::thread([this]{receive();});
    }
    ~RemoteBody() { running_=false; if(receiver_.joinable())receiver_.join(); ::close(socket_); }
    void wait_state() {
        std::unique_lock<std::mutex> lock(mutex_);
        if(!changed_.wait_for(lock,std::chrono::seconds(10),[this]{return received_>0;}))
            throw std::runtime_error("No A2W state from PC2 body bridge");
    }
    std::pair<body_link::State,uint64_t> snapshot() {
        std::lock_guard<std::mutex> lock(mutex_); return {state_,received_};
    }
    void command(const body_link::Command& target) {
        auto packet=target; packet.header.kind=body_link::COMMAND;
        packet.header.sequence=++sequence_; send(packet);
    }
    void release_native() {
        body_link::Header request; request.kind=body_link::RELEASE;
        request.sequence=++sequence_; send(request);
        std::unique_lock<std::mutex> lock(mutex_);
        if(!changed_.wait_for(lock,std::chrono::seconds(10),[&]{return reply_.header.sequence==request.sequence;}))
            throw std::runtime_error("PC2 motion release reply timed out");
        if(reply_.result)throw std::runtime_error("PC2 native motion release failed: "+std::to_string(reply_.result));
    }
    body_link::Status status() {
        body_link::Header request; request.kind=body_link::STATUS;
        request.sequence=++sequence_; send(request);
        std::unique_lock<std::mutex> lock(mutex_);
        if(!changed_.wait_for(lock,std::chrono::seconds(2),[&]{return status_.header.sequence==request.sequence;}))
            throw std::runtime_error("PC2 bridge status reply timed out");
        return status_;
    }
};
