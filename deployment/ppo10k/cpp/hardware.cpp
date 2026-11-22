#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree_arm_sdk/control/unitreeArm.h>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <stdexcept>
#include "remote_body.hpp"
#include "motion_release.hpp"
#include "arm_preparation.hpp"

namespace py = pybind11;
using namespace unitree::robot;
using namespace UNITREE_ARM;
using Clock = std::chrono::steady_clock;
using A22 = std::array<double, 22>;
using A6 = arm_preparation::A6;
using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;

// HG checksum over all 32-bit words except crc, as in SDK2 examples.
uint32_t crc(const LowCmd& message) {
    uint32_t value = 0xffffffffu;
    auto bytes = reinterpret_cast<const unsigned char*>(&message);
    for (size_t i = 0; i < sizeof(message) - 4; i += 4) {
        uint32_t word;
        std::memcpy(&word, bytes + i, 4);
        for (uint32_t bit = 0x80000000u; bit; bit >>= 1) {
            bool top = value & 0x80000000u;
            value <<= 1;
            if (top) value ^= 0x04c11db7u;
            if (word & bit) value ^= 0x04c11db7u;
        }
    }
    return value;
}

class Hardware {
    std::unique_ptr<RemoteBody> remote_body_;
    std::unique_ptr<unitreeArm> arm_;
    ChannelPublisherPtr<LowCmd> publisher_;
    ChannelSubscriberPtr<LowState> subscriber_;
    std::mutex mutex_;
    std::mutex arm_io_mutex_;
    std::atomic<bool> arm_homing_{false};
    // T leaves the arm under the SDK's native home hold. Body commands must
    // never re-enable the LOWCMD arm writer; only explicit preparation may.
    std::atomic<bool> arm_sdk_hold_{false};
    std::atomic<uint64_t> arm_generation_{0};
    std::condition_variable received_;
    bool body_received_ = false, enabled_ = false, started_ = false;
    std::atomic<bool> running_{false};
    std::thread servo_;
    A22 q_{}, dq_{}, target_q_{}, target_dq_{}, kp_{}, kd_{};
    std::array<double, 4> quaternion_{1., 0., 0., 0.};
    std::array<double, 3> gyro_{};
    uint8_t machine_ = 0;
    double gripper_ = 0., kp_factor_, kd_factor_, gripper_kp_, gripper_kd_;
    bool has_gripper_;
    A6 arm_ddq_{}, prepare_from_{}, prepare_velocity_{}, prepare_to_{}, arm_min_{}, arm_max_{};
    double arm_ff_weight_ = 0., prepare_duration_ = 3.;
    bool arm_preparation_ = false;
    Clock::time_point prepare_start_;

    void body_state(const void* data) {
        const auto& s = *static_cast<const LowState*>(data);
        std::lock_guard<std::mutex> lock(mutex_);
        machine_ = s.mode_machine();
        for (int i = 0; i < 16; ++i) {
            int p = i < 12 ? i : i + 6;
            q_[p] = s.motor_state()[i].q();
            dq_[p] = s.motor_state()[i].dq();
        }
        for (int i = 0; i < 4; ++i) quaternion_[i] = s.imu_state().quaternion()[i];
        for (int i = 0; i < 3; ++i) gyro_[i] = s.imu_state().gyroscope()[i];
        body_received_ = true;
        received_.notify_one();
    }

    void tick() {
        const auto arm_generation = arm_generation_.load();
        A22 q, dq, kp, kd;
        double gripper, gripper_kp, gripper_kd;
        uint8_t machine;
        bool enabled;
        A6 arm_ddq;
        double arm_ff_weight;
        bool preparing;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            q = target_q_; dq = target_dq_; kp = kp_; kd = kd_;
            gripper = gripper_; machine = machine_; enabled = enabled_;
            gripper_kp = gripper_kp_; gripper_kd = gripper_kd_;
            arm_ddq = arm_ddq_; arm_ff_weight = arm_ff_weight_;
            preparing = arm_preparation_;
            if (enabled && preparing) {
                auto s = arm_preparation::sample(prepare_from_, prepare_velocity_, prepare_to_,
                    std::chrono::duration<double>(Clock::now()-prepare_start_).count(), prepare_duration_);
                for (int i=0;i<6;++i) { q[12+i]=s.q[i]; dq[12+i]=s.dq[i]; }
                arm_ddq=s.ddq;
            }
        }
        if (enabled) {
          if (remote_body_) {
            body_link::Command command;
            for (int i=0;i<16;++i) {
                int p=i<12?i:i+6;
                command.q[i]=q[p]; command.dq[i]=dq[p];
                command.kp[i]=kp[p]; command.kd[i]=kd[p];
            }
            try { remote_body_->command(command); }
            catch(const std::exception& error) { std::cerr<<error.what()<<std::endl; running_=false; }
          } else {
            LowCmd msg{};
            msg.mode_pr() = 0;
            msg.mode_machine() = machine;
            for (int i = 0; i < 16; ++i) {
                int p = i < 12 ? i : i + 6;
                auto& motor = msg.motor_cmd()[i];
                motor.mode() = 1;
                motor.q() = q[p]; motor.dq() = dq[p];
                motor.kp() = kp[p]; motor.kd() = kd[p]; motor.tau() = 0.;
            }
            msg.crc() = crc(msg);
            publisher_->Write(msg);
          }
        }
        // Body publishing continues while the official SDK owns arm homing.
        std::lock_guard<std::mutex> arm_lock(arm_io_mutex_);
        if (arm_homing_ || arm_sdk_hold_ || arm_generation != arm_generation_.load()) return;
        std::vector<double> arm_kp(6), arm_kd(6);
        for (int i = 0; i < 6; ++i) {
            arm_->q[i] = q[i + 12];
            arm_->qd[i] = dq[i + 12];
            arm_->tau[i] = 0.;
            arm_kp[i] = kp[i + 12] / kp_factor_;
            arm_kd[i] = kd[i + 12] / kd_factor_;
        }
        auto* model=arm_->_ctrlComp->armModel;
        // ACT servoing evaluates dynamics at the same legal target that is
        // sent to LOWCMD, including when the target comes from PPO.
        if (enabled && (preparing || arm_ff_weight>0.)) model->jointProtect(arm_->q, arm_->qd);
        if (enabled && arm_ff_weight>0.) {
            Vec6 acc;
            for(int i=0;i<6;++i) acc[i]=arm_ddq[i];
            arm_->tau = arm_ff_weight * model->inverseDynamics(arm_->q, arm_->qd, acc, Vec6::Zero());
        }
        arm_->lowcmd->setControlGain(arm_kp, arm_kd);
        arm_->setArmCmd(arm_->q, arm_->qd, arm_->tau);
        if (has_gripper_) {
            arm_->lowcmd->setGripperGain(enabled ? gripper_kp / kp_factor_ : 0.,
                                        enabled ? gripper_kd / kd_factor_ : 0.);
            arm_->gripperQ = gripper;
            arm_->gripperW = arm_->gripperTau = 0.;
            arm_->setGripperCmd(gripper, 0., 0.);
        } else {
            arm_->lowcmd->setGripperZeroGain();
        }
        arm_->sendRecv();
        auto aq = arm_->lowstate->getQ();
        auto av = arm_->lowstate->getQd();
        std::lock_guard<std::mutex> lock(mutex_);
        for (int i = 0; i < 6; ++i) { q_[12+i] = aq[i]; dq_[12+i] = av[i]; }
    }

public:
    Hardware(std::string interface, int domain, std::string host, int peer, int local,
             bool has_gripper, double kp_factor, double kd_factor,
             double gripper_kp, double gripper_kd,
             std::string body_host, int body_peer, int body_local)
        : kp_factor_(kp_factor), kd_factor_(kd_factor), gripper_kp_(gripper_kp),
          gripper_kd_(gripper_kd), has_gripper_(has_gripper) {
        if(body_host.empty()) {
            ChannelFactory::Instance()->Init(domain, interface);
            publisher_ = std::make_shared<ChannelPublisher<LowCmd>>("rt/lowcmd");
            publisher_->InitChannel();
            subscriber_ = std::make_shared<ChannelSubscriber<LowState>>("rt/lowstate");
            subscriber_->InitChannel([this](const void* s) { body_state(s); }, 1);
        } else {
            remote_body_=std::make_unique<RemoteBody>(body_host,body_peer,body_local);
        }
        auto* components = new CtrlComponents();
        components->dt = .002;
        components->udp = new UDPPort(host, peer, local, RECVSTATE_LENGTH, BlockYN::NO, 100);
        components->armModel = has_gripper
            ? new Z1Model(Vec3(.0382, 0., 0.), .80225, Vec3(.0037, .0014, -.0003),
                          Vec3(.00057593, .00099960, .00106337).asDiagonal())
            : new Z1Model();
        arm_ = std::make_unique<unitreeArm>(components);
        const auto lower=components->armModel->getJointQMin();
        const auto upper=components->armModel->getJointQMax();
        for(int i=0;i<6;++i) { arm_min_[i]=lower[i]; arm_max_[i]=upper[i]; }
    }

    void start(bool home_on_start = true) {
        // First physical state supplies mode_machine and the startup posture.
        if(remote_body_) {
            remote_body_->wait_state();
        } else {
            std::unique_lock<std::mutex> lock(mutex_);
            received_.wait(lock, [this] { return body_received_; });
        }
        // A2W exposes the motion_switcher service through this SDK2 client.
        // The operator has already crouched the robot before starting us.
        if(remote_body_) {
            remote_body_->release_native();
        } else {
            auto result=release_native_motion();
            if(result)throw std::runtime_error("Native motion release failed: "+std::to_string(result));
        }
        std::cout << "[A2W] Native motion released." << std::endl;
        arm_->lowcmd->setZeroKp();
        arm_->lowcmd->setZeroKd();
        arm_->lowcmd->setZeroTau();
        arm_->lowcmd->setZeroDq();
        arm_->q.setZero(); arm_->qd.setZero(); arm_->tau.setZero();
        arm_->gripperQ = arm_->gripperW = arm_->gripperTau = 0.;
        arm_->sendRecvThread->start();
        arm_->setFsm(ArmFSMState::PASSIVE);
        // Shared hardware startup for keyboard and VR. Official home motion,
        // not encoder calibration. Finish before accepting preparation input.
        if (home_on_start) {
            std::cout << "[Z1] backToStart begin." << std::endl;
            arm_->backToStart();
        }
        const Vec6 home = arm_->lowstate->getQ();
        const double home_gripper = arm_->lowstate->getGripperQ();
        std::cout << (home_on_start ? "[Z1] backToStart complete; q_rad=["
                                   : "[Z1] Startup feedback (homing skipped); q_rad=[");
        for (int i=0;i<6;++i) std::cout << (i ? "," : "") << home[i];
        std::cout << "]." << std::endl;
        // Seed LOWCMD from the measured home pose, just as in the arm-only
        // cycle. No preparation trajectory or policy is started here.
        arm_->setFsm(ArmFSMState::PASSIVE);
        arm_->q = home; arm_->qd.setZero(); arm_->tau.setZero();
        arm_->gripperQ = home_gripper; arm_->gripperW = arm_->gripperTau = 0.;
        arm_->lowcmd->setZeroKp(); arm_->lowcmd->setZeroKd();
        arm_->lowcmd->setGripperZeroGain();
        arm_->setArmCmd(arm_->q, arm_->qd, arm_->tau);
        arm_->setGripperCmd(home_gripper, 0., 0.);
        arm_->setFsm(ArmFSMState::LOWCMD);
        arm_->sendRecvThread->shutdown();
        { std::lock_guard<std::mutex> lock(mutex_);
          auto aq = arm_->lowstate->getQ();
          auto av = arm_->lowstate->getQd();
          for (int i=0; i<6; ++i) { q_[12+i] = aq[i]; dq_[12+i] = av[i]; }
          target_q_ = q_; gripper_ = arm_->lowstate->getGripperQ(); }
        std::cout << "[Z1] Startup finished; waiting for operator Z." << std::endl;
        started_ = true;
        running_ = true;
        servo_ = std::thread([this] {
            auto next = Clock::now();
            while (running_) {
                tick();
                next += std::chrono::microseconds(2000);
                std::this_thread::sleep_until(next);
                if (Clock::now() > next + std::chrono::microseconds(2000)) next = Clock::now();
            }
        });
    }

    auto state() {
        std::lock_guard<std::mutex> lock(mutex_);
        if(remote_body_) {
            const auto s=remote_body_->snapshot().first;
            for(int i=0;i<16;++i) {
                int p=i<12?i:i+6; q_[p]=s.q[i]; dq_[p]=s.dq[i];
            }
            quaternion_=s.quaternion; gyro_=s.gyro; machine_=s.machine;
        }
        return std::make_tuple(q_, dq_, quaternion_, gyro_);
    }

    A6 begin_arm_preparation(const A6& from, const A6& velocity, const A6& to, double duration) {
        if (arm_sdk_hold_) resume_arm_lowcmd();
        std::lock_guard<std::mutex> lock(mutex_);
        for(int i=0;i<6;++i) prepare_to_[i]=std::clamp(to[i],arm_min_[i],arm_max_[i]);
        prepare_from_=from; prepare_velocity_=velocity; prepare_duration_=duration;
        prepare_start_=Clock::now();
        return prepare_to_;
    }

    A6 home_arm() {
        arm_homing_ = true;
        ++arm_generation_;
        // Let any in-flight LOWCMD packet finish before SDK thread takeover.
        { std::lock_guard<std::mutex> lock(arm_io_mutex_); }
        arm_->sendRecvThread->start();
        std::cout << "[Z1] T backToStart begin; body servo remains active." << std::endl;
        arm_->setFsm(ArmFSMState::PASSIVE);
        arm_->backToStart();
        // Stop the SDK sender before replacing its old LOWCMD buffers. Do not
        // re-enter LOWCMD here: JOINTCTRL holds the official home endpoint.
        arm_->sendRecvThread->shutdown();
        const Vec6 home = arm_->lowstate->getQ();
        const double gripper = arm_->lowstate->getGripperQ();
        arm_->q=home; arm_->qd.setZero(); arm_->tau.setZero();
        arm_->gripperQ=gripper; arm_->gripperW=arm_->gripperTau=0.;
        arm_->lowcmd->setZeroKp(); arm_->lowcmd->setZeroKd();
        arm_->lowcmd->setGripperZeroGain();
        arm_->setArmCmd(home, Vec6::Zero(), Vec6::Zero());
        arm_->setGripperCmd(gripper, 0., 0.);
        // CtrlComponents retains a separate serialized payload. Clearing only
        // lowcmd leaves the previous extended pose in this wire buffer.
        auto& wire=arm_->_ctrlComp->sendCmd;
        wire.track=false;
        std::memset(&wire.valueUnion, 0, sizeof(wire.valueUnion));
        for(int i=0;i<6;++i) wire.valueUnion.jointCmd[i].Pos=home[i];
        wire.valueUnion.jointCmd[6].Pos=gripper;
        arm_sdk_hold_=true;
        arm_->sendRecvThread->start();
        arm_->setFsm(ArmFSMState::JOINTCTRL);
        A6 result;
        { std::lock_guard<std::mutex> lock(mutex_);
          for(int i=0;i<6;++i) {
              result[i]=home[i]; q_[12+i]=home[i]; dq_[12+i]=0.;
              target_q_[12+i]=home[i]; target_dq_[12+i]=0.;
          }
          gripper_=gripper; arm_preparation_=false; arm_ddq_.fill(0.);
          arm_ff_weight_=0.;
          prepare_from_=result; prepare_to_=result; prepare_velocity_.fill(0.);
          for(int i=12;i<18;++i) { kp_[i]=0.; kd_[i]=0.; } }
        ++arm_generation_;
        arm_homing_=false;
        std::cout << "[Z1] T complete: native JOINTCTRL home hold; LOWCMD arm writer disabled." << std::endl;
        return result;
    }

    void resume_arm_lowcmd() {
        // Called only by an explicit Z/R preparation, never by crouch ticks.
        arm_homing_=true;
        ++arm_generation_;
        { std::lock_guard<std::mutex> lock(arm_io_mutex_); }
        arm_->setFsm(ArmFSMState::PASSIVE);
        arm_->sendRecvThread->shutdown();
        const Vec6 measured=arm_->lowstate->getQ();
        arm_->q=measured; arm_->qd.setZero(); arm_->tau.setZero();
        arm_->setArmCmd(measured, Vec6::Zero(), Vec6::Zero());
        arm_->setGripperCmd(arm_->lowstate->getGripperQ(), 0., 0.);
        arm_->sendRecvThread->start();
        arm_->setFsm(ArmFSMState::LOWCMD);
        arm_->sendRecvThread->shutdown();
        { std::lock_guard<std::mutex> lock(mutex_);
          for(int i=0;i<6;++i) {
              q_[12+i]=measured[i]; dq_[12+i]=0.;
              target_q_[12+i]=measured[i]; target_dq_[12+i]=0.;
          }
          arm_preparation_=false; arm_ddq_.fill(0.); arm_ff_weight_=0.; }
        ++arm_generation_;
        arm_sdk_hold_=false;
        arm_homing_=false;
    }

    void arm_damping() {
        if (arm_sdk_hold_) arm_->setFsm(ArmFSMState::PASSIVE);
    }

    void command(const A22& q, const A22& dq, const A22& kp, const A22& kd, double gripper,
                 const A6& arm_ddq, double arm_ff_weight, bool preparing) {
        std::lock_guard<std::mutex> lock(mutex_);
        target_q_ = q; target_dq_ = dq; kp_ = kp; kd_ = kd;
        gripper_ = gripper; enabled_ = true;
        arm_ddq_=arm_ddq; arm_ff_weight_=arm_ff_weight; arm_preparation_=preparing;
        if (arm_sdk_hold_) {
            // Body-only commands cannot refill an arm target queue while home
            // is owned by the SDK. Preparation explicitly releases this hold.
            for(int i=12;i<18;++i) {
                target_q_[i]=q_[i]; target_dq_[i]=0.; kp_[i]=0.; kd_[i]=0.;
            }
            arm_ddq_.fill(0.); arm_ff_weight_=0.; arm_preparation_=false;
        }
    }

    void stop() {
        if (!started_) return;
        // User-requested exit: publish zero drive torque/position gain and damping.
        { std::lock_guard<std::mutex> lock(mutex_);
          kp_.fill(0.); target_dq_.fill(0.);
          for (int i=0; i<18; ++i) kd_[i] = 3.;
          for (int i=18; i<22; ++i) kd_[i] = .6;
          gripper_kp_ = 0.; arm_ff_weight_=0.; arm_preparation_=false; }
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        running_ = false;
        servo_.join();
        arm_->sendRecvThread->start();
        arm_->setFsm(ArmFSMState::PASSIVE);
        arm_->sendRecvThread->shutdown();
        started_ = false;
    }

    ~Hardware() {
        stop();
        if(subscriber_)subscriber_->CloseChannel();
        if(publisher_)publisher_->CloseChannel();
    }
};

class BodyControl : public RemoteBody {
public:
    using RemoteBody::RemoteBody;
};

PYBIND11_MODULE(_hardware, m) {
    py::class_<Hardware>(m, "Hardware")
        .def(py::init<std::string,int,std::string,int,int,bool,double,double,double,double,std::string,int,int>())
        .def("start", &Hardware::start, py::arg("home_on_start")=true, py::call_guard<py::gil_scoped_release>())
        .def("home_arm", &Hardware::home_arm, py::call_guard<py::gil_scoped_release>())
        .def("arm_damping", &Hardware::arm_damping, py::call_guard<py::gil_scoped_release>())
        .def("state", &Hardware::state)
        .def("begin_arm_preparation", &Hardware::begin_arm_preparation, py::call_guard<py::gil_scoped_release>())
        .def("command", &Hardware::command, py::arg("q"), py::arg("dq"), py::arg("kp"),
             py::arg("kd"), py::arg("gripper"), py::arg("arm_ddq")=A6{},
             py::arg("arm_ff_weight")=0., py::arg("arm_preparation")=false)
        .def("stop", &Hardware::stop, py::call_guard<py::gil_scoped_release>());
    // Read-only Python surface: no command or ReleaseMode method exposed.
    py::class_<RemoteBody>(m,"BodyReader")
        .def(py::init<std::string,int,int>())
        .def("wait_state",&RemoteBody::wait_state,py::call_guard<py::gil_scoped_release>())
        .def("state",[](RemoteBody& body) {
            auto [s,n]=body.snapshot();
            py::dict d;
            d["q"]=s.q; d["dq"]=s.dq; d["quaternion"]=s.quaternion; d["gyro"]=s.gyro;
            d["tick"]=s.tick; d["source_monotonic_ns"]=s.source_monotonic_ns;
            d["messages"]=n; d["machine"]=s.machine; return d;
        })
        .def("status",[](RemoteBody& body) {
            auto s=body.status();
            py::dict d; d["states"]=s.states; d["commands"]=s.commands; d["releases"]=s.releases;
            return d;
        });
    py::class_<BodyControl,RemoteBody>(m,"BodyControl")
        .def(py::init<std::string,int,int>())
        .def("release_native",&BodyControl::release_native,py::call_guard<py::gil_scoped_release>())
        .def("command",[](BodyControl& body,const std::array<double,16>& q,
                           const std::array<double,16>& dq,const std::array<double,16>& kp,
                           const std::array<double,16>& kd) {
            body_link::Command target; target.q=q; target.dq=dq; target.kp=kp; target.kd=kd;
            body.command(target);
        });
}
