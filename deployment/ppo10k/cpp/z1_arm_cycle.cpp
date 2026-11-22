// Arm-only cycle. No body transport, policy, IK or calibration API.
#include <unitree_arm_sdk/control/unitreeArm.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>
#include <nlohmann/json.hpp>
#include "arm_preparation.hpp"

using namespace UNITREE_ARM;
using Clock = std::chrono::steady_clock;
volatile std::sig_atomic_t interrupted = 0;

void print_q(const char* phase, const Vec6& q) {
    std::cout << "{\"phase\":\"" << phase << "\",\"q_rad\":[";
    for (int i=0; i<6; ++i) std::cout << (i ? "," : "") << q[i];
    std::cout << "]}" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "usage: z1_arm_cycle LOG_DIRECTORY CONFIG_JSON [HOLD_SECONDS]\n";
        return 2;
    }
    std::ifstream config_stream(argv[2]);
    const auto config = nlohmann::json::parse(config_stream);
    const double move_seconds = config.at("arm_ready_seconds").get<double>();
    const double hold_seconds = argc == 4 ? std::stod(argv[3]) : 10.;
    const auto kp_si=config.at("prepare_arm_kp").get<std::vector<double>>();
    const auto kd_si=config.at("prepare_arm_kd").get<std::vector<double>>();
    const double kp_factor=config.at("z1_kp_factor"), kd_factor=config.at("z1_kd_factor");
    if (config.at("prepare_arm_feedforward") != "inverse_dynamics")
        throw std::runtime_error("Arm cycle uses the preparation inverse-dynamics profile");
    std::signal(SIGINT, [](int) { interrupted = 1; });
    std::signal(SIGTERM, [](int) { interrupted = 1; });
    std::cout << std::setprecision(10);
    std::ofstream trace(std::string(argv[1])+"/trajectory.csv");
    trace << std::setprecision(10) << "phase,t";
    for (auto prefix : {"target_q", "actual_q", "actual_dq"})
        for (int i=1; i<=6; ++i) trace << ',' << prefix << i;
    for (auto prefix : {"ff_tau", "kp_wire", "kd_wire"})
        for (int i=1; i<=6; ++i) trace << ',' << prefix << i;
    trace << ",fsm,max_abs_error\n";
    Vec6 ready;
    for (int i=0; i<6; ++i) ready[i] = std::clamp(
        config.at("arm_ready_q").at(i).get<double>(),
        config.at("arm_prepare_joint_min").at(i).get<double>(),
        config.at("arm_prepare_joint_max").at(i).get<double>());
    unitreeArm arm(true);
    auto* model=arm._ctrlComp->armModel;
    Vec6 zero=Vec6::Zero();
    model->jointProtect(ready,zero);
    print_q("legal_prepare_target",ready);
    arm.lowcmd->setZeroKp();
    arm.lowcmd->setZeroKd();
    arm.lowcmd->setZeroTau();
    arm.lowcmd->setZeroDq();
    arm.sendRecvThread->start();
    arm.setFsm(ArmFSMState::PASSIVE);
    print_q("before_backToStart", arm.lowstate->getQ());
    std::cout << "{\"phase\":\"backToStart\"}" << std::endl;
    // Official homing trajectory, not calibration (no encoder zero rewrite).
    arm.backToStart();
    Vec6 home = arm.lowstate->getQ();
    print_q("backToStart_complete", home);
    model->jointProtect(home,zero);
    const double gripper = arm.lowstate->getGripperQ();
    arm.setFsm(ArmFSMState::PASSIVE);
    arm.q = home; arm.qd.setZero(); arm.tau.setZero();
    arm.gripperQ = gripper; arm.gripperW = arm.gripperTau = 0.;
    arm.lowcmd->setZeroKp(); arm.lowcmd->setZeroKd();
    arm.lowcmd->setGripperZeroGain();
    arm.setArmCmd(arm.q, arm.qd, arm.tau);
    arm.setGripperCmd(gripper, 0., 0.);
    if (!interrupted) arm.setFsm(ArmFSMState::LOWCMD);
    arm.sendRecvThread->shutdown();

    std::vector<double> kp(6), kd(6);
    auto phase = [&](const char* name, const Vec6& from, const Vec6& to,
                     double seconds) {
        std::cout << "{\"phase\":\"" << name << "\",\"seconds\":" << seconds << "}" << std::endl;
        const auto start = Clock::now();
        auto next = start;
        double peak_error = 0., final_error = 0.;
        arm_preparation::A6 from_array{}, to_array{}, initial_velocity{};
        for(int i=0;i<6;++i) {from_array[i]=from[i];to_array[i]=to[i];}
        while (!interrupted) {
            const double elapsed = std::chrono::duration<double>(Clock::now()-start).count();
            const double u = std::min(elapsed/seconds, 1.);
            const auto sample=arm_preparation::sample(from_array, initial_velocity, to_array, elapsed, seconds);
            Vec6 acceleration;
            for(int i=0;i<6;++i) {
                arm.q[i]=sample.q[i];arm.qd[i]=sample.dq[i];acceleration[i]=sample.ddq[i];
                kp[i]=kp_si.at(i)/kp_factor;kd[i]=kd_si.at(i)/kd_factor;
            }
            model->jointProtect(arm.q,arm.qd);
            arm.tau=model->inverseDynamics(arm.q,arm.qd,acceleration,Vec6::Zero());
            arm.lowcmd->setControlGain(kp,kd);
            arm.lowcmd->setGripperGain(config.at("gripper_kp_si").get<double>()/kp_factor,
                                      config.at("gripper_kd_si").get<double>()/kd_factor);
            arm.setArmCmd(arm.q,arm.qd,arm.tau);
            arm.setGripperCmd(gripper,0.,0.);
            arm.sendRecv();
            const Vec6 q = arm.lowstate->getQ(), dq = arm.lowstate->getQd();
            final_error = (q-arm.q).cwiseAbs().maxCoeff();
            peak_error = std::max(peak_error,final_error);
            trace << name << ',' << elapsed;
            for (int i=0;i<6;++i) trace << ',' << arm.q[i];
            for (int i=0;i<6;++i) trace << ',' << q[i];
            for (int i=0;i<6;++i) trace << ',' << dq[i];
            for(int i=0;i<6;++i) trace << ',' << arm.tau[i];
            for(int i=0;i<6;++i) trace << ',' << kp[i];
            for(int i=0;i<6;++i) trace << ',' << kd[i];
            trace << ',' << int(arm._ctrlComp->recvState.state) << ',' << final_error << '\n';
            if (u >= 1.) break;
            next += std::chrono::microseconds(2000);
            std::this_thread::sleep_until(next);
            if (Clock::now()>next+std::chrono::microseconds(2000)) next=Clock::now();
        }
        trace.flush();
        std::cout << "{\"phase\":\"" << name << "_complete\",\"peak_error_rad\":"
                  << peak_error << ",\"final_error_rad\":" << final_error << "}" << std::endl;
        print_q(name,arm.lowstate->getQ());
    };
    if (!interrupted) phase("prepare",home,ready,move_seconds);
    if (!interrupted) phase("hold",ready,ready,hold_seconds);
    if (!interrupted) phase("return_home",ready,home,move_seconds);
    arm.sendRecvThread->start();
    arm.setFsm(ArmFSMState::PASSIVE);
    arm.sendRecvThread->shutdown();
    std::cout << "{\"phase\":\"" << (interrupted ? "interrupted" : "complete")
              << "\",\"final_state\":\"PASSIVE\"}" << std::endl;
    return interrupted ? 130 : 0;
}
