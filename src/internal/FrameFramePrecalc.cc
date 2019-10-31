#include "internal/FrameFramePrecalc.h"

namespace ldso {
    namespace internal {

        void FrameFramePrecalc::Set(shared_ptr<FrameHessian> host, shared_ptr<FrameHessian> target,
                                    shared_ptr<CalibHessian> HCalib) {

            this->host = host;
            this->target = target;

            // 计算边缘化点出的相机的相机的相对位姿(用于求雅克比矩阵)
            SE3 leftToLeft_0 = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();
            PRE_RTll_0 = (leftToLeft_0.rotationMatrix()).cast<float>();
            PRE_tTll_0 = (leftToLeft_0.translation()).cast<float>();
            // 计算相机当前状态计算的相机的相对位姿
            SE3 leftToLeft = target->PRE_worldToCam * host->PRE_camToWorld;
            PRE_RTll = (leftToLeft.rotationMatrix()).cast<float>();
            PRE_tTll = (leftToLeft.translation()).cast<float>();
            // 相对位姿的平移距离的2范数
            distanceLL = leftToLeft.translation().norm();

            Mat33f K = Mat33f::Zero();
            K(0, 0) = HCalib->fxl();
            K(1, 1) = HCalib->fyl();
            K(0, 2) = HCalib->cxl();
            K(1, 2) = HCalib->cyl();
            K(2, 2) = 1;

            // 计算帧间投影时需要的一些参数KRK^{-1}, RK^{-1}, KR
            PRE_KRKiTll = K * PRE_RTll * K.inverse();
            PRE_RKiTll = PRE_RTll * K.inverse();
            PRE_KtTll = K * PRE_tTll;
            // 计算相对光度变换的采纳数
            PRE_aff_mode = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l(),
                                                       target->aff_g2l()).cast<float>();
            PRE_b0_mode = host->aff_g2l_0().b;
        }

    }
}