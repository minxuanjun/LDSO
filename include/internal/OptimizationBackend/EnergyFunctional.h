#pragma once
#ifndef LDSO_ENERGY_FUNCTIONAL_H_
#define LDSO_ENERGY_FUNCTIONAL_H_

#include "NumTypes.h"
#include "internal/FrameHessian.h"
#include "internal/PointHessian.h"
#include "internal/CalibHessian.h"
#include "internal/OptimizationBackend/AccumulatedTopHessian.h"
#include "internal/OptimizationBackend/AccumulatedSCHessian.h"

namespace ldso {
    namespace internal {

        extern bool EFAdjointsValid;
        extern bool EFIndicesValid;
        extern bool EFDeltaValid;

        /**
         * The overall interface of optimization
         * The FullSystem class will hold an instance of EnergyFunctional, and perform optimization steps through this interface.
         *
         * I moved all the EFrame/EFPoint/EFResidual data into FrameHessian/PointHessian/PointFrameResidual
         * because I think the are the same.
         *
         * I don't know why dso designs such an optimization backend beside the fullSystem and keep its own frames,
         * residuals, points which are already stored in FullSystem class. Here you still need to add/delete/margin
         * all the stuffs and keep them synhonized with FullSystem. Looks not good. Maybe for some historical reasons?
         *
         */
        class EnergyFunctional {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            // only friend can see your private
            friend class AccumulatedTopHessian;

            friend class AccumulatedTopHessianSSE;

            friend class AccumulatedSCHessian;

            friend class AccumulatedSCHessianSSE;

            friend class PointHessian;

            friend class FrameHessian;

            friend class CalibHessian;

            friend class PointFrameResidual;

            friend class FeatureObsResidual;

            EnergyFunctional();

            ~EnergyFunctional();

            /**
             * insert a point-frame residual
             * @param r
             */
            void insertResidual(shared_ptr<PointFrameResidual> r);

            /**
             * insert a feature observation residual
             * @param r
             */
            // void insertResidual( shared_ptr<FeatureObsResidual> r );

            /**
             * add a single frame
             * @param fh
             * @param Hcalib
             */
            void insertFrame(shared_ptr<FrameHessian> fh, shared_ptr<CalibHessian> Hcalib);

            /**
             * drop a point-frame residual
             * @param r
             */
            void dropResidual(shared_ptr<PointFrameResidual> r);

            // void dropResidual( shared_ptr<FeatureObsResidual> r);

            /**
             * marginalize a given frame
             * @param fh
             */
            void marginalizeFrame(shared_ptr<FrameHessian> fh);

            /**
             * remove a point, delete all the residuals about it.
             * @param ph
             */
            void removePoint(shared_ptr<PointHessian> ph);

            /**
             * Marg all points to compute Bundle Adjustment
             */
            void marginalizePointsF();

            /**
             * remove the points marked as PS_DROP
             */
            void dropPointsF();

            /**
             * solve the all system by Gauss-Newton or LM iteration
             * GN or LM is defined in the bit of setting_solverMode
             * @param iteration
             * @param lambda
             * @param HCalib
             */
            void solveSystemF(int iteration, double lambda, shared_ptr<CalibHessian> HCalib);

            /**
             * compute frame-frame prior energy, F indicate frame, this is used to calculate frame-frame marginalization
             * @return
             */
            double calcMEnergyF();

            /**
             * compute point energy with multi-threading, L indicate point, MT indicate multi-threading
             * @return
             */
            double calcLEnergyF_MT();

            /**
             * compute the feature energy
             */
            // double calcLEnergyFeat();

            /**
             * create the indecies of H and b
             */
            void makeIDX();

            /**
             * don't know what is deltaF ...
             * \brief 1.计算当前位姿相对于线性点出的相对位姿的偏移, 更新pose, 相机内参,3d点到当前状态的的偏移delta
             * @param HCalib
             */
            void setDeltaF(shared_ptr<CalibHessian> HCalib);

            /**
             * set the adjoints of frames
             * @param Hcalib
             */
            void setAdjointsF(shared_ptr<CalibHessian> Hcalib);

            // all related frames
            std::vector<shared_ptr<FrameHessian>> frames;
            int nPoints = 0, nFrames = 0, nResiduals = 0;

            MatXX HM = MatXX::Zero(CPARS, CPARS);   // frame-frame H matrix
            VecX bM = VecX::Zero(CPARS);    // frame-frame b vector

            int resInA = 0, resInL = 0, resInM = 0;

            MatXX lastHS;
            VecX lastbS;
            VecX lastX;

            std::vector<VecX> lastNullspaces_forLogging;
            std::vector<VecX> lastNullspaces_pose;
            std::vector<VecX> lastNullspaces_scale;
            std::vector<VecX> lastNullspaces_affA;
            std::vector<VecX> lastNullspaces_affB;

            IndexThreadReduce<Vec10> *red = nullptr;  // passed by full system

            /**
             * connectivity map, the higher 32 bit of the key is host frame's id, and the lower is target frame's id
             */
            std::map<uint64_t,
                    Eigen::Vector2i,
                    std::less<uint64_t>,
                    Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>
            > connectivityMap;

        private:
            /// I really don't know what are they doing in the private functions
            // 获取camera, pose和 affine_ab的当前状态相对与线性点状态的偏移，主要用于更新先验的b
            VecX getStitchedDeltaF() const {
                VecX d = VecX(CPARS + nFrames * 8);
                d.head<CPARS>() = cDeltaF.cast<double>();
                for (int h = 0; h < nFrames; h++)
                    d.segment<8>(CPARS + 8 * h) = frames[h]->delta;
                return d;
            }

            /**
             * substitute the variable x into frameHessians
             */
            void resubstituteF_MT(const VecX &x, shared_ptr<CalibHessian> HCalib, bool MT);

            /**
             * substitute the variable xc into pointHessians
             */
            void resubstituteFPt(const VecCf &xc, Mat18f *xAd, int min, int max, Vec10 *stats,
                                 int tid);

            void accumulateAF_MT(MatXX &H, VecX &b, bool MT);

            void accumulateLF_MT(MatXX &H, VecX &b, bool MT);

            void accumulateSCF_MT(MatXX &H, VecX &b, bool MT);

            /**
             * \brief 分块求解点的光度误差能量
             * @param min
             * @param max
             * @param stats 状态变量输出值
             * @param tid 线程的id
             */
            void calcLEnergyPt(int min, int max, Vec10 *stats, int tid);

            void orthogonalize(VecX *b, MatXX *H);

            // don't use shared_ptr to handle dynamic arrays
            Mat18f *adHTdeltaF = nullptr;

            Mat88 *adHost = nullptr;    // arrays of adjoints, adHost = -Adj(HostToTarget)^T
            Mat88 *adTarget = nullptr;

            Mat88f *adHostF = nullptr;
            Mat88f *adTargetF = nullptr;

            VecC cPrior;
            VecCf cDeltaF;  // camera intrinsic change
            VecCf cPriorF;

            shared_ptr<AccumulatedTopHessianSSE> accSSE_top_L;
            shared_ptr<AccumulatedTopHessianSSE> accSSE_top_A;
            shared_ptr<AccumulatedSCHessianSSE> accSSE_bot;

            std::vector<shared_ptr<PointHessian>> allPoints;
            std::vector<shared_ptr<PointHessian>> allPointsToMarg;

            float currentLambda = 0;
        };
    }

}

#endif // LDSO_ENERGY_FUNCTIONAL_H_
