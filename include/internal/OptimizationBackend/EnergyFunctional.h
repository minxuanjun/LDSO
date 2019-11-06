#pragma once
#ifndef LDSO_ENERGY_FUNCTIONAL_H_
#define LDSO_ENERGY_FUNCTIONAL_H_

#include "NumTypes.h"
#include "internal/FrameHessian.h"
#include "internal/PointHessian.h"
#include "internal/CalibHessian.h"
#include "internal/OptimizationBackend/AccumulatedTopHessian.h"
#include "internal/OptimizationBackend/AccumulatedSCHessian.h"

namespace ldso
{
namespace internal
{

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
class EnergyFunctional
{
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

    MatXX HM = MatXX::Zero(CPARS, CPARS);   //边缘化的先验 frame-frame H matrix
    VecX bM = VecX::Zero(CPARS);    //边缘化的先验 frame-frame b vector

    int resInA = 0, resInL = 0, resInM = 0;

    // LM 优化过程中保存上一时刻的舒尔补后Hessian block, b和优化变量的状态
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
    VecX getStitchedDeltaF() const
    {
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
    /**
     * \brief 计算由激活点构造的视觉残差的hessian block
     * @param H
     * @param b
     * @param MT
     */
    void accumulateAF_MT(MatXX &H, VecX &b, bool MT);
    /**
     * \brief 计算由固定线性点的激活点构造的视觉残差的hessian block
     * @param H
     * @param b
     * @param MT
     */
    void accumulateLF_MT(MatXX &H, VecX &b, bool MT);
    /**
     * \brief 计算视觉残差的shur completion
     * @param H
     * @param b
     * @param MT
     */
    void accumulateSCF_MT(MatXX &H, VecX &b, bool MT);

    /**
     * \brief 分块求解点的光度误差能量
     * @param min
     * @param max
     * @param stats 状态变量输出值
     * @param tid 线程的id
     */
    void calcLEnergyPt(int min, int max, Vec10 *stats, int tid);

    /**
     * TODO:感觉我在胡说,希望大佬能够解答疑惑
     * \brief TODO: 实现零空间边缘化
     * 数学表达形式
     * f(x)^2 = ||Jx*delta_x - b||^2 + ||(Jx*delta_x)^T*delta_x||^2
     * Jx^T(Jx*delta_x -b) + (Jx*delta_x)^T*delta_x
     *
     * delta_x 的维度为8n, delta_null 维度为7 即(全局的旋转+平移+尺度)
     * Jx的维度为[8n, 8n], J_null的维度[8n, 7]
     * 将delta_null边缘化得到
     * H_schur = Jx^TJx - Jx^T*J_x*J_null* (J_null^T * J_x^T * J_x*J_null)^{-1}*J_null^T*J_x^T*Jx
     *         = H - H*J_null*(J_null^T*H*J_null)^{-1}*J_null^T*H
     *         = H(I - J_null(J_null^T*H*J_null)^{-1}*J_null^T*H)
     *         =
     *         =
     * b_shcur = Jx^T*b - Jx^T * J_x* J_null* (J_null^T*J_null)^{-1} * J_null^T*b
     *         = Jx^T*b(I - J_null* (J_null^T*J_null)^{-1} * J_null^T)
     * notice: J_null 在函数具体实现中用是用N表示的,
     * J_null* (J_null^T*J_null)^{-1} * J_null^T 在函数实现中用N*(Npi)^T表示
     * @param b
     * @param H
     */
    void orthogonalize(VecX *b, MatXX *H);

    // don't use shared_ptr to handle dynamic arrays
    // 存储由于绝对位姿当前状态对线性点的偏移量造成相对位姿的增量,计算公式如下所示
    // delta_th = Ad(t) * delta_t + Ad(h) * delta_h * delta_h
    Mat18f *adHTdeltaF = nullptr;

    // 相机位姿和光度仿射变换系数的伴随矩阵, double型存储
    // !!! notice 需要乘以SCALE_*,详细解释见EnergyFunctional.cc中的函数esetAdjointsF
    Mat88 *adHost = nullptr;    // arrays of adjoints, adHost = -Adj(HostToTarget)^T
    Mat88 *adTarget = nullptr;
    // 相机位姿和光度仿射变换系数的伴随矩阵,flaot型存储
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
