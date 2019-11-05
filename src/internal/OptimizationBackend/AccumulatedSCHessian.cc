#include "internal/OptimizationBackend/EnergyFunctional.h"
#include "internal/OptimizationBackend/AccumulatedSCHessian.h"
#include "internal/PointHessian.h"

namespace ldso
{

namespace internal
{

void AccumulatedSCHessianSSE::addPoint(shared_ptr<PointHessian> p, bool shiftPriorToZero, int tid)
{

    int ngoodres = 0;
    for (auto r : p->residuals)
        if (r->isActive())
            ngoodres++;

    if (ngoodres == 0) {
        p->HdiF = 0;
        p->bdSumF = 0;
        p->idepth_hessian = 0;
        p->maxRelBaseline = 0;
        return;
    }
    //!!!notice 3d点的priorF只有在初始化的的第一帧才有
    //step1: 计算Hdd, bd
    float H = p->Hdd_accAF + p->Hdd_accLF + p->priorF;  // point 的hessian Hdd是激活点的hessian + 线性化的hessian + 先验
    if (H < 1e-10) H = 1e-10; // 避免hessian值过小，造成数值求解的不稳定
    p->idepth_hessian = H;
    p->HdiF = 1.0 / H;
    p->bdSumF = p->bd_accAF + p->bd_accLF;
    if (shiftPriorToZero) p->bdSumF += p->priorF * p->deltaF; // bd 加先验的误差
    // step2: 计算Hcc_schur 和 bc_schur
    VecCf Hcd = p->Hcd_accAF + p->Hcd_accLF;
    // step2.1： 求Hcc的Schur complement 的Hcc_schur= Hcd*{Hdd}^{-1}*{Hcd}^T
    accHcc[tid].update(Hcd, Hcd, p->HdiF);
    // step2.2: 求bc的Schur complement bc_schur = Hcd*{Hdd}^{-1}*bd
    accbc[tid].update(Hcd, p->bdSumF * p->HdiF);

    assert(std::isfinite((float) (p->HdiF)));
    // step3: 计算 H_[relative_pose, relative_ab]_[relative_pose, relative_ab]
    // , H_[relative_pose, relative_ab]_c hessian block的Schur complement
    int nFrames2 = nframes[tid] * nframes[tid];


    // 每一个3d点的残差形式如下所示
    // 每一个点的两个残差都会构造两个相对位姿的hessian矩阵的schur complement, 如下图所示的
    // residual1, residual2 可以构造 H_[h_t1]_[h_t2] 的hessian矩阵schur complement
    // host_frame |  target_frame1 target_frame2 target_frame3
    //     |             |               |            |
    //     |          residual1      residual2     residual3
    //     |             |               |            |
    //  point-------------------------------------------
    //
    //step3.1遍历每一3d点个残差
    for (auto r1 : p->residuals) {
        if (!r1->isActive()) continue;
        //step3.2 获取残差对应relative_state的hessian的index
        int r1ht = r1->hostIDX + r1->targetIDX * nframes[tid];
        //step3.3 该3d点对应的另外一个残差
        for (auto r2 : p->residuals) {
            if (!r2->isActive())
                continue;
            // step3.4 计算 H_[relative_pose, relative_ab]_[relative_pose, relative_ab]的Schur complement
            // !!! notice 这个存储器写的绝了!!!
            accD[tid][r1ht + r2->targetIDX * nFrames2].update(r1->JpJdF, r2->JpJdF, p->HdiF);
        }
        // step3.5 计算H_[relative_pose, relative_ab]_c 的Schur complement
        accE[tid][r1ht].update(r1->JpJdF, Hcd, p->HdiF);
        // step3.6 计算b_[relative_pose, relative_ab] 的Schur complement
        accEB[tid][r1ht].update(r1->JpJdF, p->HdiF * p->bdSumF);
    }
}

void AccumulatedSCHessianSSE::stitchDoubleInternal(
    MatXX *H, VecX *b, EnergyFunctional const *const EF,
    int min, int max, Vec10 *stats, int tid)
{
    int toAggregate = NUM_THREADS;
    if (tid == -1) {
        toAggregate = 1;
        tid = 0;
    }    // special case: if we dont do multithreading, dont aggregate.
    if (min == max) return;


    int nf = nframes[0];
    int nframes2 = nf * nf;

    for (int k = min; k < max; k++) {
        int i = k % nf; // host frame
        int j = k / nf; // target frame

        int iIdx = CPARS + i * 8;
        int jIdx = CPARS + j * 8;
        int ijIdx = i + nf * j;

        Mat8C Hpc = Mat8C::Zero();
        Vec8 bp = Vec8::Zero();

        //step1.1: 将不同线程计算的Hpc, bp 进行累加
        for (int tid2 = 0; tid2 < toAggregate; tid2++) {
            accE[tid2][ijIdx].finish();
            accEB[tid2][ijIdx].finish();
            Hpc += accE[tid2][ijIdx].A1m.cast<double>();
            bp += accEB[tid2][ijIdx].A1m.cast<double>();
        }
        //step1.2: 将H_[relative_pose, relative_ab]_c转化为H_[absolute_pose, absolute_ab]_c__schur
        H[tid].block<8, CPARS>(iIdx, 0) += EF->adHost[ijIdx] * Hpc;
        H[tid].block<8, CPARS>(jIdx, 0) += EF->adTarget[ijIdx] * Hpc;
        b[tid].segment<8>(iIdx) += EF->adHost[ijIdx] * bp;
        b[tid].segment<8>(jIdx) += EF->adTarget[ijIdx] * bp;

        //step2: 将H_[relative_pose, relative_ab]_[relative_pose, relative_ab]_schur
        //转化为H_[absolute_pose, absolute_ab]_[absolute_pose, absolute_ab]_schur
        for (int k = 0; k < nf; k++) {
            int kIdx = CPARS + k * 8;
            int ijkIdx = ijIdx + k * nframes2;
            int ikIdx = i + nf * k;

            Mat88 accDM = Mat88::Zero();

            for (int tid2 = 0; tid2 < toAggregate; tid2++) {
                accD[tid2][ijkIdx].finish();
                if (accD[tid2][ijkIdx].num == 0) continue;
                accDM += accD[tid2][ijkIdx].A1m.cast<double>();
            }

            H[tid].block<8, 8>(iIdx, iIdx) += EF->adHost[ijIdx] * accDM * EF->adHost[ikIdx].transpose();
            H[tid].block<8, 8>(jIdx, kIdx) += EF->adTarget[ijIdx] * accDM * EF->adTarget[ikIdx].transpose();
            H[tid].block<8, 8>(jIdx, iIdx) += EF->adTarget[ijIdx] * accDM * EF->adHost[ikIdx].transpose();
            H[tid].block<8, 8>(iIdx, kIdx) += EF->adHost[ijIdx] * accDM * EF->adTarget[ikIdx].transpose();
        }
    }

    //step3: 将多线程求得的Hcc_schur进行累加
    if (min == 0) {
        for (int tid2 = 0; tid2 < toAggregate; tid2++) {
            accHcc[tid2].finish();
            accbc[tid2].finish();
            H[tid].topLeftCorner<CPARS, CPARS>() += accHcc[tid2].A1m.cast<double>();
            b[tid].head<CPARS>() += accbc[tid2].A1m.cast<double>();
        }
    }
}

void AccumulatedSCHessianSSE::stitchDouble(MatXX &H, VecX &b, const EnergyFunctional *const EF, int tid)
{

    int nf = nframes[0];
    int nframes2 = nf * nf;

    H = MatXX::Zero(nf * 8 + CPARS, nf * 8 + CPARS);
    b = VecX::Zero(nf * 8 + CPARS);


    for (int i = 0; i < nf; i++)
        for (int j = 0; j < nf; j++) {
            int iIdx = CPARS + i * 8;
            int jIdx = CPARS + j * 8;
            int ijIdx = i + nf * j;

            accE[tid][ijIdx].finish();
            accEB[tid][ijIdx].finish();

            Mat8C accEM = accE[tid][ijIdx].A1m.cast<double>();
            Vec8 accEBV = accEB[tid][ijIdx].A1m.cast<double>();

            H.block<8, CPARS>(iIdx, 0) += EF->adHost[ijIdx] * accEM;
            H.block<8, CPARS>(jIdx, 0) += EF->adTarget[ijIdx] * accEM;

            b.segment<8>(iIdx) += EF->adHost[ijIdx] * accEBV;
            b.segment<8>(jIdx) += EF->adTarget[ijIdx] * accEBV;

            for (int k = 0; k < nf; k++) {
                int kIdx = CPARS + k * 8;
                int ijkIdx = ijIdx + k * nframes2;
                int ikIdx = i + nf * k;

                accD[tid][ijkIdx].finish();
                if (accD[tid][ijkIdx].num == 0) continue;
                Mat88 accDM = accD[tid][ijkIdx].A1m.cast<double>();

                H.block<8, 8>(iIdx, iIdx) += EF->adHost[ijIdx] * accDM * EF->adHost[ikIdx].transpose();

                H.block<8, 8>(jIdx, kIdx) += EF->adTarget[ijIdx] * accDM * EF->adTarget[ikIdx].transpose();

                H.block<8, 8>(jIdx, iIdx) += EF->adTarget[ijIdx] * accDM * EF->adHost[ikIdx].transpose();

                H.block<8, 8>(iIdx, kIdx) += EF->adHost[ijIdx] * accDM * EF->adTarget[ikIdx].transpose();
            }
        }

    accHcc[tid].finish();
    accbc[tid].finish();
    H.topLeftCorner<CPARS, CPARS>() = accHcc[tid].A1m.cast<double>();
    b.head<CPARS>() = accbc[tid].A1m.cast<double>();

    // ----- new: copy transposed parts for calibration only.
    for (int h = 0; h < nf; h++) {
        int hIdx = CPARS + h * 8;
        H.block<CPARS, 8>(0, hIdx).noalias() = H.block<8, CPARS>(hIdx, 0).transpose();
    }
}

}

}