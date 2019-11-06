#include "internal/OptimizationBackend/AccumulatedTopHessian.h"
#include "internal/OptimizationBackend/EnergyFunctional.h"

namespace ldso {

    namespace internal {

        template<int mode>
        void AccumulatedTopHessianSSE::addPoint(
                shared_ptr<PointHessian> p, EnergyFunctional const *const ef,
                int tid) { // 0 = active, 1 = linearized, 2=marginalize


            assert(mode == 0 || mode == 1 || mode == 2);

            VecCf dc = ef->cDeltaF;
            float dd = p->deltaF;

            float bd_acc = 0;
            float Hdd_acc = 0;
            VecCf Hcd_acc = VecCf::Zero();
            //step1: 遍历每一个点的的残差
            for (shared_ptr<PointFrameResidual> &r : p->residuals) {
                if (mode == 0) {
                    if (r->isLinearized || !r->isActive())
                        continue;
                }
                if (mode == 1) {
                    if (!r->isLinearized || !r->isActive())
                        continue;
                }
                if (mode == 2) {    // marginalize, must be already linearized
                    if (!r->isActive())
                        continue;
                    assert(r->isLinearized);
                }

                shared_ptr<RawResidualJacobian> rJ = r->J;
                int htIDX = r->hostIDX + r->targetIDX * nframes[tid];
                Mat18f dp = ef->adHTdeltaF[htIDX]; // 取出光度误差对（线性化点的相对位姿, 相对光度系数)的增量

                //step1.1 计算误差resApprox
                VecNRf resApprox;
                if (mode == 0)
                    resApprox = rJ->resF; //active point error is evaluated at current state
                if (mode == 2)
                    resApprox = r->res_toZeroF; //marginalize error
                if (mode == 1) {
                    // f(x+ delta) = f(x) + J*delta
                    // resApprox = res_toZeroF + res_on_linearpoint = resF
                    // !!!notice:
                    // 因为resF = f(x+delta)
                    // res_toZeroF = f(x)
                    // res_toZeroF = resF - J*delta 见 Residual.cpp 217 行的fixLinearizationF函数
                    // J*delta = res_on_linearpoint
                    // 所以一阶近似的resApprox 其实就是在当前状态求解得到的而误差f(x+delta),根本就不是一阶近似,作者写的挺迷的,
                    // compute Jp*delta
                    __m128 Jp_delta_x = _mm_set1_ps(
                            rJ->Jpdxi[0].dot(dp.head<6>()) + rJ->Jpdc[0].dot(dc) + rJ->Jpdd[0] * dd);
                    __m128 Jp_delta_y = _mm_set1_ps(
                            rJ->Jpdxi[1].dot(dp.head<6>()) + rJ->Jpdc[1].dot(dc) + rJ->Jpdd[1] * dd);
                    __m128 delta_a = _mm_set1_ps((float) (dp[6]));
                    __m128 delta_b = _mm_set1_ps((float) (dp[7]));

                    for (int i = 0; i < patternNum; i += 4) {
                        // PATTERN: rtz = resF - [JI*Jp Ja Jb]*delta.
                        __m128 rtz = _mm_load_ps(((float *) &r->res_toZeroF) + i);
                        rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (rJ->JIdx)) + i), Jp_delta_x));
                        rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (rJ->JIdx + 1)) + i), Jp_delta_y));
                        rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (rJ->JabF)) + i), delta_a));
                        rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (rJ->JabF + 1)) + i), delta_b));
                        _mm_store_ps(((float *) &resApprox) + i, rtz);
                    }
                }

                //ste1.2 need to compute JI^T * r, and Jab^T * r. (both are 2-vectors).
                Vec2f JI_r(0, 0);
                Vec2f Jab_r(0, 0);
                float rr = 0;
                for (int i = 0; i < patternNum; i++) {
                    JI_r[0] += resApprox[i] * rJ->JIdx[0][i];
                    JI_r[1] += resApprox[i] * rJ->JIdx[1][i];
                    Jab_r[0] += resApprox[i] * rJ->JabF[0][i];
                    Jab_r[1] += resApprox[i] * rJ->JabF[1][i];
                    rr += resApprox[i] * resApprox[i];
                }
                //!!! Notice acc 存储着相机内参,相对位姿,相对光度系数的Hessian矩阵和系数b
                //step1.3 calculate the H_{CPARS,xi} 存储((1+ 4+ 6)/2*(4+6) = 55)元素
                acc[tid][htIDX].update(
                        rJ->Jpdc[0].data(), rJ->Jpdxi[0].data(),
                        rJ->Jpdc[1].data(), rJ->Jpdxi[1].data(),
                        rJ->JIdx2(0, 0), rJ->JIdx2(0, 1), rJ->JIdx2(1, 1));
                //step1.4 计算 H_ab_ab b_ab 和error×error
                acc[tid][htIDX].updateBotRight(
                        rJ->Jab2(0, 0), rJ->Jab2(0, 1), Jab_r[0],
                        rJ->Jab2(1, 1), Jab_r[1], rr);
                //step1.5 计算 H_[C, pose]_ab 和 b_[C, pose]
                acc[tid][htIDX].updateTopRight(
                        rJ->Jpdc[0].data(), rJ->Jpdxi[0].data(),
                        rJ->Jpdc[1].data(), rJ->Jpdxi[1].data(),
                        rJ->JabJIdx(0, 0), rJ->JabJIdx(0, 1),
                        rJ->JabJIdx(1, 0), rJ->JabJIdx(1, 1),
                        JI_r[0], JI_r[1]);
                //step1.6 累计inverse depth的 hessian 矩阵Hdd Hcd和系数b
                //!!! Notice 为什么没有计算H_pose_d 和 H_ab_d? 其实在linearizeAll函数中已经计算过了
                Vec2f Ji2_Jpdd = rJ->JIdx2 * rJ->Jpdd;
                bd_acc += JI_r[0] * rJ->Jpdd[0] + JI_r[1] * rJ->Jpdd[1];
                Hdd_acc += Ji2_Jpdd.dot(rJ->Jpdd);
                Hcd_acc += rJ->Jpdc[0] * Ji2_Jpdd[0] + rJ->Jpdc[1] * Ji2_Jpdd[1];

                nres[tid]++;
            }

            if (mode == 0) {
                p->Hdd_accAF = Hdd_acc;
                p->bd_accAF = bd_acc;
                p->Hcd_accAF = Hcd_acc;
            }
            if (mode == 1 || mode == 2) {
                p->Hdd_accLF = Hdd_acc;
                p->bd_accLF = bd_acc;
                p->Hcd_accLF = Hcd_acc;
            }
            if (mode == 2) {
                p->Hcd_accAF.setZero();
                p->Hdd_accAF = 0;
                p->bd_accAF = 0;
            }

        }

        template void
        AccumulatedTopHessianSSE::addPoint<0>(shared_ptr<PointHessian> p, EnergyFunctional const *const ef, int tid);

        template void
        AccumulatedTopHessianSSE::addPoint<1>(shared_ptr<PointHessian> p, EnergyFunctional const *const ef, int tid);

        template void
        AccumulatedTopHessianSSE::addPoint<2>(shared_ptr<PointHessian> p, EnergyFunctional const *const ef, int tid);

        void AccumulatedTopHessianSSE::stitchDouble(MatXX &H, VecX &b, EnergyFunctional const *const EF, bool usePrior,
                                                    bool useDelta, int tid) {
            H = MatXX::Zero(nframes[tid] * 8 + CPARS, nframes[tid] * 8 + CPARS);
            b = VecX::Zero(nframes[tid] * 8 + CPARS);

            for (int h = 0; h < nframes[tid]; h++)
                for (int t = 0; t < nframes[tid]; t++) {
                    int hIdx = CPARS + h * 8;
                    int tIdx = CPARS + t * 8;
                    int aidx = h + nframes[tid] * t;


                    acc[tid][aidx].finish();
                    if (acc[tid][aidx].num == 0) continue;

                    MatPCPC accH = acc[tid][aidx].H.cast<double>();


                    H.block<8, 8>(hIdx, hIdx).noalias() +=
                            EF->adHost[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adHost[aidx].transpose();

                    H.block<8, 8>(tIdx, tIdx).noalias() +=
                            EF->adTarget[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adTarget[aidx].transpose();

                    H.block<8, 8>(hIdx, tIdx).noalias() +=
                            EF->adHost[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adTarget[aidx].transpose();

                    H.block<8, CPARS>(hIdx, 0).noalias() += EF->adHost[aidx] * accH.block<8, CPARS>(CPARS, 0);

                    H.block<8, CPARS>(tIdx, 0).noalias() += EF->adTarget[aidx] * accH.block<8, CPARS>(CPARS, 0);

                    H.topLeftCorner<CPARS, CPARS>().noalias() += accH.block<CPARS, CPARS>(0, 0);

                    b.segment<8>(hIdx).noalias() += EF->adHost[aidx] * accH.block<8, 1>(CPARS, 8 + CPARS);

                    b.segment<8>(tIdx).noalias() += EF->adTarget[aidx] * accH.block<8, 1>(CPARS, 8 + CPARS);

                    b.head<CPARS>().noalias() += accH.block<CPARS, 1>(0, 8 + CPARS);
                }

            // ----- new: copy transposed parts.
            for (int h = 0; h < nframes[tid]; h++) {
                int hIdx = CPARS + h * 8;
                H.block<CPARS, 8>(0, hIdx).noalias() = H.block<8, CPARS>(hIdx, 0).transpose();

                for (int t = h + 1; t < nframes[tid]; t++) {
                    int tIdx = CPARS + t * 8;
                    // 下面加法操作作用是因为hessian矩阵块有时存储在Ｈ_host_target,有时存储在H_target_host中 
                    H.block<8, 8>(hIdx, tIdx).noalias() += H.block<8, 8>(tIdx, hIdx).transpose();
                    H.block<8, 8>(tIdx, hIdx).noalias() = H.block<8, 8>(hIdx, tIdx).transpose();
                }
            }


            if (usePrior) {
                assert(useDelta);
                H.diagonal().head<CPARS>() += EF->cPrior;
                b.head<CPARS>() += EF->cPrior.cwiseProduct(EF->cDeltaF.cast<double>());
                for (int h = 0; h < nframes[tid]; h++) {
                    H.diagonal().segment<8>(CPARS + h * 8) += EF->frames[h]->prior;
                    b.segment<8>(CPARS + h * 8) += EF->frames[h]->prior.cwiseProduct(EF->frames[h]->delta_prior);
                }
            }
        }

        void AccumulatedTopHessianSSE::stitchDoubleInternal(MatXX *H, VecX *b, EnergyFunctional const *const EF,
                                                            bool usePrior, int min, int max, Vec10 *stats, int tid) {
            int toAggregate = NUM_THREADS;
            if (tid == -1) {
                toAggregate = 1;
                tid = 0;
            }    // special case: if we dont do multithreading, dont aggregate.
            if (min == max) return;


            for (int k = min; k < max; k++) {
                int h = k % nframes[0];
                int t = k / nframes[0];

                int hIdx = CPARS + h * 8;
                int tIdx = CPARS + t * 8;
                int aidx = h + nframes[0] * t;

                assert(aidx == k);

                MatPCPC accH = MatPCPC::Zero();

                //step1: 实现多线程求解的H_[camera, relative_poses, relative_, relative_b] 和b 的累加
                for (int tid2 = 0; tid2 < toAggregate; tid2++) {
                    acc[tid2][aidx].finish(); //求解 H_[camera, relative_poses, relative_, relative_b] 和b
                    if (acc[tid2][aidx].num == 0) continue;
                    accH += acc[tid2][aidx].H.cast<double>();
                }
                //step2: 将上一步求解的H_[camera, relative_poses, relative_a, relative_b]
                // 根据伴随矩阵转换为H_[camera, absolute_pose, absolute_a, absolute_b]
                //step2.1: 恢复绝对位姿的hessian block H_h_h, H_t_h, H_t_t
                //Notice: DSO 在这里将具有Scale 属性相对变量的正规方程变成了具有没有scale属性的绝对位姿的正规矩阵
                // Scale 相当于给优化变量赋予了不同的权重
                H[tid].block<8, 8>(hIdx, hIdx).noalias() +=
                        EF->adHost[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adHost[aidx].transpose();

                H[tid].block<8, 8>(tIdx, tIdx).noalias() +=
                        EF->adTarget[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adTarget[aidx].transpose();

                H[tid].block<8, 8>(hIdx, tIdx).noalias() +=
                        EF->adHost[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adTarget[aidx].transpose();
                //step2.2: 恢复绝对位姿和相机内参的hessian block H_h_c, H_t_c
                H[tid].block<8, CPARS>(hIdx, 0).noalias() += EF->adHost[aidx] * accH.block<8, CPARS>(CPARS, 0);

                H[tid].block<8, CPARS>(tIdx, 0).noalias() += EF->adTarget[aidx] * accH.block<8, CPARS>(CPARS, 0);
                //step2.3: 恢复相机内参的hessian block H_c_c
                H[tid].topLeftCorner<CPARS, CPARS>().noalias() += accH.block<CPARS, CPARS>(0, 0);
                //step2.4: 恢复绝对位姿对应的b, b_h, b_t
                b[tid].segment<8>(hIdx).noalias() += EF->adHost[aidx] * accH.block<8, 1>(CPARS, CPARS + 8);

                b[tid].segment<8>(tIdx).noalias() += EF->adTarget[aidx] * accH.block<8, 1>(CPARS, CPARS + 8);
                //step2.5 恢复相机内参的对应的b, b_c
                b[tid].head<CPARS>().noalias() += accH.block<CPARS, 1>(0, CPARS + 8);

            }

            // step3: 给状态加先验
            // only do this on one thread.
            if (min == 0 && usePrior) {
                // 相机内参加先验
                H[tid].diagonal().head<CPARS>() += EF->cPrior;
                b[tid].head<CPARS>() += EF->cPrior.cwiseProduct(EF->cDeltaF.cast<double>());
                // 相机的位姿和光度仿射变换系数加先验
                for (int h = 0; h < nframes[tid]; h++) {
                    H[tid].diagonal().segment<8>(CPARS + h * 8) += EF->frames[h]->prior;
                    b[tid].segment<8>(CPARS + h * 8) += EF->frames[h]->prior.cwiseProduct(EF->frames[h]->delta_prior);

                }
            }
        }

    };

}