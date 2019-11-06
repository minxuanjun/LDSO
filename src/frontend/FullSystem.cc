#include "Feature.h"
#include "Frame.h"
#include "Point.h"

#include "frontend/FullSystem.h"
#include "frontend/CoarseInitializer.h"
#include "frontend/CoarseTracker.h"
#include "frontend/LoopClosing.h"

#include "internal/ImmaturePoint.h"
#include "internal/GlobalCalib.h"
#include "internal/GlobalFuncs.h"
#include "internal/OptimizationBackend/EnergyFunctional.h"

#include <opencv2/features2d/features2d.hpp>
#include <iomanip>
#include <opencv2/highgui/highgui.hpp>

using namespace ldso;
using namespace ldso::internal;

namespace ldso
{

FullSystem::FullSystem(shared_ptr<ORBVocabulary> voc)
    : coarseDistanceMap(new CoarseDistanceMap(wG[0], hG[0])),
      coarseTracker(new CoarseTracker(wG[0], hG[0])),
      coarseTracker_forNewKF(new CoarseTracker(wG[0], hG[0])),
      coarseInitializer(new CoarseInitializer(wG[0], hG[0])),
      ef(new EnergyFunctional()),
      Hcalib(new Camera(fxG[0], fyG[0], cxG[0], cyG[0])),
      globalMap(new Map(this)),
      vocab(voc)
{

    LOG(INFO) << "This is Direct Sparse Odometry, a fully direct VO proposed by TUM vision group."
                 "For more information about dso, see Direct Sparse Odometry, J. Engel, V. Koltun, "
                 "D. Cremers, In arXiv:1607.02565, 2016. For loop closing part, see "
                 "LDSO: Direct Sparse Odometry with Loop Closure, X. Gao, R. Wang, N. Demmel, D. Cremers, "
                 "In International Conference on Intelligent Robots and Systems (IROS), 2018 "
              << endl;

    Hcalib->CreateCH(Hcalib);
    lastCoarseRMSE.setConstant(100);
    ef->red = &this->threadReduce;
    mappingThread = thread(&FullSystem::mappingLoop, this);

    pixelSelector = shared_ptr<PixelSelector>(new PixelSelector(wG[0], hG[0]));
    selectionMap = new float[wG[0] * hG[0]];

    if (setting_enableLoopClosing) {
        loopClosing = shared_ptr<LoopClosing>(new LoopClosing(this));
        if (setting_fastLoopClosing)
            LOG(INFO) << "Use fast loop closing" << endl;
    }
    else {
        LOG(INFO) << "loop closing is disabled" << endl;
    }
}

FullSystem::~FullSystem()
{
    blockUntilMappingIsFinished();
    // remember to release the inner structure
    this->unmappedTrackedFrames.clear();
    if (setting_enableLoopClosing == false) {
        delete[] selectionMap;
    }
    else {
    }
}

void FullSystem::addActiveFrame(ImageAndExposure *image, int id)
{
    if (isLost)
        return;
    unique_lock<mutex> lock(trackMutex);

    LOG(INFO) << "*** taking frame " << id << " ***" << endl;

    // create frame and frame hessian
    shared_ptr<Frame> frame(new Frame(image->timestamp));
    frame->CreateFH(frame);
    allFrameHistory.push_back(frame);

    // ==== make images ==== //
    shared_ptr<FrameHessian> fh = frame->frameHessian;
    fh->ab_exposure = image->exposure_time;
    // create image gray and gradient pyramid
    fh->makeImages(image->image, Hcalib->mpCH);

    if (!initialized) {
        LOG(INFO) << "Initializing ... " << endl;
        // use initializer
        if (coarseInitializer->frameID < 0) { // first frame not set, set it
            coarseInitializer->setFirst(Hcalib->mpCH, fh);
        }
        else if (coarseInitializer->trackFrame(fh)) {
            // init succeeded
            initializeFromInitializer(fh);
            lock.unlock();
            deliverTrackedFrame(fh, true); // create new keyframe
            LOG(INFO) << "init success." << endl;
        }
        else {
            // still initializing
            frame->poseValid = false;
            // release the immature point and point of the this frame , don't need this frame, release all the internal
            frame->ReleaseAll();
        }
        return;
    }
    else {
        // init finished, do tracking
        // =========================== SWAP tracking reference?. =========================
        if (coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID) {
            unique_lock<mutex> crlock(coarseTrackerSwapMutex);
            LOG(INFO) << "swap coarse tracker to " << coarseTracker_forNewKF->refFrameID << endl;
            auto tmp = coarseTracker;
            coarseTracker = coarseTracker_forNewKF;
            coarseTracker_forNewKF = tmp;
        }

        // track the new frame and get the state
        LOG(INFO) << "tracking new frame" << endl;
        Vec4 tres = trackNewCoarse(fh);

        if (!std::isfinite((double) tres[0]) || !std::isfinite((double) tres[1]) ||
            !std::isfinite((double) tres[2]) || !std::isfinite((double) tres[3])) {
            // invalid result
            LOG(WARNING) << "Initial Tracking failed: LOST!" << endl;
            isLost = true;
            return;
        }

        bool needToMakeKF = false;
        // 如果定时关键帧插入的标志位设置为true, 那么如果关键帧的输入只有一帧或者相邻帧的时间间隔太长,则需要插入关键帧
        if (setting_keyframesPerSecond > 0) {
            needToMakeKF = allFrameHistory.size() == 1 ||
                (frame->timeStamp - frames.back()->timeStamp) >
                    0.95f / setting_keyframesPerSecond;
        }
        else {
            //　计算参考帧和当前帧的相对光度变换参数
            Vec2 refToFh = AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
                                                       coarseTracker->lastRef_aff_g2l, fh->aff_g2l());
            // keyframe creation,　见DSO论文Chapter 3.2 Frame Management 的 Step2 Keyframe Creation 的判断条件
            float b = setting_kfGlobalWeight * setting_maxShiftWeightT * sqrtf((double) tres[1]) /       //纯位移光流
                (wG[0] + hG[0]) +
                setting_kfGlobalWeight * setting_maxShiftWeightR * sqrtf((double) tres[2]) /       // 没有用
                    (wG[0] + hG[0]) +
                setting_kfGlobalWeight * setting_maxShiftWeightRT * sqrtf((double) tres[3]) /
                    // 旋转和平移复合的光溜
                    (wG[0] + hG[0]) +
                setting_kfGlobalWeight * setting_maxAffineWeight * fabs(logf((float) refToFh[0])); // 光度变换

            bool b1 = b > 1;
            // if the current photometric error larger than the initial errors
            // 如果追踪当前帧计算得到的光度误差大于初始的误差的2被,那么需要插入关键帧
            bool b2 = 2 * coarseTracker->firstCoarseRMSE < tres[0];

            needToMakeKF = allFrameHistory.size() == 1 || b1 || b2;
        }

        if (viewer)
            viewer->publishCamPose(fh->frame, Hcalib->mpCH);

        lock.unlock();
        LOG(INFO) << "deliver frame " << fh->frame->id << endl;
        deliverTrackedFrame(fh, needToMakeKF);
        LOG(INFO) << "add active frame returned" << endl
                  << endl;
        return;
    }
}

// pass the frame hessian to mapping thread
void FullSystem::deliverTrackedFrame(shared_ptr<FrameHessian> fh, bool needKF)
{
    // 单线程执行makeKeyFrame 或者执行 makeNonKeyFrame
    if (linearizeOperation) {
        if (needKF) {
            makeKeyFrame(fh);
        }
        else {
            makeNonKeyFrame(fh);
        }
    }
    else {
        // 多线程执行makeKeyFrame 或者 makeNonKeyFrame
        // step1: 获取mappingThread 的锁
        unique_lock<mutex> lock(trackMapSyncMutex);
        // step2: 将当前帧压人到unmappedTrackedFrames 队列
        unmappedTrackedFrames.push_back(fh->frame);
        // step3: 唤醒线程
        trackedFrameSignal.notify_all();
        // step4: 等待建图结束
        while (coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1) {
            LOG(INFO) << "wait for mapped frame signal" << endl;
            mappedFrameSignal.wait(lock);
        }
        // step5: 释放锁, 感觉没必要，unique_lock应该会自动释放锁
        lock.unlock();
    }
}

Vec4 FullSystem::trackNewCoarse(shared_ptr<FrameHessian> fh)
{

    assert(allFrameHistory.size() > 0);

    shared_ptr<FrameHessian> lastF = coarseTracker->lastRef;
    CHECK(coarseTracker->lastRef->frame != nullptr);

    AffLight aff_last_2_l = AffLight(0, 0);

    // try a lot of pose values and see which is the best
    std::vector<SE3, Eigen::aligned_allocator<SE3>>
        lastF_2_fh_tries;
    if (allFrameHistory.size() == 2)
        for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++) // TODO: maybe wrong, size is obviously zero
            lastF_2_fh_tries.push_back(SE3());                     // use identity
    else {

        // fill the pose tries ...
        // use the last before last and the last before before last (well my English is really poor...)
        shared_ptr<Frame> slast = allFrameHistory[allFrameHistory.size() - 2];
        shared_ptr<Frame> sprelast = allFrameHistory[allFrameHistory.size() - 3];

        SE3 slast_2_sprelast;
        SE3 lastF_2_slast;

        { // lock on global pose consistency!
            unique_lock<mutex> crlock(shellPoseMutex);
            slast_2_sprelast = sprelast->getPose() * slast->getPose().inverse();
            lastF_2_slast = slast->getPose() * lastF->frame->getPose().inverse();
            aff_last_2_l = slast->aff_g2l;
        }
        SE3 fh_2_slast = slast_2_sprelast; // assumed to be the same as fh_2_slast.

        // get last delta-movement.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast); // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() *
            lastF_2_slast); // assume double motion (frame skipped)
        lastF_2_fh_tries.push_back(
            SE3::exp(fh_2_slast.log() * 0.5).inverse() * lastF_2_slast); // assume half motion.
        lastF_2_fh_tries.push_back(lastF_2_slast);                       // assume zero motion.
        lastF_2_fh_tries.push_back(SE3());                               // assume zero motion FROM KF.

        // just try a TON of different initializations (all rotations). In the end,
        // if they don't work they will only be tried on the coarsest level, which is super fast anyway.
        // also, if tracking rails here we loose, so we really, really want to avoid that.
        // the number the disturbs is 3 * (3*3*3 -1) = 3*26
        for (float rotDelta = 0.02;
             rotDelta < 0.05; rotDelta += 0.01) { // TODO changed this into +=0.01 where DSO writes ++
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, rotDelta, 0, 0),
                    Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, 0, rotDelta, 0),
                    Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, 0, 0, rotDelta),
                    Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, -rotDelta, 0, 0),
                    Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, 0, -rotDelta, 0),
                    Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, 0, 0, -rotDelta),
                    Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, 0),
                    Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, 0, rotDelta, rotDelta),
                    Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, rotDelta, 0, rotDelta),
                    Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, 0),
                    Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, 0, -rotDelta, rotDelta),
                    Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, -rotDelta, 0, rotDelta),
                    Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, 0),
                    Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, 0, rotDelta, -rotDelta),
                    Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, rotDelta, 0, -rotDelta),
                    Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, 0),
                    Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, 0, -rotDelta, -rotDelta),
                    Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, -rotDelta, 0, -rotDelta),
                    Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta),
                    Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, rotDelta),
                    Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, -rotDelta),
                    Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, rotDelta),
                    Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, -rotDelta),
                    Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, rotDelta),
                    Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, -rotDelta),
                    Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, rotDelta),
                    Vec3(0, 0, 0))); // assume constant motion.
        }

        if (!slast->poseValid || !sprelast->poseValid || !lastF->frame->poseValid) {
            lastF_2_fh_tries.clear();
            lastF_2_fh_tries.push_back(SE3());
        }
    }

    Vec3 flowVecs = Vec3(100, 100, 100);
    SE3 lastF_2_fh = SE3();
    AffLight aff_g2l = AffLight(0, 0);

    // as long as maxResForImmediateAccept is not reached, I'll continue through the options.
    // I'll keep track of the so-far best achieved residual for each level in achievedRes.
    // If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.

    Vec5 achievedRes = Vec5::Constant(NAN);
    bool haveOneGood = false;
    int tryIterations = 0;
    for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++) {

        AffLight aff_g2l_this = aff_last_2_l;
        SE3 lastF_2_fh_this = lastF_2_fh_tries[i];

        // use coarse tracker to solve the iteration
        bool trackingIsGood = coarseTracker->trackNewestCoarse(
            fh, lastF_2_fh_this, aff_g2l_this,
            pyrLevelsUsed - 1,
            achievedRes); // in each level has to be at least as good as the last try.
        tryIterations++;

        // do we have a new winner?
        if (trackingIsGood && std::isfinite((float) coarseTracker->lastResiduals[0]) &&
            !(coarseTracker->lastResiduals[0] >= achievedRes[0])) {
            flowVecs = coarseTracker->lastFlowIndicators;
            aff_g2l = aff_g2l_this;
            lastF_2_fh = lastF_2_fh_this;
            haveOneGood = true;
        }

        // take over achieved res (always).
        if (haveOneGood) {
            for (int i = 0; i < 5; i++) {
                if (!std::isfinite((float) achievedRes[i]) ||
                    achievedRes[i] >
                        coarseTracker->lastResiduals[i]) // take over if achievedRes is either bigger or NAN.
                    achievedRes[i] = coarseTracker->lastResiduals[i];
            }
        }

        if (haveOneGood && achievedRes[0] < lastCoarseRMSE[0] * setting_reTrackThreshold)
            break;
    }

    if (!haveOneGood) {
        LOG(WARNING) << "BIG ERROR! tracking failed entirely. Take predicted pose and hope we may somehow recover."
                     << endl;
        flowVecs = Vec3(0, 0, 0);
        aff_g2l = aff_last_2_l;
        lastF_2_fh = lastF_2_fh_tries[0];
    }

    lastCoarseRMSE = achievedRes;

    // set the pose of new frame
    CHECK(coarseTracker->lastRef->frame != nullptr);
    SE3 camToWorld = lastF->frame->getPose().inverse() * lastF_2_fh.inverse();
    fh->frame->setPose(camToWorld.inverse());
    fh->frame->aff_g2l = aff_g2l;

    if (coarseTracker->firstCoarseRMSE < 0)
        coarseTracker->firstCoarseRMSE = achievedRes[0];

    LOG(INFO) << "Coarse Tracker tracked ab = " << aff_g2l.a << " " << aff_g2l.b << " (exp " << fh->ab_exposure
              << " ). Res " << achievedRes[0] << endl;
    // residul  optical flow which caused by RT   0  optical flow caused by pure translation
    return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
}

void FullSystem::blockUntilMappingIsFinished()
{
    {
        unique_lock<mutex> lock(trackMapSyncMutex);
        if (!runMapping) {
            // mapping is already finished, no need to finish again
            return;
        }
        runMapping = false;
        trackedFrameSignal.notify_all();
    }

    mappingThread.join();

    if (setting_enableLoopClosing) {
        loopClosing->SetFinish(true);
        if (globalMap->NumFrames() > 4) {
            globalMap->lastOptimizeAllKFs();
        }
    }

    // Update world points in case optimization hasn't run (with all keyframes)
    // It would maybe be better if the 3d points would always be updated as soon
    // as the poses or depths are updated (no matter if in PGO or in sliding window BA)
    globalMap->UpdateAllWorldPoints();
}

void FullSystem::makeKeyFrame(shared_ptr<FrameHessian> fh)
{

    // 获取当前帧
    shared_ptr<Frame> frame = fh->frame;
    // 获取上一帧
    auto refFrame = frames.back();
    // Step1: 设置newset frame 对应的FrameHessian的初始位姿(具有scale 属性的)和光度变换系数
    // scale 表达的意思是DSO的后端滑窗优化的过程中优化的变量的设置了一个权重,相当于对变量设置了一个信息矩阵
    {
        // shellPoseMutex used to protected the access the frame state
        unique_lock<mutex> crlock(shellPoseMutex);
        //使用frame存储的Tcw和光度参数初始化对应的FrameHessian
        fh->setEvalPT_scaled(fh->frame->getPose(), fh->frame->aff_g2l);
        // 设置关键帧帧在LoopClosing优化的初值
        fh->frame->setPoseOpti(Sim3(fh->frame->getPose().matrix()));
    }

    LOG(INFO) << "frame " << fh->frame->id << " is marked as key frame, active keyframes: " << frames.size()
              << endl;
    // Step2: 使用newest Frame 更新滑窗中其他帧的immature point
    // trace new keyframe to update the inverse depth of the
    // immature point in slide widow by epipolar search
    traceNewCoarse(fh);

    unique_lock<mutex> lock(mapMutex);
    // Step3: 标记要被边缘化的关键帧
    // == flag frames to be marginalized  == //
    flagFramesForMarginalization(fh);

    // Step4: 将当前帧插入到滑窗中
    // Step4.1: 将最新帧插入到滑窗的数据结构的中，计算其在滑窗和全局地图中的id
    {
        unique_lock<mutex> lck(framesMutex);
        // idx is the id of frames in sliding window
        fh->idx = frames.size();
        frames.push_back(fh->frame);
        fh->frame->kfId = fh->frameID = globalMap->NumFrames(); //关键帧在全局地图的id
    }
    // Step4.2 将最新帧对应的FrameHessian插入到EnergyFunctional中,
    // insert FrameHessian of newest keyframes to EnergyFunctional, and calculate the the adjoint matrix and
    // make the id host target of point residual from screatch
    ef->insertFrame(fh, Hcalib->mpCH);
    // Step4.3
    // calculate the relative pose the photometric affine coefficient
    // and calculate the increment the pose, the camera and inverse depth
    setPrecalcValues();

    // =========================== add new residuals for old points =========================
    LOG(INFO) << "adding new residuals" << endl;
    int numFwdResAdde = 0;
    //step5 add new residuals for old active points in frames except newest frame
    //step5.1 enumerate all frames in sliding window
    for (auto fht : frames) {
        shared_ptr<FrameHessian> &fh1 = fht->frameHessian;
        if (fh1 == fh) // except newest frame
            continue;
        //step5.2 enumerate features in the frame
        for (auto feat : fht->features) {
            if (feat->status == Feature::FeatureStatus::VALID &&
                feat->point->status == Point::PointStatus::ACTIVE) {

                shared_ptr<PointHessian> ph = feat->point->mpPH;

                //step5.3 add new residuals into this point hessian
                shared_ptr<PointFrameResidual> r(
                    new PointFrameResidual(ph, fh1, fh)); // residual from fh1 to fh

                r->setState(ResState::IN);
                ph->residuals.push_back(r);
                //step5.4 将视觉残差插入到能量函数中，同时计算该视觉残差因子中的H_poseab_inverse_depth
                ef->insertResidual(r);
                //step5.5 更新视觉残差因子上一次的残差记录值
                ph->lastResiduals[1] = ph->lastResiduals[0];
                ph->lastResiduals[0] = pair<shared_ptr<PointFrameResidual>, ResState>(r, ResState::IN);
                numFwdResAdde++;
            }
        }
    }

    // =========================== Activate Points (& flag for marginalization). =========================
    // step6: convert the immature point in slide window to Point
    activatePointsMT();
    // step7: 重新梳理滑窗中关键帧的id和激活点光度误差对应的host与target frame的id
    ef->makeIDX();

    // =========================== OPTIMIZE ALL =========================
    fh->frameEnergyTH = frames.back()->frameHessian->frameEnergyTH;
    LOG(INFO) << "call optimize on kf " << frame->kfId << endl;
    // step8: 滑窗优化
    float rmse = optimize(setting_maxOptIterations);
    LOG(INFO) << "optimize is done!" << endl;

    // =========================== Figure Out if INITIALIZATION FAILED =========================
    int numKFs = globalMap->NumFrames();
    if (numKFs <= 4) {
        if (numKFs == 2 && rmse > 20 * benchmark_initializerSlackFactor) {
            LOG(WARNING) << "I THINK INITIALIZATINO FAILED! Resetting." << endl;
            LOG(INFO) << "rmse = " << rmse << endl;
            initFailed = true;
        }
        if (numKFs == 3 && rmse > 13 * benchmark_initializerSlackFactor) {
            LOG(WARNING) << "I THINK INITIALIZATINO FAILED! Resetting." << endl;
            LOG(INFO) << "rmse = " << rmse << endl;
            initFailed = true;
        }
        if (numKFs == 4 && rmse > 9 * benchmark_initializerSlackFactor) {
            LOG(WARNING) << "I THINK INITIALIZATINO FAILED! Resetting." << endl;
            LOG(INFO) << "rmse = " << rmse << endl;
            initFailed = true;
        }
    }

    if (isLost)
        return;

    // =========================== REMOVE OUTLIER =========================
    removeOutliers();

    // swap the coarse Tracker for new kf
    {
        unique_lock<mutex> crlock(coarseTrackerSwapMutex);
        coarseTracker_forNewKF->makeK(Hcalib->mpCH);
        vector<shared_ptr<FrameHessian>> fhs;
        for (auto &f : frames)
            fhs.push_back(f->frameHessian);
        coarseTracker_forNewKF->setCoarseTrackingRef(fhs);
    }

    // =========================== (Activate-)Marginalize Points =========================
    // traditional bundle adjustment when marging all points
    flagPointsForRemoval();
    ef->dropPointsF();

    getNullspaces(
        ef->lastNullspaces_pose,
        ef->lastNullspaces_scale,
        ef->lastNullspaces_affA,
        ef->lastNullspaces_affB);

    ef->marginalizePointsF();

    // =========================== add new Immature points & new residuals =========================
    makeNewTraces(fh, 0);

    // record the relative poses, note we are building a covisibility graph in fact
    auto minandmax = std::minmax_element(frames.begin(), frames.end(), CmpFrameKFID());
    unsigned long minKFId = (*minandmax.first)->kfId;
    unsigned long maxKFId = (*minandmax.second)->kfId;

    if (setting_fastLoopClosing == false) {
        // record all active keyframes
        for (auto &fr : frames) {
            auto allKFs = globalMap->GetAllKFs();
            for (auto &f2 : allKFs) {
                if (f2->kfId > minKFId && f2->kfId < maxKFId && f2 != fr) {
                    unique_lock<mutex> lock(fr->mutexPoseRel);
                    unique_lock<mutex> lock2(f2->mutexPoseRel);
                    fr->poseRel[f2] = Sim3((fr->getPose() * f2->getPose().inverse()).matrix());
                    f2->poseRel[fr] = Sim3((f2->getPose() * fr->getPose().inverse()).matrix());
                }
            }
        }
    }
    else {
        // only record the reference and first frame and also update the keyframe poses in window
        {
            unique_lock<mutex> lock(frame->mutexPoseRel);
            frame->poseRel[refFrame] = Sim3((frame->getPose() * refFrame->getPose().inverse()).matrix());
            auto firstFrame = frames.front();
            frame->poseRel[firstFrame] = Sim3((frame->getPose() * firstFrame->getPose().inverse()).matrix());
        }

        // update the poses in window
        for (auto &fr : frames) {
            if (fr == frame)
                continue;
            for (auto rel : fr->poseRel) {
                auto f2 = rel.first;
                fr->poseRel[f2] = Sim3((fr->getPose() * f2->getPose().inverse()).matrix());
            }
        }
    }

    // visualization
    if (viewer)
        viewer->publishKeyframes(frames, false, Hcalib->mpCH);

    // =========================== Marginalize Frames =========================
    {
        unique_lock<mutex> lck(framesMutex);
        for (unsigned int i = 0; i < frames.size(); i++)
            if (frames[i]->frameHessian->flaggedForMarginalization) {
                LOG(INFO) << "marg frame " << frames[i]->id << endl;
                CHECK(frames[i] != coarseTracker->lastRef->frame);
                marginalizeFrame(frames[i]);
                i = 0;
            }
    }

    // add current kf into map and detect loops
    globalMap->AddKeyFrame(fh->frame);
    if (setting_enableLoopClosing) {
        loopClosing->InsertKeyFrame(frame);
    }
    LOG(INFO) << "make keyframe done" << endl;
}

void FullSystem::makeNonKeyFrame(shared_ptr<FrameHessian> &fh)
{
    {
        unique_lock<mutex> crlock(shellPoseMutex);
        fh->setEvalPT_scaled(fh->frame->getPose(), fh->frame->aff_g2l);
    }
    // update the inverse depth of immature point then dele
    traceNewCoarse(fh);
    fh->frame->ReleaseAll(); // no longer needs it
}

void FullSystem::marginalizeFrame(shared_ptr<Frame> &frame)
{

    // marginalize or remove all this frames points
    ef->marginalizeFrame(frame->frameHessian);

    // drop all observations of existing points in that frame
    for (shared_ptr<Frame> &fr : frames) {
        if (fr == frame)
            continue;
        for (auto feat : fr->features) {
            if (feat->status == Feature::FeatureStatus::VALID &&
                feat->point->status == Point::PointStatus::ACTIVE) {

                shared_ptr<PointHessian> ph = feat->point->mpPH;
                // remove the residuals projected into this frame
                size_t n = ph->residuals.size();
                for (size_t i = 0; i < n; i++) {
                    auto r = ph->residuals[i];
                    if (r->target.lock() == frame->frameHessian) {
                        if (ph->lastResiduals[0].first == r)
                            ph->lastResiduals[0].first = nullptr;
                        else if (ph->lastResiduals[1].first == r)
                            ph->lastResiduals[1].first = nullptr;
                        ef->dropResidual(r);
                        i--;
                        n--;
                    }
                }
            }
        }
    }

    // remove this frame from recorded frames
    frame->ReleaseAll(); // release all things in this frame
    deleteOutOrder<shared_ptr<Frame>>(frames, frame);

    // reset the optimization idx
    for (unsigned int i = 0; i < frames.size(); i++)
        frames[i]->frameHessian->idx = i;

    setPrecalcValues();
    ef->setAdjointsF(Hcalib->mpCH);
}

void FullSystem::flagFramesForMarginalization(shared_ptr<FrameHessian> &newFH)
{
    // setting_minFrameAge -> 1
    // setting_maxFrames -> 6,7
    if (setting_minFrameAge > setting_maxFrames) {
        for (size_t i = setting_maxFrames; i < frames.size(); i++) {
            shared_ptr<FrameHessian> &fh = frames[i - setting_maxFrames]->frameHessian;
            LOG(INFO) << "frame " << fh->frame->kfId << " is set as marged" << endl;
            fh->flaggedForMarginalization = true;
        }
        return;
    }

    int flagged = 0;

    // marginalize all frames that have not enough points.

    // step1.1 : 遍历滑窗中所有帧
    for (int i = 0; i < (int) frames.size(); i++) {
        // step1.2 : 获取帧所对应的FrameHessian
        shared_ptr<FrameHessian> &fh = frames[i]->frameHessian;
        // step1.3 : 统计当前帧的in(未成熟点+ 激活点) 和out(外点和边缘化点)的个数
        int in = 0, out = 0;
        for (auto &feat : frames[i]->features) {
            if (feat->status == Feature::FeatureStatus::IMMATURE) {
                in++;
                continue;
            }

            shared_ptr<Point> p = feat->point;
            if (p && p->status == Point::PointStatus::ACTIVE)
                in++;
            else
                out++;
        }
        // step1.4 计算第i帧和滑窗中最后一帧的相对光度变换参数
        Vec2 refToFh = AffLight::fromToVecExposure(frames.back()->frameHessian->ab_exposure, fh->ab_exposure,
                                                   frames.back()->frameHessian->aff_g2l(), fh->aff_g2l());

        // some kind of marginlization conditions
        // if the immature point and active point less than 0.05

        // step1.5 标记哪些帧应该被边缘化，
        // a. 未成熟点和激活点的比例太小
        // b. 与滑窗中最后一帧的的相对光度变换太大
        // c. 当前滑窗中的帧数减去被边缘化的帧数必须大于滑窗中设置的最小帧数
        if ((in < setting_minPointsRemaining * (in + out) ||
            fabs(logf((float) refToFh[0])) > setting_maxLogAffFacInWindow) &&
            ((int) frames.size()) - flagged > setting_minFrames) {
            LOG(INFO) << "frame " << fh->frame->kfId << " is set as marged" << endl;
            fh->flaggedForMarginalization = true;
            flagged++;
        }
    }

    // still too much, marginalize one
    // step2 :如果滑窗帧的数还是太大,那么采用边缘化准则3来判断哪些帧应该被边缘化,见论文公式8.20
    if ((int) frames.size() - flagged >= setting_maxFrames) {
        double smallestScore = 1;
        shared_ptr<Frame> toMarginalize = nullptr;
        // 获取滑窗中的最后一帧
        shared_ptr<Frame> latest = frames.back();

        // step2.1 遍历滑窗中的每一帧
        for (auto &fr : frames) {
            // the frameId of newest frame is 0
            // 如果第i帧到滑窗中最后一帧的时间检测过长或者是最新帧,都不会被删除,为了保持到最新帧的相对光度差异较小
            if (fr->frameHessian->frameID > latest->frameHessian->frameID - setting_minFrameAge ||
                fr->frameHessian->frameID == 0)
                continue;
            double distScore = 0;
            // 计算公式 8.20
            for (FrameFramePrecalc &ffh : fr->frameHessian->targetPrecalc) {
                // 计算总的逆距离和时，要排除滑窗中最新的两帧和自己, 见论文
                if (ffh.target.lock()->frameID > latest->frameHessian->frameID - setting_minFrameAge + 1 ||
                    ffh.target.lock() == ffh.host.lock())
                    continue;

                distScore += 1 / (1e-5 + ffh.distanceLL);
            }
            distScore *= -sqrtf(fr->frameHessian->targetPrecalc.back().distanceLL);
            //
            if (distScore < smallestScore) {
                smallestScore = distScore;
                toMarginalize = fr;
            }
        }
        //step2.2 如果找到离当前帧最近的一帧,那么将其对应的FrameHessian标记为要被边缘化
        if (toMarginalize) {
            toMarginalize->frameHessian->flaggedForMarginalization = true;
            LOG(INFO) << "frame " << toMarginalize->kfId << " is set as marged" << endl;
            flagged++;
        }
    }
}

float FullSystem::optimize(int mnumOptIts)
{

    // step1: 依据滑窗中关键帧的数目动态的调节优化的次数
    if (frames.size() < 2)
        return 0;
    if (frames.size() < 3)
        mnumOptIts = 20;
    if (frames.size() < 4)
        mnumOptIts = 15;

    // step2: get statistics and active residuals.
    activeResiduals.clear();
    int numPoints = 0;
    int numLRes = 0;
    for (shared_ptr<Frame> &fr : frames) {
        for (auto &feat : fr->features) {
            shared_ptr<Point> p = feat->point;
            if (feat->status == Feature::FeatureStatus::VALID && p && p->status == Point::PointStatus::ACTIVE) {
                auto ph = p->mpPH;
                for (auto &r : ph->residuals) {
                    if (!r->isLinearized) {
                        activeResiduals.push_back(r);
                        r->resetOOB();
                    }
                    else {
                        numLRes++;
                    }
                }
            }
            numPoints++;
        }
    }

    LOG(INFO) << "active residuals: " << activeResiduals.size() << endl;
    // step3: 计光度残差的雅克比矩阵J_I, J_geo, J_photo 和 error,同时剔除外点到I0, I1的约束
    Vec3 lastEnergy = linearizeAll(false);
    // step3.1 计算视觉光度残差
    double lastEnergyL = calcLEnergy(); //calculate the point frame photometric error, L indicate landmark
    // step3.2 计算先验残差
    double lastEnergyM = calcMEnergy(); //calculate the marginalize prior error, M indicate marginalize

    // apply res, calculate some hessian block, in other world, calculate Hessian block of the [pose, a, b]-[inverse depth]
    // step3.3 计算相对位姿，相对光度和逆深度的hessian block
    if (multiThreading)
        threadReduce.reduce(bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0,
                            activeResiduals.size(), 50);
    else
        applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0);

    printOptRes(lastEnergy, lastEnergyL, lastEnergyM, 0, 0, frames.back()->frameHessian->aff_g2l().a,
                frames.back()->frameHessian->aff_g2l().b);

    double lambda = 1e-1;
    float stepsize = 1;
    VecX previousX = VecX::Constant(CPARS + 8 * frames.size(), NAN);

    //step4 LM迭代优化
    for (int iteration = 0; iteration < mnumOptIts; iteration++) {
        // solve!
        // step4.1 保存迭代之前的状态
        backupState(iteration != 0);
        // step4.2 求解正规方程Hx = b
        solveSystem(iteration, lambda);
        // step4.3 判断迭代前后两次的增量夹角的余弦值
        double incDirChange = (1e-20 + previousX.dot(ef->lastX)) / (1e-20 + previousX.norm() * ef->lastX.norm());
        previousX = ef->lastX;  // 更新上一次的增量
        // step4.4 计算步长值
        if (std::isfinite(incDirChange) && (setting_solverMode & SOLVER_STEPMOMENTUM)) {
            float newStepsize = exp(incDirChange * 1.4);
            if (incDirChange < 0 && stepsize > 1)
                stepsize = 1;

            stepsize = sqrtf(sqrtf(newStepsize * stepsize * stepsize * stepsize));
            if (stepsize > 2)
                stepsize = 2;
            if (stepsize < 0.25)
                stepsize = 0.25;
        }
        //step4.5 从上一时刻的状态叠加计算的增量来更新优化变量的状态
        bool canbreak = doStepFromBackup(stepsize, stepsize, stepsize, stepsize, stepsize);

        // eval new energy!
        // step4.6 计算状态更新后的误差

        Vec3 newEnergy = linearizeAll(false); //计光度残差的雅克比矩阵J_I, J_geo, J_photo 和 error,同时剔除外点到I0, I1的约束
        double newEnergyL = calcLEnergy(); // 计算视觉残差
        double newEnergyM = calcMEnergy(); // 计算先验残差

        printOptRes(newEnergy, newEnergyL, newEnergyM, 0, 0, frames.back()->frameHessian->aff_g2l().a,
                    frames.back()->frameHessian->aff_g2l().b);

        //ste4.7 control the lambda in LM
        if (setting_forceAceptStep || (newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM <
            lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM)) {

            // energy is decreasing
            // 计算在新的相对位姿，相对光度和逆深度的hessian block
            if (multiThreading)
                threadReduce.reduce(bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0,
                                    activeResiduals.size(), 50);
            else
                applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0);
            // 更新上一次的状态能量值
            lastEnergy = newEnergy;
            lastEnergyL = newEnergyL;
            lastEnergyM = newEnergyM;
            // 减小惩罚系数
            lambda *= 0.25;
        }
        else {
            // energy increses, reload the backup state and increase lambda
            loadSateBackup();
            lastEnergy = linearizeAll(false);
            lastEnergyL = calcLEnergy();
            lastEnergyM = calcMEnergy();
            // 增大惩罚系数
            lambda *= 1e2;
        }
        // 判断优化是否可以终止
        if (canbreak && iteration >= setting_minOptIterations)
            break;
    }//step4 LM迭代优化结束

    Vec10 newStateZero = Vec10::Zero();
    newStateZero.segment<2>(6) = frames.back()->frameHessian->get_state().segment<2>(6);
    //ste5： 设置最新帧的位姿和仿射变换系数的线性化点
    frames.back()->frameHessian->setEvalPT(frames.back()->frameHessian->PRE_worldToCam, newStateZero);

    EFDeltaValid = false;
    EFAdjointsValid = false;
    //step6. 计算伴随矩阵
    ef->setAdjointsF(Hcalib->mpCH);
    //step7: 主要计算相机的相对位姿
    setPrecalcValues();
    //step8: 求解每个pattern对应的jacobians:(J_I, J_geo, J_photo)和光度误差，
    //将3d点到I0, I1不好的视觉残差因子去掉
    lastEnergy = linearizeAll(true); // fix all the linearizations
    //step9: 判断追踪是否失败
    if (!std::isfinite((double) lastEnergy[0]) || !std::isfinite((double) lastEnergy[1]) ||
        !std::isfinite((double) lastEnergy[2])) {
        LOG(WARNING) << "KF Tracking failed: LOST!";
        isLost = true;
    }
    //step10: 更新整个滑窗中帧的位姿
    // set the estimated pose into frame
    {
        unique_lock<mutex> crlock(shellPoseMutex);
        for (auto fr : frames) {
            fr->setPose(fr->frameHessian->PRE_camToWorld.inverse());
            if (fr->kfId >= globalMap->getLatestOptimizedKfId()) {
                fr->setPoseOpti(Sim3(fr->getPose().matrix()));
            }
            fr->aff_g2l = fr->frameHessian->aff_g2l();
        }
    }
    //step11: 返回激活点的平均能量
    return sqrtf((float) (lastEnergy[0] / (patternNum * ef->resInA)));
}

void FullSystem::setGammaFunction(float *BInv)
{

    if (BInv == nullptr)
        return;

    // copy BInv.
    memcpy(Hcalib->mpCH->Binv, BInv, sizeof(float) * 256);

    // invert.
    for (int i = 0; i < 255; i++) {

        // find val, such that Binv[val] = i.
        // I dont care about speed for this, so do it the stupid way.

        for (int s = 1; s < 255; s++) {
            if (BInv[s] <= i && BInv[s + 1] >= i) {
                Hcalib->mpCH->B[i] = s + (i - BInv[s]) / (BInv[s + 1] - BInv[s]);
                break;
            }
        }
    }

    Hcalib->mpCH->B[0] = 0;
    Hcalib->mpCH->B[255] = 255;
}

shared_ptr<PointHessian>
FullSystem::optimizeImmaturePoint(shared_ptr<internal::ImmaturePoint> point, int minObs,
                                  vector<shared_ptr<ImmaturePointTemporaryResidual>> &residuals)
{
    int nres = 0;
    shared_ptr<Frame> hostFrame = point->feature->host.lock();
    assert(hostFrame); // the feature should have a host frame
    // step1: 给未成熟点添加测量值
    for (auto fr : frames) {
        if (fr != hostFrame) {
            residuals[nres]->state_NewEnergy = residuals[nres]->state_energy = 0;
            residuals[nres]->state_NewState = ResState::OUTLIER;
            residuals[nres]->state_state = ResState::IN;
            residuals[nres]->target = fr->frameHessian;
            nres++;
        }
    }
    assert(nres == frames.size() - 1);
    // step2: 设置迭代的初始状态
    float lastEnergy = 0;
    float lastHdd = 0;
    float lastbd = 0;
    float currentIdepth = (point->idepth_max + point->idepth_min) * 0.5f;

    // step3: 设置迭代初始的H,b矩阵: Hdd bdd, 将Residual新的状态赋值给旧的状态
    for (int i = 0; i < nres; i++) {
        lastEnergy += point->linearizeResidual(Hcalib->mpCH, 1000, residuals[i], lastHdd, lastbd, currentIdepth);
        residuals[i]->state_state = residuals[i]->state_NewState;
        residuals[i]->state_energy = residuals[i]->state_NewEnergy;
    }
    // 如果旧的误差值太大或者lasstHdd(信息矩阵)过大,直接返回
    if (!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act) {
        // !!! 最好写成 return nullptr
        return 0;
    }

    // do LM iteration for this immature point
    float lambda = 0.1;
    for (int iteration = 0; iteration < setting_GNItsOnPointActivation; iteration++) {
        float H = lastHdd;
        H *= 1 + lambda;
        float step = (1.0 / H) * lastbd;  //delta = H^{-1}*b
        float newIdepth = currentIdepth - step; // x_new = x_old - delta

        float newHdd = 0;
        float newbd = 0;
        float newEnergy = 0;
        for (int i = 0; i < nres; i++) {
            // compute the energy in other frames
            newEnergy += point->linearizeResidual(Hcalib->mpCH, 1, residuals[i], newHdd, newbd, newIdepth);
        }

        if (!std::isfinite(lastEnergy) || newHdd < setting_minIdepthH_act) {
            // !!! notice 最好写成 return nullptr
            return 0;
        }

        if (newEnergy < lastEnergy) {
            currentIdepth = newIdepth;
            lastHdd = newHdd;
            lastbd = newbd;
            lastEnergy = newEnergy;
            for (int i = 0; i < nres; i++) {
                residuals[i]->state_state = residuals[i]->state_NewState;
                residuals[i]->state_energy = residuals[i]->state_NewEnergy;
            }
            lambda *= 0.5;
        }
        else {
            lambda *= 5;
        }

        // 如果增量变得很小,迭代优化提前终止
        if (fabsf(step) < 0.0001 * currentIdepth)
            break;
    }

    if (!std::isfinite(currentIdepth)) {
        return nullptr;
    }

    int numGoodRes = 0;
    for (int i = 0; i < nres; i++)
        if (residuals[i]->state_state == ResState::IN)
            numGoodRes++;
    // 如果一个点的好的观测值太小,直接返回
    if (numGoodRes < minObs) {
        // an outlier
        return nullptr;
    }

    point->feature->CreateFromImmature(); // create a point from immature feature
    shared_ptr<PointHessian> p = point->feature->point->mpPH;

    // set residual status in new map point
    p->lastResiduals[0].first = nullptr;
    p->lastResiduals[0].second = ResState::OOB;
    p->lastResiduals[1].first = nullptr;
    p->lastResiduals[1].second = ResState::OOB;

    p->setIdepthZero(currentIdepth); // 设置3d点的线性化点
    p->setIdepth(currentIdepth);     // 设置3d点的当前状态

    // move the immature point residuals into the new map point
    for (int i = 0; i < nres; i++)
        if (residuals[i]->state_state == ResState::IN) {
            shared_ptr<FrameHessian> host = point->feature->host.lock()->frameHessian;
            shared_ptr<FrameHessian> target = residuals[i]->target.lock();
            shared_ptr<PointFrameResidual> r(new PointFrameResidual(p, host, target));

            r->state_NewEnergy = r->state_energy = 0;
            r->state_NewState = ResState::OUTLIER;
            r->setState(ResState::IN);
            p->residuals.push_back(r);

            //  lastResiduals[0] 存储3d点到最新帧I1残差的状态
            if (target == frames.back()->frameHessian) {
                p->lastResiduals[0].first = r;
                p->lastResiduals[0].second = ResState::IN;
            } // lastResiduals[1] 存储3d点到次新帧的残差的状态
            else if (target == (frames.size() < 2 ? nullptr : frames[frames.size() - 2]->frameHessian)) {
                p->lastResiduals[1].first = r;
                p->lastResiduals[1].second = ResState::IN;
            }
        }

    return p;
}

void FullSystem::traceNewCoarse(shared_ptr<FrameHessian> fh)
{

    unique_lock<mutex> lock(mapMutex);

    int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0, trace_skip = 0, trace_badcondition = 0,
        trace_uninitialized = 0;

    Mat33f K = Mat33f::Identity();
    K(0, 0) = Hcalib->mpCH->fxl();
    K(1, 1) = Hcalib->mpCH->fyl();
    K(0, 2) = Hcalib->mpCH->cxl();
    K(1, 2) = Hcalib->mpCH->cyl();

    //Step1: 遍历滑窗中的每一关键帧(host)
    for (shared_ptr<Frame> fr : frames) {
        //Step2: 获取host关键帧对应的FrameHessian
        shared_ptr<FrameHessian> host = fr->frameHessian;

        //Step3: 计算host 和target(最新帧)的一些参数
        //计算相对的几何参数
        SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
        Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
        Vec3f Kt = K * hostToNew.translation().cast<float>();
        //计算相对的光度变换参数
        Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(),
                                                fh->aff_g2l())
            .cast<float>();
        //Step4： 遍历host frame 中的feature
        for (auto feat : fr->features) {
            // update the status of immuature points
            // 判断feature是否是未成熟点
            if (feat->status == Feature::FeatureStatus::IMMATURE && feat->ip) {
                // update the immature points
                shared_ptr<ImmaturePoint> ph = feat->ip;
                ph->traceOn(fh, KRKi, Kt, aff, Hcalib->mpCH);

                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD)
                    trace_good++;
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION)
                    trace_badcondition++;
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB)
                    trace_oob++;
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
                    trace_out++;
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED)
                    trace_skip++;
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED)
                    trace_uninitialized++;
                trace_total++;
            }
        }
    }
}

void FullSystem::activatePointsMT()
{
    // this will turn immature points into real points
    // step1: 根据能量函数中激活的点数动态的调节currentMinActDist
    // ！！！ notice nPoints 好用只有-- 没有++ 是不是一个bug
    if (ef->nPoints < setting_desiredPointDensity * 0.66)
        currentMinActDist -= 0.8;
    if (ef->nPoints < setting_desiredPointDensity * 0.8)
        currentMinActDist -= 0.5;
    else if (ef->nPoints < setting_desiredPointDensity * 0.9)
        currentMinActDist -= 0.2;
    else if (ef->nPoints < setting_desiredPointDensity)
        currentMinActDist -= 0.1;

    if (ef->nPoints > setting_desiredPointDensity * 1.5)
        currentMinActDist += 0.8;
    if (ef->nPoints > setting_desiredPointDensity * 1.3)
        currentMinActDist += 0.5;
    if (ef->nPoints > setting_desiredPointDensity * 1.15)
        currentMinActDist += 0.2;
    if (ef->nPoints > setting_desiredPointDensity)
        currentMinActDist += 0.1;

    if (currentMinActDist < 0)
        currentMinActDist = 0;
    if (currentMinActDist > 4)
        currentMinActDist = 4;

    auto newestFr = frames.back();
    vector<shared_ptr<FrameHessian>> frameHessians;
    for (auto fr : frames)
        frameHessians.push_back(fr->frameHessian);

    //step2: make dist map
    coarseDistanceMap->makeK(Hcalib->mpCH);
    coarseDistanceMap->makeDistanceMap(frameHessians, newestFr->frameHessian);

    //step3 查找滑窗中除最后一帧的满足一定条件的immature point转化为激活点的候选点
    vector<shared_ptr<ImmaturePoint>> toOptimize;
    toOptimize.reserve(20000);

    //step3.1 enumerate all frames in sliding window except newest frame
    for (auto host : frameHessians) {
        if (host == newestFr->frameHessian)
            continue;

        SE3 fhToNew = newestFr->frameHessian->PRE_worldToCam * host->PRE_camToWorld;
        Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
        Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());
        //step3.2 enumerate all immature points in host frame
        for (size_t i = 0; i < host->frame->features.size(); i++) {
            shared_ptr<Feature> &feat = host->frame->features[i];
            if (feat->status == Feature::FeatureStatus::IMMATURE && feat->ip) {

                shared_ptr<Feature> &feat = host->frame->features[i];
                shared_ptr<ImmaturePoint> &ph = host->frame->features[i]->ip;
                ph->idxInImmaturePoints = i;

                //step3.2.1 判断条件1 delete points that have never been traced successfully, or that are outlier on the last trace.
                if (!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER) {
                    feat->status = Feature::FeatureStatus::OUTLIER;
                    feat->ReleaseImmature();
                    continue;
                }
                //step3.2.2 判断条件2
                bool canActivate = (ph->lastTraceStatus == IPS_GOOD || ph->lastTraceStatus == IPS_SKIPPED ||
                    ph->lastTraceStatus == IPS_BADCONDITION || ph->lastTraceStatus == IPS_OOB) &&
                    ph->lastTracePixelInterval < 8 && ph->quality > setting_minTraceQuality &&
                    (ph->idepth_max + ph->idepth_min) > 0;

                if (!canActivate) {
                    // if point will be out afterwards, delete it instead.
                    // 如果immature point 所在的帧即将要被边缘化,或者上一次追踪状态为IPS_OOB, 那么将点标记为外点
                    if (ph->feature->host.lock()->frameHessian->flaggedForMarginalization ||
                        ph->lastTraceStatus == IPS_OOB) {
                        feat->status = Feature::FeatureStatus::OUTLIER;
                        feat->ReleaseImmature();
                    }
                    continue;
                }
                //step3.2.3 判断条件3 使用距离图
                // see if we need to activate point due to distance map.
                Vec3f ptp = KRKi * Vec3f(feat->uv[0], feat->uv[1], 1) +
                    Kt * (0.5f * (ph->idepth_max + ph->idepth_min));
                int u = ptp[0] / ptp[2] + 0.5f;
                int v = ptp[1] / ptp[2] + 0.5f;

                if ((u > 0 && v > 0 && u < wG[1] && v < hG[1])) {

                    float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u + wG[1] * v] +
                        (ptp[0] - floorf((float) (ptp[0])));

                    // NOTE: the shit my_type is used here, my_type 表示的feature在图像金字塔的层级
                    if (dist >= currentMinActDist * ph->my_type) {
                        // 将当前点加入到距离图途中, 更近距离图
                        coarseDistanceMap->addIntoDistFinal(u, v);
                        // 将要转化为active point的 immature point 加入到toOptimize 数组中
                        toOptimize.push_back(ph);
                    }
                }
                else {
                    // drop it
                    feat->status = Feature::FeatureStatus::OUTLIER;
                    feat->ReleaseImmature();
                }
            }
        }
    }

    vector<shared_ptr<PointHessian>> optimized;
    optimized.resize(toOptimize.size());
    //step4: 将候选点active point 转化为真正是的 active point
    // this will actually turn immature points into point hessians
    if (multiThreading) {
        threadReduce.reduce(
            bind(&FullSystem::activatePointsMT_Reductor, this, &optimized, &toOptimize, _1, _2, _3, _4), 0,
            toOptimize.size(), 50);
    }
    else {
        activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);
    }

    for (size_t k = 0; k < toOptimize.size(); k++) {
        shared_ptr<PointHessian> newpoint = optimized[k];
        shared_ptr<ImmaturePoint> ph = toOptimize[k];

        if (newpoint != nullptr) {

            // remove the immature point
            ph->feature->status = Feature::FeatureStatus::VALID;

            ph->feature->point->mpPH = newpoint;
            ph->feature->ReleaseImmature();
            newpoint->takeData();

            for (auto r : newpoint->residuals)
                ef->insertResidual(r);
        }
        else if (newpoint == nullptr || ph->lastTraceStatus == IPS_OOB) {
            ph->feature->status = Feature::FeatureStatus::OUTLIER;
            ph->feature->ReleaseImmature();
        }
    }
}

void FullSystem::activatePointsMT_Reductor(
    std::vector<shared_ptr<PointHessian>> *optimized,
    std::vector<shared_ptr<ImmaturePoint>> *toOptimize,
    int min, int max, Vec10 *stats, int tid)
{

    vector<shared_ptr<ImmaturePointTemporaryResidual>> tr(frames.size(), nullptr);

    for (auto &t : tr) {
        // create residual
        t = shared_ptr<ImmaturePointTemporaryResidual>(new ImmaturePointTemporaryResidual());
    }

    for (int k = min; k < max; k++) {
        (*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k], 1, tr);
    }
}

void FullSystem::flagPointsForRemoval()
{

    assert(EFIndicesValid);
    std::vector<shared_ptr<FrameHessian>>
        fhsToMargPoints;

    for (int i = 0; i < (int) frames.size(); i++)
        if (frames[i]->frameHessian->flaggedForMarginalization)
            fhsToMargPoints.push_back(frames[i]->frameHessian);

    int flag_oob = 0, flag_in = 0, flag_inin = 0, flag_nores = 0;

    // go through all active frames
    for (auto &fr : frames) {
        shared_ptr<FrameHessian> host = fr->frameHessian;
        for (auto &feat : fr->features) {
            if (feat->status == Feature::FeatureStatus::VALID &&
                feat->point->status == Point::PointStatus::ACTIVE) {

                shared_ptr<PointHessian> ph = feat->point->mpPH;

                if (ph->idepth_scaled < 0 || ph->residuals.size() == 0) {
                    // no residuals or idepth invalid
                    ph->point->status = Point::PointStatus::OUTLIER;
                    feat->status = Feature::FeatureStatus::OUTLIER;
                    flag_nores++;
                }
                else if (ph->isOOB(fhsToMargPoints) || host->flaggedForMarginalization) {
                    // the point goes out the boundary, or the host frame is marged
                    flag_oob++;
                    if (ph->isInlierNew()) {
                        flag_in++;
                        int ngoodRes = 0;
                        for (auto r : ph->residuals) {
                            r->resetOOB();
                            r->linearize(this->Hcalib->mpCH);
                            r->isLinearized = false;
                            r->applyRes(true);
                            if (r->isActive()) {
                                r->fixLinearizationF(this->ef);
                                ngoodRes++;
                            }
                        }

                        if (ph->idepth_hessian > setting_minIdepthH_marg) {
                            // idepth is good, margin it.
                            flag_inin++;
                            ph->point->status = Point::PointStatus::MARGINALIZED;
                        }
                        else {
                            // idepth not good, drop it.
                            ph->point->status = Point::PointStatus::OUT;
                        }
                    }
                    else {
                        // not active, drop it.
                        ph->point->status = Point::PointStatus::OUT;
                    }
                }
            }
        }
    }

    LOG(INFO) << "Flag: nores: " << flag_nores << ", oob: " << flag_oob << ", marged: " << flag_inin << endl;
}

void FullSystem::makeNewTraces(shared_ptr<FrameHessian> newFrame, float *gtDepth)
{

    if (setting_pointSelection == 1) {
        LOG(INFO) << "using LDSO point selection strategy " << endl;
        newFrame->frame->features.reserve(setting_desiredImmatureDensity);
        detector.DetectCorners(setting_desiredImmatureDensity, newFrame->frame);
        for (auto &feat : newFrame->frame->features) {
            // create a immature point
            feat->ip = shared_ptr<ImmaturePoint>(
                new ImmaturePoint(newFrame->frame, feat, 1, Hcalib->mpCH));
        }
        LOG(INFO) << "new features features created: " << newFrame->frame->features.size() << endl;
    }
    else if (setting_pointSelection == 0) {
        LOG(INFO) << "using original DSO point selection strategy" << endl;
        pixelSelector->allowFast = true;
        int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap, setting_desiredImmatureDensity);
        newFrame->frame->features.reserve(numPointsTotal);

        for (int y = patternPadding + 1; y < hG[0] - patternPadding - 2; y++)
            for (int x = patternPadding + 1; x < wG[0] - patternPadding - 2; x++) {
                int i = x + y * wG[0];
                if (selectionMap[i] == 0)
                    continue;

                shared_ptr<Feature> feat(new Feature(x, y, newFrame->frame));
                feat->ip = shared_ptr<ImmaturePoint>(
                    new ImmaturePoint(newFrame->frame, feat, selectionMap[i], Hcalib->mpCH));
                if (!std::isfinite(feat->ip->energyTH)) {
                    feat->ReleaseAll();
                    continue;
                }
                else
                    newFrame->frame->features.push_back(feat);
            }
        LOG(INFO) << "new features features created: " << newFrame->frame->features.size() << endl;
    }
    else if (setting_pointSelection == 2) {
        // random pick
        LOG(INFO) << "using random point selection strategy" << endl;
        cv::RNG rng;
        newFrame->frame->features.reserve(setting_desiredImmatureDensity);
        for (int i = 0; i < setting_desiredImmatureDensity; i++) {
            int x = rng.uniform(20, wG[0] - 20);
            int y = rng.uniform(20, hG[0] - 20);
            shared_ptr<Feature> feat(new Feature(x, y, newFrame->frame));
            feat->ip = shared_ptr<ImmaturePoint>(
                new ImmaturePoint(newFrame->frame, feat, 1, Hcalib->mpCH));
            if (!std::isfinite(feat->ip->energyTH)) {
                feat->ReleaseAll();
                continue;
            }
            else
                newFrame->frame->features.push_back(feat);
        }
        LOG(INFO) << "new features features created: " << newFrame->frame->features.size() << endl;
    }
}

void FullSystem::initializeFromInitializer(shared_ptr<FrameHessian> newFrame)
{
    unique_lock<mutex> lock(mapMutex);

    shared_ptr<FrameHessian> firstFrame = coarseInitializer->firstFrame;
    shared_ptr<Frame> fr = firstFrame->frame;
    firstFrame->idx = frames.size();

    frames.push_back(fr);
    firstFrame->frameID = globalMap->NumFrames();
    ef->insertFrame(firstFrame, Hcalib->mpCH);
    setPrecalcValues();

    fr->features.reserve(wG[0] * hG[0] * 0.2f);

    float sumID = 1e-5, numID = 1e-5;
    for (int i = 0; i < coarseInitializer->numPoints[0]; i++) {
        sumID += coarseInitializer->points[0][i].iR;
        numID++;
    }
    float rescaleFactor = 1 / (sumID / numID);

    // randomly sub-select the points I need.
    float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[0];

    LOG(INFO) << "Initialization: keep " << 100 * keepPercentage << " (need " << setting_desiredPointDensity
              << ", have " << coarseInitializer->numPoints[0] << ")!" << endl;

    // Create features in the first frame.
    for (size_t i = 0; i < size_t(coarseInitializer->numPoints[0]); i++) {

        if (rand() / (float) RAND_MAX > keepPercentage)
            continue;
        Pnt *point = coarseInitializer->points[0] + i;

        shared_ptr<Feature> feat(new Feature(point->u + 0.5f, point->v + 0.5f, firstFrame->frame));
        feat->ip = shared_ptr<ImmaturePoint>(
            new ImmaturePoint(firstFrame->frame, feat, point->my_type, Hcalib->mpCH));

        if (!std::isfinite(feat->ip->energyTH)) {
            feat->ReleaseImmature();
            continue;
        }

        feat->CreateFromImmature();
        shared_ptr<PointHessian> ph = feat->point->mpPH;
        if (!std::isfinite(ph->energyTH)) {
            feat->ReleaseMapPoint();
            continue;
        }
        feat->ReleaseImmature(); // no longer needs the immature part
        fr->features.push_back(feat);

        ph->setIdepthScaled(point->iR * rescaleFactor);
        ph->setIdepthZero(ph->idepth);
        ph->hasDepthPrior = true;
        ph->point->status = Point::PointStatus::ACTIVE;
        ph->takeData(); // set the idepth into optimization
    }

    SE3 firstToNew = coarseInitializer->thisToNext;
    firstToNew.translation() /= rescaleFactor;

    // really no lock required, as we are initializing.
    {
        unique_lock<mutex> crlock(shellPoseMutex);
        firstFrame->setEvalPT_scaled(fr->getPose(), firstFrame->frame->aff_g2l);
        newFrame->frame->setPose(firstToNew);
        newFrame->setEvalPT_scaled(newFrame->frame->getPose(), newFrame->frame->aff_g2l);
    }

    initialized = true;
    globalMap->AddKeyFrame(fr);
    LOG(INFO) << "Initialized from initializer, points: " << firstFrame->frame->features.size() << endl;
}

void FullSystem::removeOutliers()
{
    int numPointsDropped = 0;
    for (auto &fr : frames) {
        for (auto &feat : fr->features) {
            if (feat->status == Feature::FeatureStatus::VALID && feat->point &&
                feat->point->status == Point::PointStatus::ACTIVE) {

                shared_ptr<PointHessian> ph = feat->point->mpPH;
                if (ph->residuals.empty()) {
                    ph->point->status = Point::PointStatus::OUTLIER;
                    feat->status = Feature::FeatureStatus::OUTLIER;
                    numPointsDropped++;
                }
            }
        }
    }

    LOG(INFO) << "remove outliers done, outliers: " << numPointsDropped << endl;
    ef->dropPointsF();
}

void FullSystem::setPrecalcValues()
{
    // step1: 主要计算相机的相对位姿: 线性点(计算雅克比矩阵)、和当前点(计算误差)
    // step1.1: 遍历滑窗中的每一帧 host frame
    for (auto &fr : frames) {
        // step1.2 : host frame 中对应的FrameHessian中存储的targetPrecalc resize到滑窗帧的大小
        fr->frameHessian->targetPrecalc.resize(frames.size());
        // step1.3 主要计算相机的相对位姿: 线性点(计算雅克比矩阵)、和当前点(计算误差)
        for (size_t i = 0; i < frames.size(); i++)
            fr->frameHessian->targetPrecalc[i].Set(fr->frameHessian, frames[i]->frameHessian, Hcalib->mpCH);
    }
    // step1.2 计算当前状态计算的到的相对位姿相对于线性点出求得的相对位姿的增量,更新pose, 相机内参和3d点当前状态到线性点的偏移
    ef->setDeltaF(Hcalib->mpCH);
}

void FullSystem::solveSystem(int iteration, double lambda)
{
    //step1 calculate the nullspace of the state variable
    ef->lastNullspaces_forLogging = getNullspaces(
        ef->lastNullspaces_pose,
        ef->lastNullspaces_scale,
        ef->lastNullspaces_affA,
        ef->lastNullspaces_affB);
    //step2
    ef->solveSystemF(iteration, lambda, Hcalib->mpCH);
}

Vec3 FullSystem::linearizeAll(bool fixLinearization)
{

    double lastEnergyP = 0;
    double lastEnergyR = 0;
    double num = 0;

    std::vector<shared_ptr<PointFrameResidual>>
        toRemove[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++)
        toRemove[i].clear();
    // step1: 求解每个pattern对应的jacobians:(J_I, J_geo, J_photo)和光度误差
    if (multiThreading) {
        threadReduce.reduce(
            bind(&FullSystem::linearizeAll_Reductor, this, fixLinearization, toRemove, _1, _2, _3, _4),
            0, activeResiduals.size(), 0);
        lastEnergyP = threadReduce.stats[0];
    }
    else {
        Vec10 stats;
        linearizeAll_Reductor(fixLinearization, toRemove, 0, activeResiduals.size(), &stats, 0);
        // 统计每个点的误差函数值
        lastEnergyP = stats[0];
    }
    //step2: 设定阈值
    setNewFrameEnergyTH();
    //step3: 如果要固定线性化点, 需要更新到激活点与I0, I1帧的残差状态
    if (fixLinearization) {

        for (auto r : activeResiduals) {
            shared_ptr<PointHessian> ph = r->point.lock();
            if (ph->lastResiduals[0].first == r)
                ph->lastResiduals[0].second = r->state_state;
            else if (ph->lastResiduals[1].first == r)
                ph->lastResiduals[1].second = r->state_state;
        }
        //将3d点到I0, I1不好的视觉残差因子去掉
        int nResRemoved = 0;
        for (int i = 0; i < NUM_THREADS; i++) {
            for (auto r : toRemove[i]) {
                shared_ptr<PointHessian> ph = r->point.lock();

                if (ph->lastResiduals[0].first == r)
                    ph->lastResiduals[0].first = 0;
                else if (ph->lastResiduals[1].first == r)
                    ph->lastResiduals[1].first = 0;

                ef->dropResidual(r); // this will actually remove the residual
                nResRemoved++;
            }
        }
    }
    return Vec3(lastEnergyP, lastEnergyR, num);
}

void FullSystem::linearizeAll_Reductor(
    bool fixLinearization, std::vector<shared_ptr<PointFrameResidual>>

*toRemove,
    int min,
    int max, Vec10 *stats,
    int tid)
{

    for (int k = min; k < max; k++) {
        shared_ptr<PointFrameResidual> r = activeResiduals[k];
        // 求解每个pattern 对应的J_I, J_geo 和J_photo ,同时进行外点检测和剔除
        // notice , J_geo在状态的线性化点求解, J_photo, J_I在当前状态进行求解
        (*stats)[0] += r->linearize(Hcalib->mpCH);
        // 固定线性化点, 边缘化时会用到, notice：只会对激活点对应的光度残差因子进行边缘化,
        // TODO: 有待考虑,会把他们放入到toRemove数组中,意味着视觉残差中只有激活点的光度残差
        if (fixLinearization) {
            //update residual state and calculate Hessian block of the [pose, a, b]-[inverse depth]
            r->applyRes(true);

            if (r->isActive()) {
                // isNew is always true
                if (r->isNew) {
                    shared_ptr<PointHessian> p = r->point.lock();
                    shared_ptr<FrameHessian> host = r->host.lock();
                    shared_ptr<FrameHessian> target = r->target.lock();
                    Vec3f ptp_inf = host->targetPrecalc[target->idx].PRE_KRKiTll *
                        Vec3f(p->u, p->v, 1); // projected point assuming infinite depth.
                    Vec3f ptp = ptp_inf + host->targetPrecalc[target->idx].PRE_KtTll *
                        p->idepth_scaled; // projected point with real depth.
                    // TODO:
                    float relBS = 0.01 * ((ptp_inf.head<2>() / ptp_inf[2]) -
                        (ptp.head<2>() / ptp[2])).norm(); // 0.01 = one pixel.

                    if (relBS > p->maxRelBaseline)
                        p->maxRelBaseline = relBS;

                    p->numGoodResiduals++;
                }
            }
            else {
                //TODO: 将不好的视觉残差扔掉,
                toRemove[tid].push_back(activeResiduals[k]);
            }
        }
    }
}

// applies step to linearization point.
bool FullSystem::doStepFromBackup(float stepfacC, float stepfacT, float stepfacR, float stepfacA, float stepfacD)
{

    Vec10 pstepfac;
    pstepfac.segment<3>(0).setConstant(stepfacT);
    pstepfac.segment<3>(3).setConstant(stepfacR);
    pstepfac.segment<4>(6).setConstant(stepfacA);

    float sumA = 0, sumB = 0, sumT = 0, sumR = 0, sumID = 0, numID = 0;

    float sumNID = 0;

    if (setting_solverMode & SOLVER_MOMENTUM) {
        //step1: 更新相机的内参
        Hcalib->mpCH->setValue(Hcalib->mpCH->value_backup + Hcalib->mpCH->step);
        for (auto &fr : frames) {
            auto fh = fr->frameHessian;
            Vec10 step = fh->step;
            //step2: 计算帧pose帧的增量
            step.head<6>() += 0.5f * (fh->step_backup.head<6>());
            //step3: 更新帧的位姿
            fh->setState(fh->state_backup + step);
            //step4: 计算帧的affine transform 参数 a,b和位姿增量的二范数
            sumA += step[6] * step[6];
            sumB += step[7] * step[7];
            sumT += step.segment<3>(0).squaredNorm();
            sumR += step.segment<3>(3).squaredNorm();
            //step5: 更新3d点逆深度和线性化点
            for (auto feat : fr->features) {
                if (feat->status == Feature::FeatureStatus::VALID && feat->point &&
                    feat->point->status == Point::PointStatus::ACTIVE) {

                    auto ph = feat->point->mpPH;
                    float step = ph->step + 0.5f * (ph->step_backup);
                    ph->setIdepth(ph->idepth_backup + step);
                    sumID += step * step;
                    sumNID += fabsf(ph->idepth_backup);
                    numID++;
                    // Notice 更新3d点的线性化点为最新状态
                    ph->setIdepthZero(ph->idepth_backup + step);
                }
            }
        }
    }
    else {
        //更新相机的内参
        Hcalib->mpCH->setValue(Hcalib->mpCH->value_backup + stepfacC * Hcalib->mpCH->step);
        for (auto &fr : frames) {
            auto fh = fr->frameHessian;
            fh->setState(fh->state_backup + pstepfac.cwiseProduct(fh->step));
            sumA += fh->step[6] * fh->step[6];
            sumB += fh->step[7] * fh->step[7];
            sumT += fh->step.segment<3>(0).squaredNorm();
            sumR += fh->step.segment<3>(3).squaredNorm();

            for (auto feat : fr->features) {
                if (feat->status == Feature::FeatureStatus::VALID && feat->point &&
                    feat->point->status == Point::PointStatus::ACTIVE) {
                    auto ph = feat->point->mpPH;
                    ph->setIdepth(ph->idepth_backup + stepfacD * ph->step);
                    sumID += ph->step * ph->step;
                    sumNID += fabsf(ph->idepth_backup);
                    numID++;
                    ph->setIdepthZero(ph->idepth_backup + stepfacD * ph->step);
                }
            }
        }
    }
    //step6: 计算增量的平局值
    sumA /= frames.size();
    sumB /= frames.size();
    sumR /= frames.size();
    sumT /= frames.size();
    sumID /= numID;
    sumNID /= numID;

    EFDeltaValid = false;
    //step7:  主要计算相机的相对位姿: 线性点(计算雅克比矩阵)、和当前点(计算误差),
    // 计算当前状态计算的到的相对位姿相对于线性点出求得的相对位姿的增量,更新pose,
    // 相机内参和3d点当前状态到线性点的偏移
    setPrecalcValues();

    return sqrtf(sumA) < 0.0005 * setting_thOptIterations &&
        sqrtf(sumB) < 0.00005 * setting_thOptIterations &&
        sqrtf(sumR) < 0.00005 * setting_thOptIterations &&
        sqrtf(sumT) * sumNID < 0.00005 * setting_thOptIterations;
}

void FullSystem::backupState(bool backupLastStep)
{

    if (setting_solverMode & SOLVER_MOMENTUM) {
        if (backupLastStep) {
            Hcalib->mpCH->step_backup = Hcalib->mpCH->step;
            Hcalib->mpCH->value_backup = Hcalib->mpCH->value;
            for (auto &fr : frames) {
                auto fh = fr->frameHessian;
                fh->step_backup = fh->step;
                fh->state_backup = fh->get_state();
                for (auto feat : fr->features) {
                    if (feat->point && feat->point->status == Point::PointStatus::ACTIVE) {
                        auto ph = feat->point->mpPH;
                        ph->idepth_backup = ph->idepth;
                        ph->step_backup = ph->step;
                    }
                }
            }
        }
        else {
            Hcalib->mpCH->step_backup.setZero();
            Hcalib->mpCH->value_backup = Hcalib->mpCH->value;
            for (auto &fr : frames) {
                auto fh = fr->frameHessian;
                fh->step_backup.setZero();
                fh->state_backup = fh->get_state();
                for (auto feat : fr->features) {
                    if (feat->point && feat->point->status == Point::PointStatus::ACTIVE) {
                        auto ph = feat->point->mpPH;
                        ph->idepth_backup = ph->idepth;
                        ph->step_backup = 0;
                    }
                }
            }
        }
    }
    else {
        Hcalib->mpCH->value_backup = Hcalib->mpCH->value;
        for (auto &fr : frames) {
            auto fh = fr->frameHessian;
            fh->state_backup = fh->get_state();
            for (auto feat : fr->features) {
                if (feat->status == Feature::FeatureStatus::VALID &&
                    feat->point->status == Point::PointStatus::ACTIVE) {
                    auto ph = feat->point->mpPH;
                    ph->idepth_backup = ph->idepth;
                }
            }
        }
    }
}

void FullSystem::loadSateBackup()
{

    Hcalib->mpCH->setValue(Hcalib->mpCH->value_backup);
    for (auto fr : frames) {
        auto fh = fr->frameHessian;
        fh->setState(fh->state_backup);
        for (auto feat : fr->features) {
            if (feat->point && feat->point->status == Point::PointStatus::ACTIVE) {
                auto ph = feat->point->mpPH;
                ph->setIdepth(ph->idepth_backup);
                ph->setIdepthZero(ph->idepth_backup);
            }
        }
    }

    EFDeltaValid = false;
    setPrecalcValues();
}

double FullSystem::calcLEnergy()
{
    if (setting_forceAceptStep)
        return 0;
    return ef->calcLEnergyF_MT();
}

double FullSystem::calcMEnergy()
{
    if (setting_forceAceptStep)
        return 0;
    return ef->calcMEnergyF();
}

void FullSystem::applyRes_Reductor(bool copyJacobians, int min, int max, Vec10 *stats, int tid)
{
    for (int k = min; k < max; k++)
        activeResiduals[k]->applyRes(true);
}

std::vector<VecX> FullSystem::getNullspaces(
    std::vector<VecX> &nullspaces_pose,
    std::vector<VecX> &nullspaces_scale,
    std::vector<VecX> &nullspaces_affA,
    std::vector<VecX> &nullspaces_affB)
{

    nullspaces_pose.clear();
    nullspaces_scale.clear();
    nullspaces_affA.clear();
    nullspaces_affB.clear();

    int n = CPARS + frames.size() * 8; // 状态变量的个数
    std::vector<VecX> nullspaces_x0_pre;

    for (int i = 0; i < 6; i++) {
        VecX nullspace_x0(n);
        nullspace_x0.setZero();
        for (auto fr : frames) {
            auto fh = fr->frameHessian;
            nullspace_x0.segment<6>(CPARS + fh->idx * 8) = fh->nullspaces_pose.col(i); // pose scale_state 的零空间
            nullspace_x0.segment<3>(CPARS + fh->idx * 8) *= SCALE_XI_TRANS_INVERSE;    // state 的零空间
            nullspace_x0.segment<3>(CPARS + fh->idx * 8 + 3) *= SCALE_XI_ROT_INVERSE;
        }
        nullspaces_x0_pre.push_back(nullspace_x0);
        nullspaces_pose.push_back(nullspace_x0);
    }
    for (int i = 0; i < 2; i++) {
        VecX nullspace_x0(n);
        nullspace_x0.setZero();
        for (auto fr : frames) {
            auto fh = fr->frameHessian;
            nullspace_x0.segment<2>(CPARS + fh->idx * 8 + 6) =
                fh->nullspaces_affine.col(i).head<2>(); // affine transform scale_state 的零空间
            nullspace_x0[CPARS + fh->idx * 8 + 6] *= SCALE_A_INVERSE; // affine transform state 的零空间
            nullspace_x0[CPARS + fh->idx * 8 + 7] *= SCALE_B_INVERSE;
        }
        nullspaces_x0_pre.push_back(nullspace_x0);
        if (i == 0)
            nullspaces_affA.push_back(nullspace_x0);
        if (i == 1)
            nullspaces_affB.push_back(nullspace_x0);
    }

    VecX nullspace_x0(n);
    nullspace_x0.setZero();
    for (auto fr : frames) {
        auto fh = fr->frameHessian;
        nullspace_x0.segment<6>(CPARS + fh->idx * 8) = fh->nullspaces_scale;   // pose scale_state 尺度的零空间
        nullspace_x0.segment<3>(CPARS + fh->idx * 8) *= SCALE_XI_TRANS_INVERSE; // pose state 尺度的零空间
        nullspace_x0.segment<3>(CPARS + fh->idx * 8 + 3) *= SCALE_XI_ROT_INVERSE;
    }
    nullspaces_x0_pre.push_back(nullspace_x0);
    nullspaces_scale.push_back(nullspace_x0);

    return nullspaces_x0_pre;
}

void FullSystem::setNewFrameEnergyTH()
{

    // collect all residuals and make decision on TH.
    allResVec.clear();
    allResVec.reserve(activeResiduals.size() * 2);
    auto fr = frames.back();
    auto newFrame = fr->frameHessian;

    for (auto &r : activeResiduals)
        if (r->state_NewEnergyWithOutlier >= 0 && r->target.lock() == newFrame) {
            allResVec.push_back(r->state_NewEnergyWithOutlier);
        }

    if (allResVec.size() == 0) {
        newFrame->frameEnergyTH = 12 * 12 * patternNum;
        return; // should never happen, but lets make sure.
    }

    int nthIdx = setting_frameEnergyTHN * allResVec.size();

    assert(nthIdx < (int) allResVec.size());
    assert(setting_frameEnergyTHN < 1);

    std::nth_element(allResVec.begin(), allResVec.begin() + nthIdx, allResVec.end());
    float nthElement = sqrtf(allResVec[nthIdx]);

    newFrame->frameEnergyTH = nthElement * setting_frameEnergyTHFacMedian;
    newFrame->frameEnergyTH = 26.0f * setting_frameEnergyTHConstWeight +
        newFrame->frameEnergyTH * (1 - setting_frameEnergyTHConstWeight);
    newFrame->frameEnergyTH = newFrame->frameEnergyTH * newFrame->frameEnergyTH;
    newFrame->frameEnergyTH *= setting_overallEnergyTHWeight * setting_overallEnergyTHWeight;
}

void FullSystem::printOptRes(const Vec3 &res, double resL, double resM, double resPrior, double LExact, float a,
                             float b)
{
    char buff[256] = {};
    sprintf(buff, "A(%f)=(AV %.3f). Num: A(%'d) + M(%'d); ab %f %f!\n",
            res[0],
            sqrtf((float) (res[0] / (patternNum * ef->resInA))),
            ef->resInA,
            ef->resInM,
            a,
            b);
    LOG(INFO) << string(buff);
}

void FullSystem::mappingLoop()
{

    unique_lock<mutex> lock(trackMapSyncMutex);

    while (runMapping) {

        // wait an unmapped frame
        while (unmappedTrackedFrames.size() == 0) {
            trackedFrameSignal.wait(lock);
            if (!runMapping)
                break;
        }
        if (!runMapping)
            break;

        // get an unmapped frame, tackle it.
        shared_ptr<Frame> fr = unmappedTrackedFrames.front();
        auto fh = fr->frameHessian;
        unmappedTrackedFrames.pop_front();

        // guaranteed to make a KF for the very first two tracked frames.
        if (globalMap->NumFrames() <= 2) {
            lock.unlock();
            makeKeyFrame(fh);
            lock.lock();
            mappedFrameSignal.notify_all();
            continue;
        }

        if (unmappedTrackedFrames.size() > 3)
            needToKetchupMapping = true;

        if (unmappedTrackedFrames.size() > 0) {
            // if there are other frames to track, do that first.
            lock.unlock();
            makeNonKeyFrame(fh);
            lock.lock();

            if (needToKetchupMapping && unmappedTrackedFrames.size() > 0) {
                auto fr = unmappedTrackedFrames.front();
                shared_ptr<FrameHessian> fh = fr->frameHessian;
                unmappedTrackedFrames.pop_front();
                {
                    unique_lock<mutex> crlock(shellPoseMutex);
                    fh->setEvalPT_scaled(fr->getPose(), fh->frame->aff_g2l);
                }
            }
        }
        else {
            if (setting_realTimeMaxKF || needNewKFAfter >= int(frames.back()->id)) {
                lock.unlock();
                makeKeyFrame(fh);
                needToKetchupMapping = false;
                lock.lock();
            }
            else {
                lock.unlock();
                makeNonKeyFrame(fh);
                lock.lock();
            }
        }

        mappedFrameSignal.notify_all();
    }
    LOG(INFO) << "MAPPING FINISHED!";
}

bool FullSystem::saveAll(const string &filename)
{
    ofstream fout(filename, ios::out | ios::binary);
    if (!fout)
        return false;
    int nKF = globalMap->NumFrames();
    fout.write((char *) &nKF, sizeof(int));
    auto allKFs = globalMap->GetAllKFs();
    for (auto &frame : allKFs) {
        frame->save(fout);
    }
    fout.close();
    LOG(INFO) << "DONE!" << endl;
    return true;
}

bool FullSystem::loadAll(const string &filename)
{

    ifstream fin(filename);
    if (!fin)
        return false;
    int numKF = 0;
    fin.read((char *) &numKF, sizeof(numKF));

    vector<shared_ptr<Frame>> allKFs;
    allKFs.resize(numKF, nullptr);
    for (auto &kf : allKFs) {
        kf = shared_ptr<Frame>(new Frame());
    }

    int i = 0;
    while (!fin.eof() && i < int(allKFs.size())) {
        shared_ptr<Frame> &newFrame = allKFs[i];
        newFrame->load(fin, newFrame, allKFs);
        i++;
    }

    fin.close();

    if (viewer)
        viewer->publishKeyframes(allKFs, false, Hcalib->mpCH);

    frames = allKFs;
    for (auto &kf : allKFs) {
        globalMap->AddKeyFrame(kf);
    }

    LOG(INFO) << "Loaded total " << frames.size() << " keyframes" << endl;
    return true;
}

void FullSystem::printResult(const string &filename, bool printOptimized)
{

    unique_lock<mutex> lock(trackMutex);
    unique_lock<mutex> crlock(shellPoseMutex);

    std::ofstream myfile(filename);
    myfile << std::setprecision(15);

    auto allKFs = globalMap->GetAllKFs();
    LOG(INFO) << "total keyframes: " << allKFs.size() << endl;

    for (auto &fr : allKFs) {
        SE3 Twc;
        Sim3 Swc;
        if (printOptimized) {
            Swc = fr->getPoseOpti().inverse();
            Twc = SE3(Swc.rotationMatrix(), Swc.translation());
        }
        else
            Twc = fr->getPose().inverse();

        myfile << fr->timeStamp << " " << Twc.translation().transpose() << " " << Twc.so3().unit_quaternion().x()
               << " " << Twc.so3().unit_quaternion().y() << " " << Twc.so3().unit_quaternion().z() << " "
               << Twc.so3().unit_quaternion().w() << "\n";
    }
    myfile.close();
}

void FullSystem::printResultKitti(const string &filename, bool printOptimized)
{

    LOG(INFO) << "saving kitti trajectory..." << endl;

    unique_lock<mutex> lock(trackMutex);
    unique_lock<mutex> crlock(shellPoseMutex);
    ofstream f(filename);

    auto allKFs = globalMap->GetAllKFs();
    for (auto &fr : allKFs) {
        if (printOptimized == false) {
            SE3 Twc = fr->getPose().inverse();
            Mat33 Rwc = Twc.rotationMatrix();
            Vec3 twc = Twc.translation();
            f << fr->id << " " << setprecision(9) << Rwc(0, 0) << " " << Rwc(0, 1) << " " << Rwc(0, 2) << " "
              << twc(0) << " " << Rwc(1, 0) << " " << Rwc(1, 1) << " " << Rwc(1, 2) << " " << twc(1) << " "
              << Rwc(2, 0) << " " << Rwc(2, 1) << " " << Rwc(2, 2) << " " << twc(2) << endl;
        }
        else {
            Sim3 Swc = fr->getPoseOpti().inverse();
            Mat33 Rwc = Swc.rotationMatrix();
            Vec3 twc = Swc.translation();
            f << fr->id << " " << setprecision(9) << Rwc(0, 0) << " " << Rwc(0, 1) << " " << Rwc(0, 2) << " "
              << twc(0) << " " << Rwc(1, 0) << " " << Rwc(1, 1) << " " << Rwc(1, 2) << " " << twc(1) << " "
              << Rwc(2, 0) << " " << Rwc(2, 1) << " " << Rwc(2, 2) << " " << twc(2) << endl;
        }
    }
    f.close();

    LOG(INFO) << "done." << endl;
}
} // namespace ldso
