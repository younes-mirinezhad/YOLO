#ifndef DETECTOR_TENSORRT_H
#define DETECTOR_TENSORRT_H

#include <QObject>
#include <opencv2/opencv.hpp>
#include <NvInferRuntime.h>

struct Binding {
    size_t size = 1;
    size_t dsize = 1;
    nvinfer1::Dims dims;
    char* name;
};
struct PreParam {
    float ratio  = 1.0f;
    float dw     = 0.0f;
    float dh     = 0.0f;
    float height = 0;
    float width  = 0;
};
struct DetectedObject {
    int classID;
    std::string className;
    float confidence;
    cv::Rect box;
};
struct Frames_Detection
{
    int detectionTime_ms{-1};
    std::vector<DetectedObject> detections;
};

using Severity = nvinfer1::ILogger::Severity;
class TrtLogger : public nvinfer1::ILogger {
public:
    void setLogSeverity(Severity severity);
private:
    void log(Severity severity, const char *msg) noexcept override;
    Severity m_severity = Severity::kINFO;
};

class Detector_TensorRT : public QObject
{
public:
    explicit Detector_TensorRT(QObject *parent = nullptr);
    ~Detector_TensorRT();

    void setClassNames(std::vector<std::string> newClassNamesList);
    void setInputSize(cv::Size newInputSize);
    void setEnd2End(bool newStatus);

    bool LoadModel(std::string &modelPath);

    Frames_Detection detect(cv::Mat &srcImg);
    Frames_Detection detect(cv::cuda::GpuMat &srcImg);

private:
    void copy_from_Mat(cv::Mat &img, cv::Size &size);
    void letterbox(cv::Mat &img_input, cv::Size &size);
    void copy_from_Mat(cv::cuda::GpuMat &gImg, cv::Size &size);
    void letterbox(cv::cuda::GpuMat& gImg_input, cv::Size &size);
    void blobFromGpuMat(const cv::cuda::GpuMat& gImg_input, const std::array<float, 3>& std,
                        const std::array<float, 3>& mean, bool swapBR, bool normalize);
    Frames_Detection postprocess_normal();
    Frames_Detection postprocess_end2end();

    std::vector<std::string> _classNamesList;
    cv::Size _inputSize = cv::Size(640, 640);
    bool _end2end{false};
    bool _modelIsLoaded{false};
    nvinfer1::IExecutionContext* context{nullptr};
    cudaStream_t stream{nullptr};
    int num_inputs{0};
    int num_outputs{0};
    std::vector<Binding> output_bindings;
    std::vector<void*> device_ptrs;
    std::vector<void*> host_ptrs;
    cv::cuda::GpuMat _gBlob;
    cv::Mat _blob;
    PreParam pparam;
};

#endif // DETECTOR_TENSORRT_H
