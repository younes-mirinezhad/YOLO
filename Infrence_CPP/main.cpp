#include <QCoreApplication>
#include "detector_tensorrt.h"
#include "spdlog/spdlog.h"
#include "qthread.h"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    
    // Need to use e2e engine file
    std::string detectorModelPath = "/media/chiko/HDD_1/Work/Pars_AI/Projects/InProgress/YOLOv8/Models/Detection/yolov8n_end2end.engine";
    auto inputSize = cv::Size(640, 640);
    std::vector<std::string> classNamesList = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    auto imgPath = "/media/chiko/HDD_2/Dataset/Coco/CocoImage/img.jpeg";
    bool useGPUMat{true};

    auto detector = new Detector_TensorRT;
    detector->setInputSize(inputSize);
    detector->setClassNames(classNamesList);
    auto detectorStatus = detector->LoadModel(detectorModelPath);
    if(!detectorStatus)
        return{};

    cv::Mat img;
    cv::cuda::GpuMat gImg;
    for (int idx = 0; idx < 100; ++idx) {
        QThread::msleep(50);

        img = cv::imread(imgPath);
        if (useGPUMat) {
            gImg.upload(img);
        }

        auto t1 = std::chrono::high_resolution_clock::now();

        Frames_Detection res;

        if (!useGPUMat) {
            // -------------------- Use cv::Mat
            res = detector->detect(img);
        } else {
            // -------------------- Use cv::cuda::GpuMat
            res = detector->detect(gImg);
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        res.detectionTime_ms = dt;

        spdlog::info("----------> detection time: {} ms", res.detectionTime_ms);

        img = cv::imread(imgPath);
        auto color_box = cv::Scalar(0, 0, 255);
        for (int i = 0; i < res.detections.size(); ++i) {
            spdlog::info("--------------------> Class: {} - Conf: {} - Box: [{}x{}] , [{}x{}]",
                          res.detections[i].className, res.detections[i].confidence,
                          res.detections[i].box.x, res.detections[i].box.y,
                          res.detections[i].box.width, res.detections[i].box.height );

            cv::rectangle(img, res.detections[i].box, color_box, 2, 8);
            cv::putText(img, res.detections[i].className,
                        cv::Point(res.detections[i].box.x, res.detections[i].box.y),
                        cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        }
        spdlog::info("------------------------------------------------------------");
        cv::imshow("Detection Box", img);
    }
    cv::waitKey(0);

    return a.exec();
}
