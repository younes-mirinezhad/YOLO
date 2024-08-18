#include <QCoreApplication>
#include "EngineBuilder/enginebuilder.h"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    qDebug("---------- Start");

    auto onnxPath{"/media/chiko/HDD_1/Work/Pars_AI/Projects/InProgress/YOLOv8/Models/Detection/yolov8n_640_end2end.onnx"};
    auto enginePath{"/media/chiko/HDD_1/Work/Pars_AI/Projects/InProgress/YOLOv8/Models/Detection/yolov8n_640_end2end.engine"};
    EngineBuilder engineBuilder(onnxPath, enginePath);
    engineBuilder.buildEngine();

    qDebug("---------- End");

    return a.exec();
}
