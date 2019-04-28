TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    image_process.cpp

PATH += /usr/local/bin

INCLUDEPATH += /usr/local/include
LIBS += -L/usr/local/lib
LIBS += -lopencv_dnn -lopencv_ml \
    -lopencv_objdetect -lopencv_shape -lopencv_stitching -lopencv_superres \
    -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_highgui \
    -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_imgproc -lopencv_flann -lopencv_core \
    -lpng

#QT_CONFIG -= no-pkg-config
CONFIG  += link_pkgconfig
PKGCONFIG += opencv

HEADERS += \
    image_process.h

