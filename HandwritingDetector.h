#ifndef __HANDWRITING_DETECTOR_H__
#define  __HANDWRITING_DETECTOR_H__
#include "mbed.h"
#include "stm32f413h_discovery.h"
#include "stm32f413h_discovery_ts.h"
#include "stm32f413h_discovery_lcd.h"
#include "tensor.hpp"
#include "image.h"
#include "models/deep_mlp.hpp"

//Really this should be a singleton, but I am lazy
class HandwritingDetector {

    private:
    template<typename T>
    void clear(Image<T>& img){
        for(int i = 0; i < img.get_xDim(); i++){
            for(int j = 0; j < img.get_yDim(); j++){
                img(i,j) = 0;
            }
        }
    }

    void run_inference(void);
    void poll_display_for_touch(void);

    public:
    template<typename T>
    void printImage(const Image<T>& img){
    
        for(int i = 0; i < img.get_xDim(); i++){
            for(int j = 0; j < img.get_yDim(); j++){
                printf("%f, ", img(i,j));
            }
            printf("]\n\r");
        }
    }

    HandwritingDetector();
    ~HandwritingDetector();
    void start();

    private:

    InterruptIn button;
    TS_StateTypeDef  TS_State;
    
    uint16_t x1, y1;
    Image<float>* img;
    Context ctx;
    EventQueue q;
    Thread t;

};


#endif
