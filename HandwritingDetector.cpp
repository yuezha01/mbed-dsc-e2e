#include "HandwritingDetector.h"

void HandwritingDetector::run_inference(void) {

    Image<float> smallImage = resize(*img, 28, 28);

    pc.printf("Done padding\n\n");
    delete img;

    pc.printf("Reshaping\n\r");
    smallImage.get_data()->resize({1, 784});
    pc.printf("Creating Graph\n\r");

    get_deep_mlp_ctx(ctx, smallImage.get_data());
    pc.printf("Evaluating\n\r");
    ctx.eval();
    S_TENSOR prediction = ctx.get({"y_pred:0"});
    int result = *(prediction->read<int>(0,0));

    printf("Number guessed %d\n\r", result);

    BSP_LCD_Clear(LCD_COLOR_WHITE);
    BSP_LCD_SetTextColor(LCD_COLOR_BLACK);
    BSP_LCD_SetFont(&Font24);

    // Create a cstring
    uint8_t number[2];
    number[1] = '\0';
    //ASCII numbers are 48 + the number, a neat trick
    number[0] = 48 + result;
    BSP_LCD_DisplayStringAt(0, 120, number, CENTER_MODE);
    trigger_inference = false;
    img = new Image<float>(240, 240);
    clear(*img);
}

void HandwritingDetector::poll_display_for_touch(void) {
    if(TS_State.touchDetected) {
        /* One or dual touch have been detected          */

        /* Get X and Y position of the first touch post calibrated */
        x1 = TS_State.touchX[0];
        y1 = TS_State.touchY[0];

        img->draw_circle(x1, y1, 7); //Screen not in image x,y format. Must transpose

        BSP_LCD_SetTextColor(LCD_COLOR_GREEN);
        BSP_LCD_FillCircle(x1, y1, 5);

    }
}
HandwritingDetector::HandwritingDetector() : button(USER_BUTTON), TS_State({0}) {
    img = new Image<float>(240, 240);
    BSP_LCD_Init();
    button.rise(q.event(this, &HandwritingDetector::run_inference)); // Use button to trigger inference
    /* Touchscreen initialization */
    if (BSP_TS_Init(BSP_LCD_GetXSize(), BSP_LCD_GetYSize()) == TS_ERROR) {
        printf("BSP_TS_Init error\n");
    }


    /* Clear the LCD */
    BSP_LCD_Clear(LCD_COLOR_WHITE);

    clear(*img);

    q.call_every(5 /*ms*/, this, &HandwritingDetector::poll_display_for_touch);
}

HandwritingDetector::~HandwritingDetector() { 
    delete img;
}

void HandwritingDetector::start(){
    t.start(callback(&q, &EventQueue::dispatch_forever));
}
