## Requirements
1) [Mbed cli] pip install mbed-cli
2) [uTensor cli] pip install utensor-cgen
3) [Treasure Data Client] pip install td-client
4) Install Tensorflow (1.9.0 is used in this project)

## Instructions
1) Import Mbed OS from this repo
```model
mbed import https://www.github.com/BlackstoneEngineering/mbed-os-example-e2e-demo
```
2) Use Jupyter Notebooks in folder `tensorflow-models` to train models in Tensorflow and save models in .pb files.
3) Generate embedded C++ code with utensor-cli and save them in folder `models`
```
utensor-cli convert ./tensorflow-models/lr_model/lr_model.pb --output-nodes=y_pred
```
4) Move `uTensor.lib` and folder `models` to folder `mbed-os-example-e2e-demo`. Replace `main.cpp` in `mbed-os-example-e2e-demo` with `main.cpp` in this repo.
5) `cd mbed-os-example-e2e-demo`. Run `mbed deploy`, this fetches the necessary libraries like uTensor
6) Add folder `.update-certificates` if there isn't one. 
```
mbed dm init -a mbed_cloud_api_key -d "http://os.mbed.com" --model-name "modelname" -q --force
```
6) Compile model with Mbed OS
```
mbed compile --target DISCO_L475VG_IOT01A --toolchain GCC_ARM --profile=uTensor/build_profile/release.json
```
7) Flash to the board and run it in Serial terminal. 
8) Linear regression (lr model) and logistic regression (iris model) trained with synthesized data return accurate results!
![alt text](https://github.com/moon412/mbed-dsc-e2e/blob/master/lr_output_011819.png)
9) Next step is to train linear or logistic regression with real data from sensors.
