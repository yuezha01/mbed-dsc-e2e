## Requirements
1) [Mbed cli] pip install mbed-cli
2) [uTensor cli] pip install utensor-cgen
3) [Treasure Data Client] pip install td-client
4) Install Tensorflow (1.9.0 is used in this project)

## Instructions
1) Following the instructions and import Mbed OS and other libraries from this [repo](https://github.com/BlackstoneEngineering/mbed-os-example-e2e-demo/tree/master) 
```
mbed import https://www.github.com/BlackstoneEngineering/mbed-os-example-e2e-demo
```
Specifically,
First, download mbed os and `cd mbed-os-example-e2e-demo`
Second, configure mbed cloud api key
```
mbed config CLOUD_SDK_API_KEY 
```
Third, add folder `.update-certificates` if there isn't one. 
```
mbed dm init -a mbed_cloud_api_key -d "http://os.mbed.com" --model-name "modelname" -q --force
```
If need to fetch necessary libraries, `mbed deploy` 

2) `uTensor` is automatically downloaded from step 1 as well. It can also be manually added as following:
```
mbed add https://github.com/uTensor/uTensor
```
Or move `uTensor.lib` to the folder and `mbed deploy` 

3) Use Jupyter Notebooks in folder `tensorflow-models` to train models in Tensorflow and save models in .pb files. <br />

4) Generate embedded C++ code with utensor-cli and save them in folder `models`
```
utensor-cli convert ./tensorflow-models/mnist_model_0to9/deep_ml.pb --output-nodes=y_pred
```
5)  Replace `main.cpp` in `mbed-os-example-e2e-demo` with `main.cpp` in this repo.

6) First, compile model that classifies 0 to 4 with Mbed OS
```
mbed compile --target DISCO_F413ZH --toolchain GCC_ARM --profile=uTensor/build_profile/release.json --flash
```
Compile successfully!
![alt text](https://github.com/moon412/mbed-dsc-e2e/blob/master/compile_output_model0to4.png)

7) Flash to the board and run it with `mbed sterm -b 115200` or in Serial terminal. 

8) The DNN model classifies 0 to 4 correctly but the last 5 digits are predicted wrong. 
![alt text](https://github.com/moon412/mbed-dsc-e2e/blob/master/output_model0to4.png)

9) Copy and paste deep_mlp files from the folder `deep_mlp_0to9` and compile again.

10) Pelion update the firmware with the model
```
mbed dm update device -D 016a03d5c97d000000000001001002d8 -m DISCO_F413ZH --build ./BUILD/DISCO_F413ZH/GCC_ARM-RELEASE -vv
```
11) This time, the new model predicts all 10 digits correctly. 
![alt text](https://github.com/moon412/mbed-dsc-e2e/blob/master/output_model0to9.png)
