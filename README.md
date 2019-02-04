## Requirements
1) [Mbed cli] pip install mbed-cli
2) [uTensor cli] pip install utensor-cgen
3) [Treasure Data Client] pip install td-client
4) Install Tensorflow (1.9.0 is used in this project)

## Instructions
1) Import Mbed OS from this repo
```
mbed import https://www.github.com/BlackstoneEngineering/mbed-os-example-e2e-demo
```
2) Folder `models` already contrains C++ code for two models generated from uTensor. Don't need to train any model in Tensorflow. Move folder `models` to folder `mbed-os-example-e2e-demo`. 
3) In folder `mbed-os-example-e2e-demo`, `mbed add https://github.com/uTensor/uTensor/#c4250ffce8f5534b514eae1cf9a642f5a02f80d9`, this fetches uTensor library. 
4) Replace `main.cpp` in `mbed-os-example-e2e-demo` with `main.cpp` in this repo. In new `main.cpp`, a block of code is added at beginning of `int main(void)`to call uTensor models and return predictions. 
6) Add folder `.update-certificates` if there isn't one. 
```
mbed dm init -a mbed_cloud_api_key -d "http://os.mbed.com" --model-name "modelname" -q --force
```
6) Compile uTensor model with Mbed OS. 
```
mbed compile --target DISCO_L475VG_IOT01A --toolchain GCC_ARM --profile=uTensor/build_profile/release.json
```
7) Compile successfully! `mbed-os-example-e2e-demo.bin` is the binary. 
![alt text]()
7) Flash to the board and run it with `mbed sterm -b 115200` or in Serial terminal. <br />
8) Linear regression (lr model) and logistic regression (iris model) trained with synthesized data return accurate results! Should be able to see iris model returns an integer 1 and lr model returns a float number 5.99xxx. 
![alt text](https://github.com/moon412/mbed-dsc-e2e/blob/master/lr_output_011819.png)
9) Bug: can't register the device with Pelion.
```
[SMCC] Error occurred : MbedCloudClient::ConnectSecureConnectionFailed
[SMCC] Error code : 11
[SMCC] Error details : Client in reconnection mode SecureConnectionFailed
```
If compiling without uTensor and do `mbed compile --target DISCO_L475VG_IOT01A --toolchain GCC_ARM` without `release.json`, then there is no issue to connect and update with Pelion. 
