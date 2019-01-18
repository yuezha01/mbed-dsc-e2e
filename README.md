## Requirements
1) [Mbed cli] pip install mbed-cli
2) [uTensor cli] pip install utensor-cgen
3) [Treasure Data Client] pip install td-client

## Instructions
1) Import Mbed OS from this repo
`mbed import https://www.github.com/BlackstoneEngineering/mbed-os-example-e2e-demo`
2. Import uTensor with uTensor.lib. 
`mbed deploy`
3. Train a model in Tensorflow and save model in .pb file using Jupyter Notebook
4. Generate embedded C++ code
`utensor-cli convert ./tensorflow-models/lr_model/lr_model.pb --output-nodes=y_pred
5. Replace `main.cpp` in `mbed-os-example-e2e-demo` with `main.cpp` in this repo.
6. Compile model with Mbed OS
`mbed compile --target DISCO_L475VG_IOT01A --toolchain GCC_ARM --profile=uTensor/build_profile/release.json`
6. Flash to the board and run it in Serial terminal. 