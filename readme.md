<h1 align="center">Multi-Resolution Mixer Network for Localization of Multiple Sensors from Cumulative Power Measurements</h1>
<hr style="border: 1px solid  #256ae2 ;">


```bibtex
@inproceedings{murad2025multi,
  title={Multi-Resolution Mixer Network for Localization of Multiple Sensors from Cumulative Power Measurements},
  author={Murad, Md Mahmuddun Nabi and Yirenya-Tawiah, Daniel Kwabena and Weller, Thomas and Yilmaz, Yasin},
  booktitle={2025 IEEE Wireless Communications and Networking Conference (WCNC)},
  pages={1--6},
  year={2025},
  organization={IEEE}
}

```

## Get started
Follow these steps to get started with our proposed model:
### 1. Install Requirements
Install Python 3.10 and the necessary dependencies.

```bash
pip install -r requirements.txt
```
### 2. Generating the datasets:
To generate the train, validation, and test datasets run the following script:
```bash
	bash ./scripts/dataGeneration/generate_datasets.sh
```
Running the abovementioned command will generate datasets in the subfolders of  ```./outputs/``` folder. Copy the datasets from the subfolders and paste them inside the folders ```./data/sensor/train/```, ```./data/sensor/test/```, and ```./data/sensor/vali/```.

 
### 3. Training and Evaluating the model:
To reproduce the results of the mixer model, use the following commands:
```
bash ./scripts/Mixer/mixer_test-1.sh
bash ./scripts/Mixer/mixer_test-2.sh
bash ./scripts/Mixer/mixer_test-3.sh
bash ./scripts/Mixer/mixer_test-4.sh
```
These commands will generate the results of the corresponding test datasets. The results will be found inside the ```./logs/BridgeModel/``` folder.

