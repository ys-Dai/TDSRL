# TDSRL: Time Series Dual Self-Supervised Representation Learning for Anomaly Detection from Different Perspectives

There is the code of our paper

## Dataset

We adopt three widely-used benchmarks in time series anomaly task to evaluate our network. We have placed the pre-processed datasets in folder `dataset/pocessed`.

The NeurIPS-TS Benchmark mentioned in our paper can be found at:

https://github.com/datamllab/tods/tree/benchmark/benchmark/synthetic


## Get Start

Install Python >= 3.8, PyTorch >= 1.9 with CUDA.

'''
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
'''


## Run the codes

You can train the network by running the file `train.py`, for example:

'''
python train.py --dataset=SWaT --TA_replacing=0.5 --num_FD_perturbation_positions=15
'''

After training, you will get the model parameters in the file `logs`, which is the default path.

If you need the evaluate the network, please select your dataset, trained model and corresponding parameters. Then run the file `evaluate.py`, for example:

'''
python evaluate.py --dataset=SWaT --model_dict=logs\Example_SWaT\Example_SWaT_model.pt --parameters_dict=logs\Example_SWaT\state\Example_SWaT.pt --window_sliding=16 
'''


## Reference

We appreciate the following github repos a lot for their valuable code:

https://github.com/Jhryu30/AnomalyBERT
