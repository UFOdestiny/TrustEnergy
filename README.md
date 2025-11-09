# TrustEnergy

**TrustEnergy: A Unified Framework for Accurate and Reliable User-level Energy Usage Prediction**  
TrustEnergy is a novel graph neural network for spatiotemporal energy prediction that not only forecasts mean values but also quantifies uncertainty. By integrating a Hierarchical Spatiotemporal Representation module and an innovative Sequential Conformalized Quantile Regression module, TrustEnergy effectively captures complex urban dynamics and achieves superior performance in both accuracy and uncertainty estimation across multiple real-world datasets.

## 🔧 Implementation Details
We conduct experiments on an Quad-Core 2.40GHz – Intel® Xeon X3220, 64 GB RAM linux computing server, equipped with an NVIDIA RTX A100 GPU with 24 GB memory. We adopt PyTorch 2.3.0 and CUDA 11.8 as the default deep learning library.

## 📁 Project Structure

```
├── experiments/trustenergy/  # Traning
├── src/base/           # Fundamental model and engine
├── src/engines/        # TrustEnergy's enginee
├── src/models/         # TrustEnergy's model
├── src/utils/          # Configuration and dataloader
└── README.md           # Project documentation
```

## 📊 Baselines  

The deterministic baselines are inplemented based on 
[STGCN](https://github.com/hazdzz/STGCN),
[GWNET](https://github.com/nnzhan/Graph-WaveNet), 
[ASTGCN](https://github.com/guoshnBJTU/ASTGCN-2019-pytorch),
[AGCRN](https://github.com/LeiBAI/AGCRN),
[StemGNN](https://github.com/microsoft/StemGNN), 
[DSTAGNN](https://github.com/SYLan2019/DSTAGNN), 
[PDFormer](https://github.com/BUAABIGSCity/PDFormer), 
[PowerPM](https://github.com/KimMeen/Time-LLM), 
[Chronos](https://github.com/amazon-science/chronos-forecasting), 
and [Moment](https://github.com/moment/moment).

The probabilistic baselines are inplemented based on 
[STZINB](https://github.com/ZhuangDingyi/STZINB), 
[DiffSTG](https://github.com/wenhaomin/DiffSTG),
and [DeepSTUQ](https://github.com/WeizhuQIAN/DeepSTUQ_Pytorch). 


## 📖 Citation

If you use this code, please cite the following paper:

```bibtex
@inproceedings{yu2026trust,
  title={TrustEnergy: A Unified Framework for Accurate and Reliable User-level Energy Usage Prediction},
  author={Yu, Dahai and Xu, Rongchao and Zhuang, Dingyi and Bu, Yuheng and Wang, Shenhao and Wang, Guang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```


*This project is part of the AAAI 2026 Social Impact Track.*