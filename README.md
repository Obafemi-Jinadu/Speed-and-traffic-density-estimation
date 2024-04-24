## [Instant-level vehicle speed and traffic density estimation using deep neural network](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12526/125260J/Instant-level-vehicle-speed-and-traffic-density-estimation-using-deep/10.1117/12.2663643.short#_=_)

The repository for our published paper **Instant-level vehicle speed and traffic density estimation using deep neural network**: [Obafemi Jinadu](https://femi-jinadu.github.io/), [Victor Oludare](https://scholar.google.com/citations?user=RlbR3EgAAAAJ&hl=en), [Srijith Rajeev](https://scholar.google.com/citations?user=9vac4DkAAAAJ&hl=en), [Karen A. Panetta](https://scholar.google.com/citations?user=nsOodtAAAAAJ&hl=en), [Sos Agaian](https://scholar.google.com/citations?user=FazfMZMAAAAJ&hl=en)

My published work on speed and traffic density estimation in real time using work from the [Siamese Multi Object Tracking (SiamMOT) [1]](https://openaccess.thecvf.com/content/CVPR2021/papers/Shuai_SiamMOT_Siamese_Multi-Object_Tracking_CVPR_2021_paper.pdf) and achieves state of the art performance across diverse weather conditions. Demo videos can be found [here.](https://tufts.box.com/s/bsxwddo6g00y76wfcdof2lw9jdu59gm1)

## Abstract
The growing network of highway video surveillance cameras generates an immense amount of data that proves tedious for manual analysis. Automated real-time analysis of such data may provide many solutions, including traffic monitoring, traffic incident detection, and smart-city planning. More specifically, assessing traffic speed and density is critical in determining dynamic traffic conditions and detecting slowdowns, traffic incidents, and traffic alerts. However, despite several advancements, there are numerous challenges in estimating vehicle speed and traffic density, which are integral parts of ITS. Some of these challenges include variations in road networks, illumination constraints, weather, structure occlusion, and vehicle user-driving behavior. To address these issues, this paper proposes a novel deep learning-based framework for instant-level vehicle speed and traffic flow density estimation to effectively harness the potential of existing large-scale highway surveillance cameras to assist in real-time traffic analysis. This is achieved using the state-of-the-art region-based Siamese MOT network, SiamMOT which detects and associates object instances for multi-object tracking (MOT), to accurately estimate instant level vehicle speed in live video feeds. The UA-DETRAC [2] dataset is used to train the speed estimation model. Computer simulations show that the proposed framework a) allows the classifying of traffic density into light, medium, or heavy traffic flows, b) is robust to different types of road networks and illuminations without prior road information, and c) shows good performance when compared to current state-of-the-art methods using adequate performance metrics. 


## Pipeline
![image](https://github.com/Obafemi-Jinadu/Speed-and-traffic-density-estimation/blob/635e72b64bf718743fe210be4cf16cd3e1fbd793/speed%20est1.png)

![image](https://github.com/Obafemi-Jinadu/Speed-and-traffic-density-estimation/blob/4d4cc35fe3154b517598250ea0542cb8d3c50b23/speedest2.png)


## Results
 | Method | Scene | MAE| RMSE| R2|
| ---| --- | --- | --- |--- |
| Ours|Sunny| 1.016|2.341|0.528|
|  Ours  |Cloudy|0.571|1.012|0.700|
| Ours |Rainy|0.771|1.382|0.489|
| Ours  |Night|0.969|1.880|0.636|
|ECCNet [3]|Sunny|	2.850|	4.000|	-|
|ECCNet [3]|Cloudy|2.770|	3.510|	-|
|ECCNet [3]|Rainy|	3.180|	3.900|	-|
|ECCNet [3]|Night	|3.590	|4.480|	-|

a) cloudy - [full video here](https://tufts.app.box.com/s/bsxwddo6g00y76wfcdof2lw9jdu59gm1/file/1194038729961);
b)  rainy - [full video here](https://tufts.app.box.com/s/bsxwddo6g00y76wfcdof2lw9jdu59gm1/file/1194033842758);
c)  sunny - [full video here](https://tufts.app.box.com/s/bsxwddo6g00y76wfcdof2lw9jdu59gm1/file/1194070657007);
d)  night - [full video here](https://tufts.app.box.com/s/bsxwddo6g00y76wfcdof2lw9jdu59gm1/file/1193902776245)
![image](https://github.com/Obafemi-Jinadu/Speed-and-traffic-density-estimation/blob/e238c7dc532741c72eda3ac5a6791f2d75b95771/speedresult1.png)
![image](https://github.com/Obafemi-Jinadu/Speed-and-traffic-density-estimation/blob/f6f59e31b4a696f53410ecb9958cb223f8abde1c/spedestres2.png)

## References
[1] 	Shuai, B., Berneshawi, A., Li, X., Modolo, D. and Tighe, J., 2021. Siammot: Siamese multi-object tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 12372-12382). 

[2] S. Lyu et al., ‘UA-DETRAC 2018: Report of AVSS2018 IWT4S Challenge on Advanced Traffic Monitoring,’ in 2018 15th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS), Nov. 2018, pp. 1–6. doi: 10.1109/AVSS.2018.8639089. 

[3] Yu, C., Yang, J., Jiang, S., Zhang, Y., Li, H. and Du, L., 2022. ECCNet: Efficient chained centre network for real‐time multi‐category vehicle tracking and vehicle speed estimation. IET Intelligent Transport Systems, 16(11), pp.1489-1503. 



