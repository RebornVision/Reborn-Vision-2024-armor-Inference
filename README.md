**北京科技大学Reborn战队RM2024装甲板识别推理全开源**

## 背景介绍
适配[已开源训练网络](https://github.com/gaoxinstudent/RM-Armor-Detect.git)模型的C++推理代码及数字分类补充

 ## 视频效果

 见文件

2024赛季团队实测在intel 13th i7-13620H平台上，单本地视频读取推理5.5ms-6ms，170-180FPS；

实战基于RV框架下进行推理和其他全流程处理，延时11-12ms，83-90FPS，开启多线程推理，未开启异步推理

 ## 环境配置
 我们团队的训练配置和推理配置如下
***
| 硬件设备 |推理设备                                |
| -------- | ----------------------|
| CPU      | intel 13th i7-13620H<br />(雷神MIX NUC) |
| GPU      | \                                       |
| 内存     | 16GB DDR4                               |
| 环境配置 |推理设备                                |
| OS       | Ubuntu 22.04                            |
| CUDA     |  \                                       |
| OpenVINO |  2023.3                                  |
| Python   | 3.9                                     |
| Pytorch  | \                                       |
| OpenCV | >= 4.6.0|

### 注意事项

神经网络模型只识别红蓝及装甲板灯条四点，用RV的处理思路，去装甲板中间区域ROI进行数字分类

## 开启推理

### clion
没什么说明，记得查看模型及视频demo文件路径
### 非集成开发环境
进入文件夹  
```mkdir build```   
```cd build```  
```cmake ..```  
```make```  
```./INFERENCE```  

## 最后

本套程序为个人开源，代码极为粗糙，不代表团队水平。

本项目持续更新

如果有帮到你，请给个star，谢谢！ 

  ## 联系方式
  + 作者微信：wxid_30fyana8r4rc22
  + 作者邮箱：gaoxin_student@outlook.com
  + 团队邮箱：reborn_vision@163.com