# mind-reader
[NeurIPS2022] [Mind Reader: Reconstructing complex images from brain activities](https://nips.cc/Conferences/2022/Schedule?showEvent=53580) [[Slides](https://docs.google.com/presentation/d/1Fuff1QyC6rS0kNgQ_eQvmso0AN0WjfGATK6admCXp_0/edit?usp=sharing)]

**TL;DR**

We include additional text modality for reconstructing image stimuli from fMRI activities. Pretrained CLIP embedding space and a conditional generative model are utilized to counter data scarcity. 

**Pipeline overview**

![pipeline](https://user-images.githubusercontent.com/13376403/190880191-79dc3d2c-e631-4efd-92b3-1b954a5b7311.png)

**Code**

Mapping model is defined as ```fmri_clip_res``` in [```fmri_clip.py```](https://github.com/sklin93/mind-reader/blob/main/fmri_clip.py)

Conditional generative model is modified in StyleGAN2 folder, main files changed: [```train.py```](https://github.com/sklin93/mind-reader/blob/main/StyleGAN2/train.py), files under [training folder](https://github.com/sklin93/mind-reader/tree/main/StyleGAN2/training), and files in [torch_utils/ops](https://github.com/sklin93/mind-reader/tree/main/StyleGAN2/torch_utils/ops) because of compatibility issues (see [here](https://github.com/drboog/Lafite/pull/10/commits/91cd6c1dafed2c6f96010c3e348a450175507e3f)).

To use the trained models for generation, see this [notebook](https://colab.research.google.com/drive/1q6s9IWNkR6PsUP9NrdbiG7Euywuz6dd3?usp=sharing).

We used [wandb](https://wandb.ai/site) for hyperparameter tracking & tuning.

**Sample results**

For each two rows: top is ground truth, bottom is our reconstruction.

![5Tmh-2no44bjg34Iid9lR47aZDrZuFSuRmeiSLLIu9ybO_Td0LIUFz1yH14Y1CPwA2_gsJCq1J-QVT88rNl8XTQkQxH_J8CvfcWILZwEtX1w67Yc_VdNSsi4ZTJL](https://user-images.githubusercontent.com/13376403/190880268-125412b2-4716-40a3-844d-80ab80ea5b52.png)

![vCrha16VfRmH5D3-DN-iQKXJsHMI15PTd97mo1yKXAe5BD68-I5f-sRhlULI-TZo1EEpRr8PhdCQWCTBLkX7pxDVUX8BI-Ev2uh0NB1UUmELosad5ReuxibuFCZy](https://user-images.githubusercontent.com/13376403/190880271-7c7f0033-8ab2-40d4-8d82-b80dcd46b424.png)

![nVpS3he8E62YBfgU-XEHHqc2hWyIZ8siAW9DkPkMUSpfCzJaPZ_uCVJ_pZMtRUxswnhJsu-4WTOrLbmybfr9j2ceiEbIp4pn2WLOuDXZqdI-uxIdaoRHUwgLLknE](https://user-images.githubusercontent.com/13376403/190880272-a1793c52-2f79-4190-8d5c-b0e2866f165f.png)
