# Download dataset

## CUB
For the `CUB_cuckoo`, `CUB_oriole`, and `CUB_vireo` tasks, you will download the CUB dataset introduced in [this webpage](https://authors.library.caltech.edu/records/cvm3y-5hh21).  
The dataset is available on [this webpage](https://data.caltech.edu/records/65de6-vp158).  
Put `CUB_200_2011` in the `datasets` folder and run `process_cub.py` to obtain train/val/test splits as well as to extract all relevant task and concept metadata into pickle files.  
Finally, arrange your CUB dataset folder to be:
```
CUB/
|-- CUB_raw/
|   |–– train.pkl
|   |–– val.pkl
|   |–– test.pkl
|-- images/
```

## iNaturalist
For the `iNat_butterfly` and `iNat_lupine` tasks, you will download the iNaturalist dataset introduced in the paper [The iNaturalist Species Classification and Detection Dataset](https://arxiv.org/pdf/1707.06642).  
The dataset is available on [this webpage](https://github.com/visipedia/inat_comp/tree/master/2021). We use `Train Mini Images` and `Validation Images`.  


## Stanford Dogs
For the `StanfordDogs_terrier` task, you will download the iNaturalist dataset introduced in the paper [Novel Dataset for Fine-Grained Image Categorization: Stanford Dogs](https://people.csail.mit.edu/khosla/papers/fgvc2011.pdf).  
The dataset is available on [this webpage](http://vision.stanford.edu/aditya86/ImageNetDogs/).


## VegFru
For the `vegfru_greens` and `vegfru_allium` tasks, you will download the VegFru dataset introduced in the paper [VegFru: A Domain-Specific Dataset for Fine-grained Visual Categorization](https://openaccess.thecvf.com/content_ICCV_2017/papers/Hou_VegFru_A_Domain-Specific_ICCV_2017_paper.pdf).  
The dataset is available on [this GitHub repository](https://github.com/ustc-vim/vegfru).  

Please ensure that the datasets are placed inside your `data_dir`.