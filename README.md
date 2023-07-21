# Simulating Physics with Implicit Neural Spatial Representations

### [Project Page](https://www.cs.columbia.edu/cg/INSR-PDE/)  | [Paper](https://arxiv.org/abs/2210.00124)

<img src="https://github.com/honglin-c/INSR-PDE/blob/main/.github/images/teaser.png" width="500">

Official implementation for the paper:
> **[Simulating Physics with Implicit Neural Spatial Representations](https://www.cs.columbia.edu/cg/INSR-PDE/)**  
> [Honglin Chen*](https://www.cs.columbia.edu/~honglinchen/)<sup>1</sup>, [Rundi Wu*](https://www.cs.columbia.edu/~rundi/)<sup>1</sup>, [Eitan Grinspun](https://www.dgp.toronto.edu/~eitan/)<sup>2</sup>, [Changxi Zheng](http://www.cs.columbia.edu/~cxz/)<sup>1</sup>, [Peter Yichen Chen](https://peterchencyc.com/)<sup>3 </sup><sup>1</sup> <br>
> <sup>1</sup>Columbia University, <sup>2</sup>University of Toronto, <sup>3</sup>Massachusetts Institute of Technology <br>
> ICML 2023

## clone
```
cd /home/ubuntu/sgi/your_folder_name
git clone git@github.com:SGI-2023/INSR-hybrid.git
cd INSR-hybrid
```

## run program inside docker container
```
docker run --gpus all -it -v $HOME:/home/ubuntu sgi-2023
```
The above command will start a docker container.
Once you are in this container,
```
cd /home/ubuntu/sgi/your_folder_name/INSR-hybrid
bash scripts/xxx.sh
```

To detach from the container, you can find do: https://www.howtogeek.com/devops/how-to-detach-from-a-docker-container-without-stopping-it/

To re-enter the container
```
docker attach docker_id
```
docker_id is the number after root@ on your command line when you created the container.

Feel free to do whatever crazy things inside the container. No problem! But do not do anything crazy outside. However, you will need to **do version control (i.e., git pull/commit/push) outside the docker**.


## Experiments
Run each shell script under `scripts/` for the examples shown in the paper:
```bash
bash scripts/xxx.sh
```

For instance,
```bash
bash scripts/advect1D.sh
```

## Citation
```
@inproceedings{chenwu2023insr-pde,
    title={Simulating Physics with Implicit Neural Spatial Representations},
    author={Honglin Chen and Rundi Wu and Eitan Grinspun and Changxi Zheng and Peter Yichen Chen},
    booktitle={International Conference on Machine Learning},
    year={2023}
}
```
