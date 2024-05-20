### RUN WITH SPECIFIC VGA (2nd)
SINGULARITYENV_CUDA_VISIBLE_DEVICES=1 singularity shell --nv imc24.sif
#SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 singularity shell --nv imc24.sif
#SINGULARITYENV_CUDA_VISIBLE_DEVICES=1 singularity run --nv imc24.sif
### Install LightGlue
# git clone https://github.com/cvg/LightGlue.git
# cd LightGlue
# python3 -m pip install -e .

### Install Hierarchical-Localization
### cd /scratch/hcquach
### git clone https://github.com/quachconghoang/glue-factory.git
### cd ../glue-factory
python3 -m pip install -e .

### check data folder
### check ... homography folder
### check ... megadepth folder
