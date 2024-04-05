# SMPL HumanML3D: 3D Human Motion-Language Dataset

Follow this repository to produce SMPL version of [HumanML3D](https://github.com/EricGuo5513/HumanML3D).

### 1. Setup environment

```sh
conda env create -f environment.yaml
conda activate torch_render
```
<details>
  <summary><b>In the case of installation failure, try  this</b></summary>

Remove the following lines from .yaml:
```
body-visualizer==1.1.0
configer==1.4.1
psbody-mesh==0.4
```
And install them manually:
```
pip install git+https://github.com/nghorbani/body_visualizer.git
pip install git+https://github.com/MPI-IS/configer
pip install git+https://github.com/MPI-IS/mesh.git
```
</details>

### 2. Download SMPL+H and DMPL model

Download SMPL+H mode from [SMPL+H](https://mano.is.tue.mpg.de/download.php) (choose Extended SMPL+H model used in AMASS project) and DMPL model from [DMPL](https://smpl.is.tue.mpg.de/download.php) (choose DMPLs compatible with SMPL). Then place all the models under "./body_models/".

### 3. Download data
HumanML3D is a 3D human motion-language dataset that originates from a combination of HumanAct12 and Amass dataset. 

#### Get AMASS Data

#### Download the following subdataset from [AMASS website](https://amass.is.tue.mpg.de/download.php). Note only download the <u>SMPL+H G</u> data.

* ACCD (ACCD)
* HDM05 (MPI_HDM05)
* TCDHands (TCD_handMocap)
* SFU (SFU)
* BMLmovi (BMLmovi)
* CMU (CMU)
* Mosh (MPI_mosh)
* EKUT (EKUT)
* KIT  (KIT)
* Eyes_Janpan_Dataset (Eyes_Janpan_Dataset)
* BMLhandball (BMLhandball)
* Transitions (Transitions_mocap)
* PosePrior (MPI_Limits)
* HumanEva (HumanEva)
* SSM (SSM_synced)
* DFaust (DFaust_67)
* TotalCapture (TotalCapture)
* BMLrub (BioMotionLab_NTroje)

In the bracket we give the name of the unzipped file folder.

Unzip all datasets. You could use `tools/unzip_amass.py`. 

 Place all files under the directory **./amass_data/**. 


<details>
  <summary><b>The expected directory structure</b></summary>
  
./amass_data/  
./amass_data/ACCAD/  
./amass_data/BioMotionLab_NTroje/  
./amass_data/BMLhandball/  
./amass_data/BMLmovi/   
./amass_data/CMU/  
./amass_data/DFaust_67/  
./amass_data/EKUT/  
./amass_data/Eyes_Japan_Dataset/  
./amass_data/HumanEva/  
./amass_data/KIT/  
./amass_data/MPI_HDM05/  
./amass_data/MPI_Limits/  
./amass_data/MPI_mosh/  
./amass_data/SFU/  
./amass_data/SSM_synced/  
./amass_data/TCD_handMocap/  
./amass_data/TotalCapture/  
./amass_data/Transitions_mocap/  

**Please make sure the file path are correct.**
</details>


### 4. Process Data

We follow the original HumanML3D to process the data (framerate, segment, mirror). 

#### Process HumanAct12
First unzip 'humanact12' in './pose_data'.

The following code will run [SMPLify](https://github.com/wangsen1312/joints2smpl) to get SMPL parameters from 3D joins.

```bash
python smplify_humanact12.py
# You can accelerate the process by running the same script in parallel simultaneously and utilizing multiple GPUs.
```
Move the generated './humanact12' in root to './pose_data/'.

#### Process HumanAct12 + AMASS
Run the following to process the data. 

```bash
# Downsample AMASS data to 20 fps, runtime about  
python s1_framrate.py

# Segment, Mirror, and Relocate
python s2_seg_augmentation.py

# generate 6D rotation representation
python s3_process_init.py

# Compute the mean, std
python s4_cal_mean_std.py
```

In the end, you should find the data you need at './HumanML3D/smpl/'.