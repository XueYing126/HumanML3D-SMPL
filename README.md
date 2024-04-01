# SMPL HumanML3D: 3D Human Motion-Language Dataset

Follow this repository to produce SMPL version of [HumanML3D](https://github.com/XueYing126/HumanML3D-SMPL).

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

Download SMPL+H mode from [SMPL+H](https://mano.is.tue.mpg.de/download.php) (choose Extended SMPL+H model used in AMASS project) and DMPL model from [DMPL](https://smpl.is.tue.mpg.de/download.php) (choose DMPLs compatible with SMPL). Then place all the models under "./body_model/".

### 3. Download data
HumanML3D is a 3D human motion-language dataset that originates from a combination of HumanAct12 and Amass dataset. 

<details>
  <summary><b>Get AMASS Data</b></summary>

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

#### Unzip all datasets. In the bracket we give the name of the unzipped file folder. Please correct yours to the given names if they are not the same.

#### Place all files under the directory **./amass_data/**. The directory structure shoud look like the following:  
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

**Please make sure the file path are correct, otherwise it can not succeed.**
</details>


### 4. Process Data