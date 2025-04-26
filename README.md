# How to

1. Use Conda

   1. Install Miniconda
      ```shell
      mkdir -p ~/miniconda3
      wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
      bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
      rm ~/miniconda3/miniconda.sh

      source ~/miniconda3/bin/activate

      conda init --all
      ```
   2. Create environment

      ```shell
      conda create --name steering-env python=3.11
      conda activate steering-env
      ```
   3. Install packages
      
      ```shell
      pip install -r requirements.txt
      ```
2. Use Python Interaective Window

   1. Install Jupyter Extension
   2. Create Python Interactive Window

      Press `Shift+Command+P` and search for `Jupyter: Create Interactive Window`

# Code for "Understanding Reasoning in Thinking Language Models via Steering Vectors"

Paper can be found [here](https://openreview.net/forum?id=OwhVWNOBcz)

⚠️ This is a temporary preview of the code used for the experiments. A more complete and polished version will be released in a couple weeks (Apr 25th)
