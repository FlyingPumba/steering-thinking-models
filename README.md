# How to

## Requirements

- A workstation node with 24GB VRAM to load DeepSeek-R1-Distill-Llama-8B, i.e. 1x NVIDIA 3090
- API keys from [OpenAI Platform](https://platform.openai.com/api-keys) and [OpenRouter](https://openrouter.com/settings/keys)

## Steps

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
   4. Select Interpreter on VS Code

      Press `Shift+Command+P`, search for `Select Interpreter` and select `steering-env`

2. Use Python Interaective Window on VS Code

   1. Install Jupyter Extension
   2. Create Python Interactive Window

      Press `Shift+Command+P` and search for `Jupyter: Create Interactive Window`

3. Set up the `.env` file

   ```
   OPENAI_ORG_ID=get the org id from https://platform.openai.com/settings/organization/general
   OPENAI_API_KEY=get the api key from https://platform.openai.com/api-keys
   OPENROUTER_API_KEY=get the api key from https://openrouter.com/settings/keys
   ```

4. Set the limits in the [OpenAI Platform account](https://platform.openai.com/settings/organization/limits)

   Set it to $10.

5. Add balance to [OpenAI Platform account](https://platform.openai.com/settings/organization/billing/overview)

   Add $5.
   
   If you get the `insufficient_quota` error, click the `Cancel plan` button. Then, add another $5.

# Code for "Understanding Reasoning in Thinking Language Models via Steering Vectors"

Paper can be found [here](https://openreview.net/forum?id=OwhVWNOBcz)

⚠️ This is a temporary preview of the code used for the experiments. A more complete and polished version will be released in a couple weeks (Apr 25th)
