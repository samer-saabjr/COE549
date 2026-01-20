# Installation

## 1. Installing Conda

If you already have Miniconda or Anaconda installed, you may skip this step.

You can install Miniconda for your platform by following the instructions at the following link: 

https://www.anaconda.com/docs/getting-started/miniconda/install

We recommend you install Miniconda as opposed to the full Anaconda platform because it is a faster download, but Anaconda will work too.

## 2. Creating Conda environment

After installing Conda, run the following command:
```bash
conda create -n llms_course python==3.11.4
```

## 3. Installing Dependencies

Once you have activated your Conda environment, run the following to install the project:
```bash
conda activate llms_course
pip install -e .
```
This will install the project as a Python library on your system.

Note, the following command is incorrect:
```bash
pip install -r requirements.txt # Incorrect, use `pip install -e .` instead
```

## 4. Verify Install

You should now be able to run the following in Python without causing an error:
```python
import llms_course
```

# API Key Setup

To run an LLM, you will need an API key. We will be using Google Gemini (free tier). 

## 1. Getting The API Key

You can acquire a free API key at the following link:

https://aistudio.google.com/app/apikey

It should look something like:
```
AIzaSyDPu...
```

## 2. Add API Key To Project

Rename your `.env.template` file to `.env`, then set the `GOOGLE_API_KEY` variable to your API key.

Your `.env` should look something like:
```bash
GOOGLE_API_KEY=AIzaSyDPu...
```

Ensure there are no spaces around the `=` sign.