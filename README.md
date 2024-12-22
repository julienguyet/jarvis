# Jarvis :robot:
---

Hello there :wave:

Jarvis is a personal assitant which can be run locally. :rocket:

This project is an illustration on how AI can today be made accessible to anyone, for free, while respecting data privacy (the data is saved on your machine and you're the only one to have access to it).

Jarvis is powered by **Llama 3.2** :llama: for conversation handling, while **Whisper** and **Parler** handle audio :loud_sound:

Note: all models have been chosen to fit on tiny GPUs. That said, for good results I highly recommend a machine with 16GB of RAM minimum. If you have more, feel free to explore more advanced models on the [Hugging Face Hub](https://huggingface.co/models)

## 1. Dependencies :hammer_and_wrench:
---

To get started, follow the below instructions. 

First, set up a virtual environment. Here we will be using Anaconda:
```
conda create --name jarvis python=3.10
conda activate jarvis
```

Once this is done, clone the repository and install requirements:
```
pip install -r requirements.txt
```

Finally, you will need to install the [ffmpeg package](https://en.wikipedia.org/wiki/FFmpeg) so your system can manage audio files. The easiest way to do so is by running (within your virtual environment):

```
conda install -c conda-forge ffmpeg
```

When done, please move to step 2. 

## 2. Run the Application :technologist:
---

At the root of the directory open the app.py file and add your hugging face [token](https://huggingface.co/docs/hub/security-tokens) on line 11:

```python
login(token='YOUR_TOKEN_HERE')
```

Then, in your terminal run:

```
python app.py
```

The very first start might take time as the models must be installed locally. Once it is done, you will see:

![terminal](https://github.com/user-attachments/assets/6e3872a4-f0af-4f6f-bfbe-6399ac686772)

Go the given URL to access the web app:

![jarvis_new_home](https://github.com/user-attachments/assets/6e3872a4-f0af-4f6f-bfbe-6399ac686772)

And that's it! The chat is yours to use. If you prefer, you can also record an audio and Jarvis will talk back to you!
