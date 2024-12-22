# Jarvis :robot:
---

Hello there :wave:

Jarvis is a personal assitant which can be run locally. 

This project is an illustration on how AI can today be made accessible to anyone, for free, while respecting data privacy (the data is saved on your machine and you're the only one to have access to it).

The only limitation might be your GPU: for good results I highly recommend a machine with 16GB of RAM minimum. If you have more, well... :rocket:

## Dependencies :hammer_and_wrench:
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

You should see the [web app](https://github.com/user-attachments/assets/99b5a970-e3c0-4dbd-8d8b-8f857c97cfd2).

[jarvis_llama](https://github.com/user-attachments/assets/7f759355-6211-459f-a09f-7126f0998856)

![jarvis_start](https://github.com/user-attachments/assets/6e3872a4-f0af-4f6f-bfbe-6399ac686772)