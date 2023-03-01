# Updated Setup Instructions
Run `conda env create -f setup.yml` and then `conda activate cs224n_dfp1`

Or to update, `conda env update --file setup.yml`

# CS 224N Default Final Project - Multitask BERT

This is the starting code for the default final project for the Stanford CS 224N class. You can find the handout [here](https://web.stanford.edu/class/cs224n/project/default-final-project-bert-handout.pdf)

In this project, you will implement some important components of the BERT model to better understanding its architecture. 
You will then use the embeddings produced by your BERT model on three downstream tasks: sentiment classification, paraphrase detection and semantic similarity.

After finishing the BERT implementation, you will have a simple model that simultaneously performs the three tasks.
You will then implement extensions to improve on top of this baseline.

## Setup instructions

* Follow `setup.sh` to properly setup a conda environment and install dependencies.
* There is a detailed description of the code structure in [STRUCTURE.md](./STRUCTURE.md), including a description of which parts you will need to implement.
* You are only allowed to use libraries that are installed by `setup.sh`, external libraries that give you other pre-trained models or embeddings are not allowed (e.g., `transformers`).

## Handout

Please refer to the handout for a through description of the project and its parts.

### Acknowledgement

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

# Git Best Practices
1) Get your local main branch up to date with master

<img width="415" alt="Screen Shot 2022-02-18 at 4 59 48 PM" src="https://user-images.githubusercontent.com/61396391/221059696-5e1b93c2-23a4-4ff3-99a5-f6fe43d25b27.png">

2) Create a new branch (title it <name>/<topic>) 

<img width="519" alt="Screen Shot 2022-02-18 at 5 00 49 PM" src="https://user-images.githubusercontent.com/61396391/221059941-9f4d06c4-9f40-480a-bb60-fdfdd505e29e.png">

3) Make whatever changes...

Go crazy! 💪

4) Commit your changes!

<img width="726" alt="Screen Shot 2022-02-18 at 5 02 04 PM" src="https://user-images.githubusercontent.com/61396391/221059978-264a6b55-e950-4ad2-90d4-f2801e2cb11d.png">


5) Push your changes and copy the pull request link

<img width="550" alt="Screen Shot 2022-02-18 at 5 04 10 PM" src="https://user-images.githubusercontent.com/61396391/221059996-2b406d17-3328-4002-9b85-c7b853b78dac.png">


6) Pull up the link in GitHub. Add some comments to the pull request, and some screen shots / videos of what you built! Then click “Create pull request”! 🙌

<img width="1543" alt="Screen Shot 2022-02-18 at 5 04 55 PM" src="https://user-images.githubusercontent.com/61396391/221060026-2fa170aa-5f97-454e-89ff-ece21f9fd68d.png">


