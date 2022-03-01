# MetaNet
Comparison of neural network training results with and without meta-information

## Explanation of the experiment
The task is to compare the results of training models with different uses of meta-information that comes from pictures.

I came up with three different models. I will tell you more about each of them:
1. CNN fine-tune pretrained model with backprop:

    The pre-trained resnet18, which I unfreezed the encoder layers and just train by the pictures.
2. Meta-extractor:

   The pre-trained resnet18, which we freeze the encoder layers and remove the gradient calculation from them. At the output of the encoder, we have a vector of meta-information that is fed to the classifier input.
3. Model with two parallel encoders which i called "ParallelNet".
   
   
## How to run an experiment
It uses a configurable script to start the pipeline of data processing and training

You must write the name of the selected model in the configurator.
You can choose 3 types of models: "cnn", "meta_extractor", "parallel_net"

```bash
--model="meta_extractor"
```
## Results
Meta-extractor : best val accuracy = 0.85

ParallelNet : best val accuracy = 0.92