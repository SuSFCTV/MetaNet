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

   ![ParallelNet](https://github.com/SuSFCTV/MetaNet/blob/dev/docs/parallel_net.png)

   A picture is fed to the input, which goes to two encoders. The first one is a small CNN with a low number of parameters. The output will be a vector.

   The second one is a pre-trained model, from which I removed the classifier and froze the weights. At the output, I get a meta-information vector. 

   Then these two vectors are concatenated and fed to the classifier as input.
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