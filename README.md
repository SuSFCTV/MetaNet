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
## Metrics
   I used the **accuracy** metric because we have the same number of images in both classes in the validation and training data.
## How to run an experiment
It uses a configurable script to start the pipeline of data processing and training
```bash
python main.py --model="parallel_net"
```


Or you can write the name of the selected model in the **main.py** configurator.
You can choose 3 types of models: "cnn", "meta_extractor", "parallel_net"

```bash
--model="meta_extractor"
```
## Results
You can check the loss graphs here:
[ссылочка](https://github.com/SuSFCTV/MetaNet/tree/dev/results)

**Meta-extractor** : best val accuracy = 0.85

**Fine-tune model** : best val accuracy = 0.88

**ParallelNet** : best val accuracy = 0.92

**Based on the results, it can be concluded that the use of images and meta-information at the same time has a good effect on the result**