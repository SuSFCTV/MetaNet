from models.meta_extractor.training_meta import training_meta_extractor
from models.parallel_net.training_parallelnet import training_parallel_net
from models.cnn.training_cnn_finetune import training_cnn_finetune
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="parallel_net", help="choose and type: "
                                                                          "parallel_net/meta_extractor/cnn")
    opt = parser.parse_args()
    if opt.model == "meta_extractor":
        training_meta_extractor()
    elif opt.model == "parallel_net":
        training_parallel_net()
    elif opt.model == "cnn":
        training_cnn_finetune()
