from torch import nn
from typing import List


class Settings:

    def __init__(self):
        """
        TODO:
            calculate FLOPs in inference 
            settings.yaml
        """
        self._project_dir = "D:/self-studies/bachelors_final_project/research_py" # required to be hard coded since vs code changes the dir -> if i debug the dir is different from if i run from the terminal
        self._dataset = "omnifall"
        self._train = False
        self._test = True
        self._inference = False

        self._split_format = "cs-staged"
        self._ucf101_path = "E:/Datasets"
        self._inference_path = "E:/Datasets/fall_detection_inference"
        self._inference_save_dir = "D:/Datasets/fall_detection_inference_saves"
        self._inference_save_res = 500
        self._inference_target = 4 # None to visualise model predictions instead of a specific class
        self._disk_C = "C:/"
        self._disk_E = "E:/"
        self._omnifall_path = "Datasets/omnifall"
        self._omnifall_subsets = ["le2i", "GMDCSA24", "OOPS"]
        self._omnifall_corrupt_clips = ["Subject_1/Fall/14", "Subject_1/Fall/15",
                                        "falls/DontBeSuchaBaby-KidFailsSeptember2018_FailArmy21", "falls/DontGetZapped-ThrowbackThursdayAugust201789",
                                        "falls/DontRocktheBoat-ThrowbackFailsJuly201781", "falls/DoubleFailsNovember2017_FailArmy2",
                                        "falls/AreYouSerious-ThrowbackThursdaySeptember2017_FailArmy10", "falls/FailsoftheMonthFebruary2017_FailArmy37",
                                        "falls/FailsoftheWeek-BigAir_BiggerFailsMarch2017_FailArmy33", "falls/FailsoftheWeek-BigAir_BiggerFailsMarch2017_FailArmy39", # these do not exist in my files (deleted?)
                                        "falls/FailsoftheWeekMarch2017_FailArmy28", "falls/GuessitsTimetoLeave-ThrowbackThursdayOctober2017_FailArmy29",
                                        "falls/HopelessRomantic-FailsoftheWeekOctober2018_FailArmy5", "falls/ItsAllDownHillFromHere-ThrowbackFailsAugust201717",
                                        "falls/LetsGetIt-FailArmyAfterDarkep2170", "falls/LaughingCameraman-BestLaughsEverJanuary2017_FailArmy4",
                                        "falls/PolePosition-FailsoftheWeekSeptember2018_FailArmy17", "falls/WakeboardWipeout-FailsoftheWeekApril2019_FailArmy7",
                                        "falls/WheelieGoneWrong-FailsoftheWeekMay2018_FailArmy3", "falls/BeeKeeperBusiness-FailsoftheWeekNovember2018_FailArmy34"
                                        ]
        self._weights_path = "ablation_studies/baseline"
        self._dataset_labels = ["walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying", "stand_up", "standing", "other"]
        # applied after the weighting based on sample sizes
        self._label_weights = [1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self._apply_cls_weights = True
        self._work_model = "work"
        self._test_model = "baseline2"
        self._inference_model = "baseline2"
        self._inference_cam = "ScoreCAM"

        self._train_batch_size = 20
        self._val_batch_size = 20
        self._test_batch_size = 20
        self._image_size = 224
        self._video_length = 10 # frames

        self._criterion = "ce"
        self._self_adaptive_training = True
        self._TRADES = False # THE CODE IS UNTESTED DUE TO CUDA OUT OF MEMORY (6-year-old gpu)
        self._sce_alpha = 1
        self._sce_beta = 0.2
        self._trades_adversarial_factor = 0.001
        self._trades_steps = 1
        self._trades_step_size = 0.03
        self._trades_beta = 1.0
        self._trades_epsilon = 0.031
        self._sat_momentum = 0.9
        self._sat_start = 11
        self._sat_label_weights = True
        self._rnn_type = nn.LSTM # DO NOT INIT HERE
        self._frozen_layers = 3
        self._lstm_input_size = 80 
        self._lstm_hidden_size = 40
        self._lstm_num_layers = 1
        self._rnn_point_wise = True
        self._lstm_bias = True
        self._lstm_dropout_prob = 0.0
        self._lstm_bidirectional = False

        self._min_epochs = 20
        self._max_epochs = 100
        self._early_stop_tries = 10
        self._validation_interval = 2
        self._warmup_length = 4

        self._learning_rate = 0.001
        self._weight_decay = 0.0005
        self._label_smoothing = 0.0
        self._cls_weights_factor = 0.4
        self._cls_ignore_thresh = 10

        self._train_num_workers = 6
        self._val_num_workers = 2
        self._test_num_workers = 2
        self._amp = True # TODO: currently hardcoded to make training even possible
        self._async_transfers = True
        self._train_dev = "cuda:0"
        
        # credit for imagenet mean and stdev:
        #   https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
        self._mean = (0.485, 0.456, 0.406)
        self._standard_deviation = (0.229, 0.224, 0.225)

    @property
    def project_dir(self) -> str:
        return self._project_dir

    @property
    def train(self) -> bool:
        return self._train
    
    @train.setter
    def train(self, train: bool) -> None:
        self._train = train
    
    @property
    def test(self) -> bool:
        return self._test
    
    @test.setter
    def test(self, test: bool) -> None:
        self._test = test
    
    @property
    def inference(self) -> bool:
        return self._inference
    
    @inference.setter
    def inference(self, inference: bool) -> None:
        self._inference = inference
    
    @property
    def inference_path(self) -> str:
        return self._inference_path

    @property
    def inference_target(self) -> int:
        return self._inference_target
    
    @property
    def inference_cam(self) -> str:
        return self._inference_cam

    @inference_cam.setter
    def inference_cam(self, cam) -> None:
        self._inference_cam = cam
    
    @property
    def inference_save_dir(self) -> str:
        return self._inference_save_dir
    
    @property
    def inference_save_res(self) -> int:
        return self._inference_save_res
    
    @property
    def split_format(self) -> str:
        return self._split_format
    
    @property
    def dataset(self) -> str:
        return self._dataset
    
    @dataset.setter
    def dataset(self, dataset: str) -> None:
        self._dataset = dataset
    
    def disk(self, omnifall_subset: str) -> str:
        match omnifall_subset:
            case "GMDCSA24":
                return self._disk_C
            case "le2i":
                return self._disk_C
            case "mcfd":
                return self._disk_C
            case "OOPS":
                return self._disk_E
            case _:
                raise RuntimeError("Provide a valid subset")

    @property
    def dataset_path(self) -> str:
        match self.dataset:
            case "omnifall":
                return self._omnifall_path
            case "ucf101":
                return self._ucf101_path
            case _:
                raise RuntimeError("Provide a valid dataset")

    @property
    def omnifall_subsets(self) -> List[str]:
        return self._omnifall_subsets
    
    @property
    def omnifall_corrupt_clips(self) -> List[str]:
        return self._omnifall_corrupt_clips

    @property
    def weights_path(self) -> str:
        return self._weights_path
    
    @property
    def dataset_labels(self) -> List[str]:
        return self._dataset_labels
    
    @property
    def label_weights(self) -> List[float]:
        return self._label_weights
    
    @property
    def apply_cls_weights(self) -> bool:
        return self._apply_cls_weights
    
    @property
    def work_model(self) -> str:
        return self._work_model
    
    @property
    def test_model(self) -> str:
        return self._test_model
    
    @property
    def inference_model(self) -> str:
        return self._inference_model
    
    @property
    def train_batch_size(self) -> int:
        return self._train_batch_size
    
    @property
    def val_batch_size(self) -> int:
        return self._val_batch_size
    
    @property
    def test_batch_size(self) -> int:
        return self._test_batch_size
    
    @property
    def image_size(self) -> int:
        return self._image_size
    
    @property
    def video_length(self) -> int:
        return self._video_length

    @property
    def train_num_workers(self) -> int:
        return self._train_num_workers
    
    @property
    def val_num_workers(self) -> int:
        return self._val_num_workers

    @property
    def test_num_workers(self) -> int:
        return self._test_num_workers

    @property
    def mean(self) -> List[float]:
        return self._mean
    
    @property
    def standard_deviation(self) -> List[float]:
        return self._standard_deviation

    @property
    def rnn_type(self) -> nn.Module:
        return self._rnn_type
    
    @property
    def frozen_layers(self) -> int:
        return self._frozen_layers

    @property
    def criterion(self) -> str:
        return self._criterion
    
    @property
    def self_adaptive_training(self) -> bool:
        return self._self_adaptive_training
    
    @property
    def trades(self) -> bool:
        return self._TRADES
    
    @property
    def sce_alpha(self) -> float:
        return self._sce_alpha
    
    @property
    def sce_beta(self) -> float:
        return self._sce_beta
    
    @property
    def trades_adversarial_factor(self) -> float:
        return self._trades_adversarial_factor
    
    @property
    def trades_steps(self) -> int:
        return self._trades_steps
    
    @property
    def trades_step_size(self) -> float:
        return self._trades_step_size
    
    @property
    def trades_beta(self) -> float:
        return self._trades_beta
    
    @property
    def trades_epsilon(self) -> float:
        return self._trades_epsilon
    
    @property
    def sat_momentum(self) -> float:
        return self._sat_momentum
    
    @property
    def sat_start(self) -> int:
        return self._sat_start
    
    @property
    def sat_label_weights(self) -> bool:
        return self._sat_label_weights

    @property
    def lstm_input_size(self) -> int:
        return self._lstm_input_size
    
    @property
    def lstm_hidden_size(self) -> int:
        return self._lstm_hidden_size

    @property
    def lstm_num_layers(self) -> int:
        return self._lstm_num_layers

    @property
    def lstm_bias(self) -> bool:
        return self._lstm_bias

    @property
    def lstm_dropout_prob(self) -> float:
        return self._lstm_dropout_prob

    @property
    def rnn_point_wise(self) -> bool:
        return self._rnn_point_wise

    @property
    def lstm_bidirectional(self) -> bool:
        return self._lstm_bidirectional

    @property
    def min_epochs(self) -> int:
        return self._min_epochs
    
    @property
    def early_stop_tries(self) -> int:
        return self._early_stop_tries
    
    @property
    def max_epochs(self) -> int:
        return self._max_epochs
    
    @property
    def learning_rate(self) -> float:
        return self._learning_rate
    
    @property
    def weight_decay(self) -> float:
        return self._weight_decay
    
    @property
    def label_smoothing(self) -> float:
        return self._label_smoothing
    
    @property
    def cls_weight_factor(self) -> float:
        return self._cls_weights_factor
    
    @property
    def cls_ignore_thresh(self) -> int:
        return self._cls_ignore_thresh
    
    @property
    def validation_interval(self) -> int:
        return self._validation_interval
    
    @property
    def warmup_length(self) -> int:
        return self._warmup_length

    @property
    def amp(self) -> bool:
        return self._amp
    
    @property
    def async_transfers(self) -> bool:
        return self._async_transfers
    
    @property
    def train_dev(self) -> str:
        return self._train_dev


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")