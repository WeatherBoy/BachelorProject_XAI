# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# A utils for plots and tables
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# This program was created on Mon May 30 2022 by Felix Bo Caspersen (s183319), Mathematics and Technology - DTU

ORDERED_MODELS_PATHS = [
    "0001_plot_seresnet152_poorly_regularised",                                                                              
    "0002_plot_seresnet152_well_regularised",                                                                                
    "0003_plot_EfficientNet_b7_SecondAttempt_warm_restart_BIG2smallLR_weightDecay_1e6",                                           
    "0004_plot_Transfer_Learning_EffNet_b7_weight_decay_1e6_BIG2smallLR",
    "0005_plot_EfficientNet_b7_150_Epochs_weight_decay_1e5",
    "0006_plot_EfficientNet_b7_150_Epochs",
    "0007_plot_EfficientNet_b7_400_Epochs",
    "0008_plot_EfficientNet_b7_SecondAttempt_adam",
    "0009_plot_EfficientNet_b7_SecondAttempt_warm_restart_BIG2smallLR_weightDecay_1e6",
    "0010_plot_EfficientNet_b7_SecondAttempt_warm_restart",
    "0011_plot_EfficientNet_b7_SecondAttempt",
    "0012_plot_EfficientNet_b7",
    "0013_plot_EfficientNet_GBAR1",
    "0014_plot_main_cls_but_altered_by_Felix",
    "0015_plot_ResNet18_CIFAR100_with_validation",
    "0016_plot_torchAttack#3_ResNet18_CIFAR100_manxi_parameters_epoch150",
    "0017_plot_torchAttack#3_ResNet18_CIFAR100_manxi_parameters_epoch500",
    "0018_plot_Transfer_Learning_EffNet_b7_weight_decay_1e5_1To1e4LR",
    "0019_plot_Transfer_Learning_EffNet_b7_weight_decay_1e5_medium_LR",
    "0020_plot_Transfer_Learning_EffNet_b7_weight_decay_1e6_BIG2smallLR",
    "0021_plot_Transfer_Learning_EffNet_b7_weight_decay_1e6_bigToSmallLR_100_EPOCHS",
    "0022_plot_Transfer_Learning_EffNet_b7_weight_decay_1e6_mediumSmallLR",
    "0023_plot_Transfer_Learning_EffNet_b7_weight_decay_1e7_medium_LR",
    "0024_plot_Transfer_Learning_EffNet_b7_weight_decay_1e9_1To1e4LR",
    "0025_plot_Transfer_Learning_EffNet_b7",
    "0026_PLOT_efficientnet_b7_cifar100_warm_restart_batch_128_LR1e1_to_1e6_weightDecay_1e6_Epochs_300",
    "0027_plot_Transfer_Learning_EfficientNet_b7_ThirdAttempt_warm_restart_batch_128_LR1e1_to_1e5_weightDecay_1e6_Epochs_300"
]

def msg(
    message: str,
):
    """
    Input:
        message (str): a message of type string, which will be printed to the terminal
            with some decoration.

    Description:
        This function takes a message and prints it nicely

    Output:
        This function has no output, it prints directly to the terminal
    """

    # word_list makes sure that the output of msg is more readable
    sentence_list = message.split(sep="\n")
    # the max-function can apparently be utilised like this:
    longest_sentence = max(sentence_list, key=len)

    n = len(longest_sentence)
    n2 = n // 2 - 1
    print(">" * n2 + "  " + "<" * n2)
    print(message)
    print(">" * n2 + "  " + "<" * n2 + "\n")
