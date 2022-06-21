# Data Intelligence Challenge 2022 - TU/e
A project that pitches an automated vaccumcleaner for larger appartment complexes.


## How to run
To run the implementation, we **highly** recommend using Google Colab as GYM and stable-baselines3 require many packages to be installed which are out-of-the-box avaialable in Colab.

Our submission contains both the report and the folder ```Data Intelligence Challenge``` which contains the entire codebase. The training and evaluation of the model are done from the Jupyter notebook ```DIC_ppo_assignment3v2.ipynb```.

### Google Colab (highly recommended)
To run the code on Google Colab, you should do the following things

1. Upload the folder ```Data Intelligence Challenge``` and it's contents to your Google Drive (this should automatically be in ```My Drive``` or ```Mijn Drive```).
2. Upload the Jupyter Notebook ```DIC_ppo_assignment3v2.ipynb``` to Google Colab which is located in ```Data Intelligence Challenge/2AMC15/```.
3. Run the first cell and connect your Google Drive to Colab
4. You are good to go!

*Troubleshooting tips:* If Colab can't find ```Data Intelligence Challenge/2AMC15/``` check that the uploaded folder is indeed located in ```My Drive``` and that ny double clicking on it that it's name is actually ```Data Intelligence Challenge```

### Local


## How to replicate experiments
To replicate the experiments in the report, either the model can be rerun manually in the training header or the pretrained models can be used in the evaluation header. The configurations of map, starting positions and battery sizes are provided at the top of both respective cells; the unused configuarations should be commented out.


```
# Settings to use
map = "tiny_map.json"
startPos = [[0.5, 1.6]]
battery_size = 20

# Commented out settings to not use
# map = "living_map.json"
# startPos = [[1, 2.8]]
# battery_size = 40

# map = "bunker_map.json"
# startPos = [[9.5, 1.6]]
# battery_size = 100

```

Due to the significant training time we recommend to use the pretrained model from the evaluation section

## Contributors
* Cas Teeuwen
* Max van Hoven
* Joris Smeets
* Lucas Snijder
* Maiko Bergman
* Tijs Teulings
