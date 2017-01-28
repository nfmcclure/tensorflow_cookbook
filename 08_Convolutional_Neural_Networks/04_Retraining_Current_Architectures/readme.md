Retraining (fine-tuning) Current CNN Architectures
==================================================

The purpose of the script provided in this section is to download the CIFAR-10 data, and sort it out in the proper folder structure for running it through the TensorFlow fine-tuning tutorial.  The script should create the following folder structure.

```
-train_dir
  |--airplane
  |--automobile
  |--bird
  |--cat
  |--deer
  |--dog
  |--frog
  |--horse
  |--ship
  |--truck
-validation_dir
  |--airplane
  |--automobile
  |--bird
  |--cat
  |--deer
  |--dog
  |--frog
  |--horse
  |--ship
  |--truck
```

After this is done, we proceed with the [TensorFlow fine-tuning tutorial](https://github.com/tensorflow/models/tree/master/inception).

