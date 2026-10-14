# Lab 1: Pneumonia Classification

Name: [your name]  
Student ID: [your ID]  
GitHub: https://github.com/Aurora-chang/Lab1

Replace all bracketed text after running the experiments. Do not submit this template as a finished report. Export the final document as `LAB1_yourstudentID_name.pdf`.

## 1. Introduction (5%)

[Explain the binary classification task, the dataset, and the goal of comparing two residual network architectures. Cite the dataset and ResNet paper.]

## 2. Experiment setups (25%)

### Models

ResNet18 uses basic residual blocks; ResNet50 uses bottleneck blocks. Both start from ImageNet V1 weights. Their original classifier is replaced with a newly initialized linear layer with two outputs, and all parameters are fine-tuned.

[Explain residual connections and architecture differences in your own words. Record actual parameter counts if you include them.]

### Data loader and preprocessing

The custom `ChestXrayDataset` opens images, converts them to RGB, transforms them, and returns tensors with labels. The default split is a seeded, stratified 80/20 split of training images; the original test split is held out. Training uses resizing to 224×224, ±10° rotation, brightness/contrast jitter, and ImageNet normalization. Validation/test omit random augmentation.

[Record actual sample counts per split/class from split_manifest.json. State whether you instead used --official-val. Explain the augmentation choices and limitations of image-level splitting.]

### Training configuration

[Record actual hardware, library versions, epochs, batch size, seed, optimizer, learning rate, weight decay, weighted cross-entropy, and checkpoint selection. Use config.json; do not assume defaults if changed.]

## 3. Experiment results (30%)

| Model | Selected epoch | Test accuracy (%) | Test F1 | Test precision | Test recall | Training time |
|---|---|---|---|---|---|---|
| ResNet18 | [actual] | [actual] | [actual] | [actual] | [actual] | [actual] |
| ResNet50 | [actual] | [actual] | [actual] | [actual] | [actual] | [actual] |

The final test metrics come from the checkpoint selected using validation F1. PNEUMONIA is the positive class. Accuracy = (TP+TN)/N, precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN).

[Insert screenshots of final test output for both models. Highlight the highest test accuracy and highest F1 among the compared models; these may belong to different models.]

![Training accuracy comparison](results/comparison/train_accuracy.png)
![Training F1 comparison](results/comparison/train_f1.png)
![Validation accuracy comparison](results/comparison/val_accuracy.png)
![Validation F1 comparison](results/comparison/val_f1.png)
![ResNet18 final test confusion matrix](results/resnet18/confusion_matrix.png)
![ResNet50 final test confusion matrix](results/resnet50/confusion_matrix.png)

[Describe measured trends. Confusion matrix rows=true, columns=predicted, class order NORMAL/PNEUMONIA.]

## 4. Discussion (35%)

Base the discussion on your actual curves and confusion matrices:

- Which model performed better, and how large was the measured difference?
- Did training and validation trends show overfitting? Point to particular epochs.
- How did class imbalance affect accuracy, recall, and F1?
- Which errors dominated: NORMAL predicted as PNEUMONIA or the reverse?
- Was ResNet50's extra training cost justified by the observed improvement?
- What limitations follow from one seed, the validation split, patient overlap, and this dataset?
- What would you test next? Clearly separate future ideas from completed experiments.

[Write your own evidence-based discussion here. If you claim augmentation or pretraining improved results, support the claim with an actual controlled comparison.]

## 5. GitHub link (5%)

https://github.com/Aurora-chang/Lab1

## References

- Lab1_spec_v2.pdf (course assignment).
- Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- He et al., Deep Residual Learning for Image Recognition: https://arxiv.org/abs/1512.03385
- Torchvision ResNet18: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- Torchvision ResNet50: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html
