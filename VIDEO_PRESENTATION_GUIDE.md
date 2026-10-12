# Video Presentation Guide

A speaking plan for an explanatory video of about **10 minutes** (within the 8-12 minute target). Each row says what to show, what to say in your own words, and the technical points to make sure you mention. Speak from the charts; do not read the notebook aloud.

| Fact to quote | Value from this run |
|---|---|
| Images / classes | 2,443 images, 4 classes |
| Class sizes | Glioma 805, Meningioma 545, No Tumor 483, Pituitary 610 |
| Mirrored pairs / split across supplied folders | 352 / 164 |
| New split (train / val / test) | 1,704 / 372 / 367 |
| Selection scores (validation, worst shift) | EfficientNetB0 0.922, MobileNetV2 0.918, Custom CNN 0.645 |
| Test macro F1 under the worst shift | Custom CNN 0.650, MobileNetV2 0.927, EfficientNetB0 0.912 |
| Custom CNN - test accuracy / macro F1 | 0.817 / 0.798 |
| MobileNetV2 - test accuracy / macro F1 | 0.940 / 0.940 |
| EfficientNetB0 - test accuracy / macro F1 | 0.916 / 0.912 |
| Selected model | MobileNetV2 (fine-tuned) |
| Selected model - weakest class | Meningioma (F1 0.881) |
| Missed tumors / false alarms on the test set | 0 of 294 / 3 of 73 |
| Errors caught by the 70% confidence threshold | 6 of 22 |

### Speaking script

| # | Section | Time | Show on screen | What to explain | Technical points to hit |
|---|---|---|---|---|---|
| 1 | Introduction | 0:30 | Notebook title, introduction | "I'm Sahil. This project classifies brain MRI slices into glioma, meningioma, pituitary tumor or no tumor, and deploys the model in a web app." | Assistive tool, not a diagnostic device |
| 2 | Problem statement | 0:30 | Problem statement, use-case table | "Reading MRI takes specialist time. A classifier could help sort the queue or give a second opinion. Those are possible uses - I did not test them clinically." | Multi-class, single-label image classification |
| 3 | Dataset | 0:50 | Dataset loading output, data-quality chart | "I didn't hard-code anything. The loader finds the classes and counts the files, then decodes every image to check for corruption and duplicates." | Counts are computed; all images one size; grayscale stored as RGB |
| 4 | EDA | 1:30 | Charts 1, 3, 6, 9, 12 | "Two findings shaped the project. First, brightness and framing differ by class, so a model could cheat by learning scan style. Second, the dataset contains mirrored copies of images, and the supplied split put the two copies in different folders." | Class imbalance; shortcut risk; **data leakage** and how it was detected with image hashes |
| 5 | Preprocessing | 0:50 | Sections 6.5, 6.6, 6.8 | "Every image is converted to RGB, resized to 224 by 224 and divided by 255. I rebuilt the split as 70/15/15 so mirrored pairs stay together, and the notebook asserts there is no overlap." | One preprocessing function for training and the app; grouped stratified split; class weights; why not SMOTE |
| 6 | Data augmentation | 0:40 | Chart 17, augmentation table | "Augmentation is only on training data. I kept it mild because tumor differences are subtle. I used horizontal flip because I predict tumor type, not location, and no vertical flip." | Training only; each augmentation has a reason |
| 7 | Custom CNN | 0:50 | CNN code, `model.summary()` | "Four convolution blocks with batch norm, ReLU and pooling, then global average pooling, dropout and softmax. I kept it under a million parameters because the dataset is small." | Role of each layer; why GAP instead of Flatten |
| 8 | Transfer learning | 1:00 | `build_transfer_model`, stage descriptions | "MobileNetV2 and EfficientNetB0 start from ImageNet features. Stage one trains only the new head with the backbone frozen. Stage two unfreezes the top layers with a ten times smaller learning rate." | Why freeze first; why low learning rate; BatchNorm kept frozen |
| 9 | Training | 0:40 | Training curves | "EarlyStopping watches validation loss and stops when it stops improving; ModelCheckpoint keeps the best epoch; ReduceLROnPlateau lowers the learning rate. Here is where training and validation separate." | Overfitting visible as a gap; best epoch restored |
| 10 | Evaluation | 1:00 | Classification reports, confusion matrix, per-class chart | "I report precision, recall and F1 per class, not only accuracy. This row shows which class is confused with which. The error I care most about is a tumor predicted as no tumor." | Macro F1; recall for tumor classes; test set used once |
| 11 | Model comparison | 0:50 | Comparison table and charts | "I chose the model on validation data, using its worst macro-F1 after small brightness and contrast shifts, because my EDA showed brightness is tied to the class. Look at the robustness chart: this is what happens to a model that learned the shortcut. I did not pick on test accuracy." | Robust selection score; shortcut learning; size and speed trade-off |
| 12 | Explainability | 0:40 | Grad-CAM figure | "Grad-CAM shows which regions pushed the model towards its answer. It is an indication, not a tumor outline." | What Grad-CAM is; its limits; head-focus check |
| 13 | Streamlit application | 1:00 | Live app: upload a sample image | Upload an image from `assets/sample_images`, show the prediction, the probabilities, the low-confidence warning and the Grad-CAM toggle. Then upload a non-image file to show the error message. | Model loaded once (cached); same preprocessing; no stack traces |
| 14 | Results | 0:30 | Conclusion table | State the selected model and its test accuracy, macro F1 and weakest class from the table above. | Quote the computed numbers exactly |
| 15 | Limitations | 0:30 | Responsible AI section | "One public dataset, one split, no patient IDs, uncalibrated confidence, and known acquisition bias. It has not been validated clinically." | Honest limits |
| 16 | Conclusion | 0:20 | Conclusion | "I built a leak-free pipeline, compared three models fairly and deployed one. Next steps would be external validation and calibration." | Close with the disclaimer |

**Total: about 10 minutes.**

**Delivery tips.** Record the Streamlit demo after starting the app once, so the model is already cached. Point at charts with the cursor. When you quote a number, have that table visible.

### Questions to be ready for

| Question | Short answer in your own words |
|---|---|
| Why a CNN? | Images have spatial structure. Convolutions reuse the same small filter everywhere, so they detect a pattern wherever it appears and need far fewer parameters than a fully connected network. |
| Why transfer learning? | The dataset is small. A network pretrained on ImageNet already knows edges, textures and shapes, so only the higher-level features have to adapt. |
| Why these pretrained models? | MobileNetV2 and EfficientNetB0 are small enough to train on a CPU and to serve in a web app. ResNet50 and InceptionV3 are about ten times larger, which is risky with under two thousand training images. |
| Why 224 x 224? | It is the resolution the pretrained models were trained at, and it keeps enough detail while making CPU training feasible. The originals are square, so nothing is distorted. |
| Why these augmentations? | Small rotations, shifts and zoom mimic differences in head position; brightness and contrast mimic scanner settings and weaken the brightness shortcut found in the EDA. No vertical flip because upside-down heads are not realistic. |
| Why normalise? | Inputs between 0 and 1 keep gradients well-behaved and match what the layers expect. Each model rescales internally to its backbone's range. |
| Why this split? | 70/15/15 gives enough training data and a few hundred images each for validation and test. It is stratified by class and grouped so mirrored copies never cross subsets. |
| Why not use the supplied split? | Mirrored copies of the same image were in different folders, which is data leakage and would inflate the scores. |
| How was class imbalance handled? | Class weights in the loss, computed from training labels. Not SMOTE, because blending two MRI images pixel by pixel does not give a realistic scan. |
| Why is accuracy insufficient? | With unequal classes a model can score well overall and still be weak on one class, and accuracy treats a missed tumor like any other error. |
| Why do precision, recall and F1 matter? | Recall counts missed cases, precision counts false alarms, and F1 balances them. Macro F1 gives each class equal weight. |
| What does the confusion matrix tell us? | Which specific classes are mixed up, and in particular whether tumors are being predicted as no tumor. |
| Why did one model outperform another? | State what the table shows. Pretrained models bring general features; the custom CNN learns from scratch on little data. I measured the difference; I did not prove the cause. |
| Why not just pick the highest validation score? | Because a model can get a high score by reading scan brightness, which differs by class in this dataset. I re-scored every model after shifting brightness and contrast slightly and selected on the worst case. An unaugmented CNN that looked best on clean images failed that test in my first run. |
| What is overfitting? | The model memorises training images instead of learning general patterns: training accuracy keeps rising while validation performance stalls or gets worse. |
| How did EarlyStopping help? | It stops when validation loss has not improved for several epochs and restores the best weights, so I keep the model from before it started overfitting. |
| Why fine-tune? | The top layers of an ImageNet model are specialised for everyday objects. Unfreezing them at a low learning rate lets them adapt to MRI without destroying what they know. |
| What does Grad-CAM show? | A coarse heatmap of the regions that most increased the score of the predicted class. It is not a segmentation and not proof of correct reasoning. |
| What are the model's limitations? | Small single-source dataset, acquisition bias between classes, no patient-level split, 2-D slices only, uncalibrated confidence, no external validation. |
| Why not use it for autonomous diagnosis? | It makes errors, some with high confidence; it has not been clinically validated or approved; and a wrong answer can harm a patient. A qualified professional must make the decision. |
| How does Streamlit use the trained model? | It loads the saved `.h5` file once with caching, runs the uploaded image through the same preprocessing function as training, calls `predict`, and displays the class and probabilities. |
