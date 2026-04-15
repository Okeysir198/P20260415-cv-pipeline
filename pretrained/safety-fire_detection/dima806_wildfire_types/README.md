---
license: apache-2.0
metrics:
- accuracy
- f1
base_model:
- google/vit-base-patch16-224-in21k
---
Returns wildfire type given an image with about 90% accuracy.

See https://www.kaggle.com/code/dima806/wildfire-image-detection-vit for more details.

```
Classification report:

                                             precision    recall  f1-score   support

                        Both_smoke_and_fire     0.9623    0.9091    0.9350       253
                  Fire_confounding_elements     0.9306    0.8976    0.9138       254
Forested_areas_without_confounding_elements     0.9215    0.8780    0.8992       254
                 Smoke_confounding_elements     0.8370    0.8898    0.8626       254
                           Smoke_from_fires     0.8755    0.9409    0.9070       254

                                   accuracy                         0.9031      1269
                                  macro avg     0.9054    0.9031    0.9035      1269
                               weighted avg     0.9053    0.9031    0.9035      1269
```