## Caffe2 Utility Module
This module contains various methods that are generally used for organizing and training and inference.

## Requirements
1. Python 2.7
2. Ubuntu 16.04
3. Pre-installed caffe2.

## Usage
This module has to be downloaded before using the utility functions.
Note: Still working on developing this package.

Below are the utility functions supported currently:

```
write_db(db_type, db_name, features, labels)
convertRawDataToNCHWFormat(raw, num_channels, width, height)
AddInputLayer(model, batch_size, db, db_type)
AddAccuracy(model, softmax, label)
AddTrainingParameters(model, softmax, label)
AddBookKeepingOperators(model)
```

## Example:

```
import caffe2_utils as u

# Dataset conversions.
u.convertRawDataToNCHWFormat(<numpy_raw_features>, num_channels, width, height)
u.write_db(<db_format>, <db_name>, <numpy_features_nchw>, <numpy_labels>)

...

```

TODO: Need to add various features and convert this module into a pip package.
