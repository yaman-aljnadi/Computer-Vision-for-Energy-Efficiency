# For YOLO LOGS AND CONFUSION MATRIX PLEASE REFER TO [YOLOv11](..Seg_Logs/Yolov11)

# Confusion Matrix of Bounding Boxes

## Detectron2_Aug_Original

### Confusion Matrix

|             | None | Garage | House | Other | Propane | Trailer | Background |
|-------------|------|--------|-------|-------|---------|---------|------------|
| **None**    | 0    | 0      | 0     | 0     | 0       | 0       | 0          |
| **Garage**  | 0    | 37     | 17    | 0     | 0       | 0       | 8          |
| **House**   | 0    | 13     | 244   | 2     | 0       | 2       | 8          |
| **Other**   | 0    | 1      | 7     | 10    | 0       | 0       | 0          |
| **Propane** | 0    | 0      | 0     | 0     | 150     | 1       | 44         |
| **Trailer** | 0    | 0      | 0     | 0     | 1       | 28      | 8          |
| **Background** | 0 | 22     | 29    | 10    | 28      | 8       | 0          |


## Detectron2_Aug_Greyscale

### Confusion Matrix

|             | None | Garage | House | Other | Propane | Trailer | Background |
|-------------|------|--------|-------|-------|---------|---------|------------|
| **None**    | 0    | 0      | 0     | 0     | 0       | 0       | 0          |
| **Garage**  | 0    | 41     | 12    | 1     | 0       | 0       | 8          |
| **House**   | 0    | 16     | 238   | 4     | 0       | 2       | 9          |
| **Other**   | 0    | 2      | 4     | 11    | 0       | 0       | 1          |
| **Propane** | 0    | 0      | 0     | 0     | 151     | 0       | 44         |
| **Trailer** | 0    | 0      | 0     | 0     | 0       | 31      | 6          |
| **Background** | 0 | 28     | 22    | 19    | 25      | 14      | 0          |

## Detectron2_Aug_Enhanced

### Confusion Matrix

|             | None | Garage | House | Other | Propane | Trailer | Background |
|-------------|------|--------|-------|-------|---------|---------|------------|
| **None**    | 0    | 0      | 0     | 0     | 0       | 0       | 0          |
| **Garage**  | 0    | 50     | 5     | 1     | 0       | 1       | 5          |
| **House**   | 0    | 19     | 237   | 5     | 0       | 3       | 5          |
| **Other**   | 0    | 3      | 4     | 11    | 0       | 0       | 0          |
| **Propane** | 0    | 0      | 0     | 1     | 152     | 0       | 42         |
| **Trailer** | 0    | 0      | 2     | 0     | 0       | 32      | 3          |
| **Background** | 0 | 37     | 46    | 17    | 25      | 12      | 0          |


## Detectron2_Aug_Enhanced_Greyscale

### Confusion Matrix

|             | None | Garage | House | Other | Propane | Trailer | Background |
|-------------|------|--------|-------|-------|---------|---------|------------|
| **None**    | 0    | 0      | 0     | 0     | 0       | 0       | 0          |
| **Garage**  | 0    | 40     | 14    | 1     | 0       | 0       | 5          |
| **House**   | 0    | 14     | 241   | 2     | 0       | 1       | 11         |
| **Other**   | 0    | 0      | 5     | 9     | 0       | 0       | 1          |
| **Propane** | 0    | 0      | 0     | 0     | 151     | 1       | 15         |
| **Trailer** | 0    | 0      | 0     | 0     | 0       | 31      | 6          |
| **Background** | 0 | 27     | 30    | 15    | 30      | 10      | 0          |


## MMDetection_Aug_Original

### Confusion Matrix

|             | None | Garage | House | Other | Propane | Trailer | Background |
|-------------|------|--------|-------|-------|---------|---------|------------|
| **None**    | 0    | 0      | 0     | 0     | 0       | 0       | 0          |
| **Garage**  | 0    | 39     | 14    | 0     | 0       | 1       | 8          |
| **House**   | 0    | 8      | 251   | 2     | 0       | 1       | 7          |
| **Other**   | 0    | 2      | 5     | 11    | 0       | 0       | 0          |
| **Propane** | 0    | 0      | 0     | 0     | 143     | 1       | 51         |
| **Trailer** | 0    | 0      | 1     | 0     | 0       | 31      | 5          |
| **Background** | 0 | 21     | 31    | 14    | 34      | 9       | 0          |



## MMDetection_Aug_Greyscale

### Confusion Matrix

|             | None | Garage | House | Other | Propane | Trailer | Background |
|-------------|------|--------|-------|-------|---------|---------|------------|
| **None**    | 0    | 0      | 0     | 0     | 0       | 0       | 0          |
| **Garage**  | 0    | 37     | 18    | 0     | 0       | 2       | 5          |
| **House**   | 0    | 10     | 245   | 2     | 0       | 2       | 10         |
| **Other**   | 0    | 1      | 6     | 9     | 0       | 0       | 2          |
| **Propane** | 0    | 0      | 0     | 0     | 144     | 1       | 50         |
| **Trailer** | 0    | 0      | 0     | 0     | 0       | 37      | 0          |
| **Background** | 0 | 24     | 38    | 16    | 51      | 11      | 0          |



# Confusion Matrix of Segmentation

## Detectron2_Aug_Original

### Confusion Matrix

|             | None | Garage | House | Other | Propane | Trailer | Background |
|-------------|------|--------|-------|-------|---------|---------|------------|
| **None**    | 0    | 0      | 0     | 0     | 0       | 0       | 0          |
| **Garage**  | 0    | 38     | 17    | 0     | 0       | 0       | 7          |
| **House**   | 0    | 13     | 244   | 2     | 0       | 2       | 8          |
| **Other**   | 0    | 1      | 7     | 10    | 0       | 0       | 0          |
| **Propane** | 0    | 0      | 0     | 0     | 149     | 0       | 46         |
| **Trailer** | 0    | 0      | 0     | 0     | 1       | 28      | 8          |
| **Background** | 0 | 21     | 29    | 10    | 29      | 9       | 0          |

## Detectron2_Aug_Original_Greyscale

### Confusion Matrix

|             | None | Garage | House | Other | Propane | Trailer | Background |
|-------------|------|--------|-------|-------|---------|---------|------------|
| **None**    | 0    | 0      | 0     | 0     | 0       | 0       | 0          |
| **Garage**  | 0    | 42     | 12    | 1     | 0       | 0       | 7          |
| **House**   | 0    | 15     | 238   | 4     | 0       | 2       | 10         |
| **Other**   | 0    | 2      | 4     | 12    | 0       | 0       | 0          |
| **Propane** | 0    | 0      | 0     | 0     | 149     | 0       | 46         |
| **Trailer** | 0    | 0      | 0     | 0     | 0       | 31      | 6          |
| **Background** | 0 | 28     | 22    | 18    | 27      | 14      | 0          |

## Detectron2_Aug_Enhanced

### Confusion Matrix

|             | None | Garage | House | Other | Propane | Trailer | Background |
|-------------|------|--------|-------|-------|---------|---------|------------|
| **None**    | 0    | 0      | 0     | 0     | 0       | 0       | 0          |
| **Garage**  | 0    | 51     | 5     | 1     | 0       | 1       | 4          |
| **House**   | 0    | 19     | 236   | 5     | 0       | 3       | 6          |
| **Other**   | 0    | 3      | 4     | 11    | 0       | 0       | 0          |
| **Propane** | 0    | 0      | 0     | 0     | 152     | 0       | 43         |
| **Trailer** | 0    | 0      | 2     | 0     | 0       | 32      | 3          |
| **Background** | 0 | 36     | 47    | 18    | 25      | 12      | 0          |


## Detectron2_Aug_Enhanced_Greyscale

### Confusion Matrix

|             | None | Garage | House | Other | Propane | Trailer | Background |
|-------------|------|--------|-------|-------|---------|---------|------------|
| **None**    | 0    | 0      | 0     | 0     | 0       | 0       | 0          |
| **Garage**  | 0    | 42     | 14    | 1     | 0       | 0       | 5          |
| **House**   | 0    | 14     | 241   | 2     | 0       | 1       | 11         |
| **Other**   | 0    | 0      | 5     | 12    | 0       | 0       | 1          |
| **Propane** | 0    | 0      | 0     | 0     | 178     | 0       | 17         |
| **Trailer** | 0    | 0      | 0     | 0     | 0       | 31      | 6          |
| **Background** | 0 | 35     | 39    | 23    | 75      | 13      | 0          |



## MMDetection_Aug_Original

### Confusion Matrix

|             | None | Garage | House | Other | Propane | Trailer | Background |
|-------------|------|--------|-------|-------|---------|---------|------------|
| **None**    | 0    | 0      | 0     | 0     | 0       | 0       | 0          |
| **Garage**  | 0    | 39     | 14    | 0     | 0       | 1       | 8          |
| **House**   | 0    | 8      | 251   | 2     | 0       | 1       | 7          |
| **Other**   | 0    | 2      | 5     | 11    | 0       | 0       | 0          |
| **Propane** | 0    | 0      | 0     | 0     | 145     | 0       | 50         |
| **Trailer** | 0    | 0      | 1     | 0     | 0       | 31      | 5          |
| **Background** | 0 | 21     | 31    | 14    | 32      | 10      | 0          |


## MMDetection_Aug_Greyscale

### Confusion Matrix

|             | None | Garage | House | Other | Propane | Trailer | Background |
|-------------|------|--------|-------|-------|---------|---------|------------|
| **None**    | 0    | 0      | 0     | 0     | 0       | 0       | 0          |
| **Garage**  | 0    | 37     | 18    | 0     | 0       | 2       | 5          |
| **House**   | 0    | 10     | 244   | 2     | 0       | 2       | 11         |
| **Other**   | 0    | 1      | 6     | 9     | 0       | 0       | 2          |
| **Propane** | 0    | 0      | 0     | 0     | 144     | 0       | 51         |
| **Trailer** | 0    | 0      | 0     | 0     | 0       | 36      | 1          |
| **Background** | 0 | 24     | 39    | 16    | 51      | 13      | 0          |
