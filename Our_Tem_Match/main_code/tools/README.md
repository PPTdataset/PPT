# tools

### combine_images: 
Combine original image and fixed image and generate a whole image
```pythons
python combine_images.py --image_path path/to/original/and/fixed/images --saves_path path/to/generated/images
```

### constract_image: 
Constract original image and fixed image. 
output: a binary mask and a gray scale img where the posibility of defect increases along with the color from black to white 
```pythons
python combine_images.py --image_real path/to/real/image --image_fake path/to/fake/image --saves_path path/to/generated/images
```

### defect_generator
Generate defect images based on ok images and tc images
```pythons
python defect_generator.py
```

### json_generator
generate json file for mask
```pythons
python json_generator.py
```

### validate
Obtain offline scores
```pythons
python validate.py --pre_path path/to/json/we/generate --val_path path/to/json/of/ground/truth
```
