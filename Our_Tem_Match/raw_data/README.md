You should put train and test data here and have the following directory & file structure:

```
Raw_data
├── Template
│   └── Image_1.bmp
│   └── Image_2.bmp
│   ...
│   └── Image_100.bmp
├── OK_img
│   └── 0a0dbecdc0b141618c0624406de9b3c3.bmp
│   ...
│   └── ffe350fb4000413cb26784673e1e52b2.bmp
├── TC_img
│   └── 000ZYVs38Gj08xgQ3yC1el0r4fv8J6.bmp
│   ...
│   └── ZzYLP01EkT1uySNYQO81w3g1mZ5u1b.bmp
```

where Template and OK_img are used for training and TC_img is used for testing.
In addition, OK_img is only used to guide the template slice position, and the actual image data does not participate in the training process.