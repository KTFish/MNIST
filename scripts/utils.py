import random
import matplotlib.pyplot as plt


def visualize_sample_image(class_names, dataloader):
    ### Visualise an image
    image_batch, label_batch = next(iter(dataloader))

    # Get random image index
    idx = random.randint(0, 32)

    # Get single image from batch
    image, label = image_batch[idx], label_batch[idx]

    # Plot
    plt.imshow(image.permute(1, 2, 0), cmap="gray")
    plt.title(class_names[label])
    plt.show()
